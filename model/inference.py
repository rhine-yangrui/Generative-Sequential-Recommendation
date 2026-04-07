"""
Beam Search 推理：给定用户历史，生成 top-k 推荐 item。

流程：
  1. 历史序列 → token 序列
  2. GPT-2 beam search 生成完整 Semantic ID 的新 token
  3. 还原成 semantic_id，做范围合法性检查
  4. 查反向索引找对应 item（精确匹配 or Hamming 距离最近邻）
"""

import torch
import numpy as np
from transformers import LogitsProcessor, LogitsProcessorList

from model.tokenizer import (
    seq_to_tokens, tokens_to_semantic_id, PAD_TOKEN, K_LEVELS, LEVEL_OFFSETS
)


class LevelConstrainedLogitsProcessor(LogitsProcessor):
    """
    在 beam search 每一步强制只生成当前层合法的 token。

    生成步骤：
      step d -> 只允许第 d 层对应范围内的 token
    """
    def __init__(self, input_len, k_levels, level_offsets):
        self.input_len = input_len
        self.k_levels = k_levels
        self.level_offsets = level_offsets

    def __call__(self, input_ids, scores):
        gen_step = input_ids.shape[1] - self.input_len
        level = gen_step % len(self.k_levels)

        mask = torch.full_like(scores, float('-inf'))
        start = self.level_offsets[level]
        end = start + self.k_levels[level]
        mask[:, start:end] = 0.0
        return scores + mask


def build_reverse_index(semantic_ids):
    """
    构建 semantic_id → item_id 的反向索引，用于 beam search 后查找 item。

    Returns:
        sid_to_item:  dict，semantic_id -> item_id
        sid_array:    np.array，shape (N, D)，所有 item 的 semantic_id 矩阵
        item_id_list: list，和 sid_array 行对齐的 item_id
    """
    sid_to_item  = {tuple(int(x) for x in sid): iid for iid, sid in semantic_ids.items()}
    item_id_list = list(semantic_ids.keys())
    sid_array    = np.array([semantic_ids[iid] for iid in item_id_list], dtype=np.int32)
    return sid_to_item, sid_array, item_id_list


def hamming_nearest(candidate_sid, sid_array, item_id_list, exclude_ids=None):
    """
    当 beam search 生成的 semantic_id 没有精确匹配时，
    用 Hamming 距离找最近邻 item。
    """
    cand      = np.array(candidate_sid, dtype=np.int32)
    distances = (sid_array != cand).sum(axis=1)

    if exclude_ids:
        for i, iid in enumerate(item_id_list):
            if iid in exclude_ids:
                distances[i] = 999

    return item_id_list[distances.argmin()]


def predict_topk(model, history_seq, semantic_ids, sid_to_item,
                 sid_array, item_id_list, k=10, beam_width=50, device='cpu'):
    """
    给定用户历史序列，用 beam search 生成 top-k 推荐。

    Args:
        model:        训练好的 GPT-2 模型
        history_seq:  用户历史 item_id 列表（不含 target）
        semantic_ids: item_id -> semantic_id
        sid_to_item:  semantic_id -> item_id 反向索引
        sid_array:    (N, D) 所有 item 的 semantic_id 矩阵
        item_id_list: 和 sid_array 对齐的 item_id 列表
        k:            返回 top-k 个推荐
        beam_width:   beam search 宽度（建议 ≥ k*3）

    Returns:
        recommended_items: list of item_id，长度 ≤ k
    """
    model.eval()
    input_tokens = seq_to_tokens(history_seq, semantic_ids, maxlen=20)
    input_ids    = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
    input_len    = len(input_tokens)

    constrained_processor = LevelConstrainedLogitsProcessor(
        input_len=input_len,
        k_levels=K_LEVELS,
        level_offsets=LEVEL_OFFSETS,
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=len(K_LEVELS),
            num_beams=beam_width,
            num_return_sequences=min(beam_width, k * 5),
            early_stopping=False,
            pad_token_id=PAD_TOKEN,
            logits_processor=LogitsProcessorList([constrained_processor]),
        )

    recommended = []
    seen_items  = set()

    for output in outputs:
        new_tokens = output[len(input_tokens):].tolist()
        if len(new_tokens) < len(K_LEVELS):
            continue

        try:
            candidate_sid = tokens_to_semantic_id(new_tokens[:len(K_LEVELS)])
            # 合法性检查：每层的值必须在对应码本范围内
            if any(not (0 <= code < k) for code, k in zip(candidate_sid, K_LEVELS)):
                continue
        except Exception:
            continue

        # 精确匹配
        item_id = sid_to_item.get(candidate_sid)

        # 无精确匹配 → Hamming 最近邻
        if item_id is None:
            item_id = hamming_nearest(candidate_sid, sid_array, item_id_list, seen_items)

        if item_id not in seen_items:
            recommended.append(item_id)
            seen_items.add(item_id)

        if len(recommended) >= k:
            break

    return recommended
