"""
Beam Search 推理（T5 encoder-decoder）：给定用户历史，生成 top-k 推荐 item。

流程：
  1. 历史序列 → 拍平 token 序列（左 PAD 到固定长度）
  2. T5 encoder 编码 → decoder beam search 生成 4 个 Semantic ID token
  3. 还原成 semantic_id，做范围合法性检查
  4. 查反向索引找对应 item（精确匹配 or Hamming 距离最近邻）
"""

import torch
import numpy as np
from transformers import LogitsProcessor, LogitsProcessorList

from model.tokenizer import (
    seq_to_t5_tokens, tokens_to_semantic_id, PAD_TOKEN, K_LEVELS, LEVEL_OFFSETS
)


class LevelConstrainedLogitsProcessor(LogitsProcessor):
    """
    在 T5 decoder beam search 每一步强制只生成当前层合法的 token。

    decoder_input_ids 起始长度 = 1（仅 decoder_start_token），
    第一次预测对应 level 0，后续依次。
    """
    def __init__(self, k_levels, level_offsets):
        self.k_levels = k_levels
        self.level_offsets = level_offsets

    def __call__(self, input_ids, scores):
        gen_step = input_ids.shape[1] - 1
        if gen_step >= len(self.k_levels):
            return scores
        level = gen_step
        mask  = torch.full_like(scores, float('-inf'))
        start = self.level_offsets[level]
        end   = start + self.k_levels[level]
        mask[:, start:end] = 0.0
        return scores + mask


def build_reverse_index(semantic_ids):
    """
    构建 semantic_id → item_id 反向索引。

    Returns:
        sid_to_item:  dict, semantic_id -> item_id
        sid_array:    np.array (N, D)
        item_id_list: list 与 sid_array 行对齐
    """
    sid_to_item  = {tuple(int(x) for x in sid): iid for iid, sid in semantic_ids.items()}
    item_id_list = list(semantic_ids.keys())
    sid_array    = np.array([semantic_ids[iid] for iid in item_id_list], dtype=np.int32)
    return sid_to_item, sid_array, item_id_list


def hamming_nearest(candidate_sid, sid_array, item_id_list, exclude_ids=None):
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
    T5 beam search 生成 top-k 推荐。

    Args:
        model:        训练好的 T5 模型
        history_seq:  用户历史 item_id 列表（不含 target）
        semantic_ids: item_id -> semantic_id
        sid_to_item:  semantic_id -> item_id
        sid_array:    (N, D) 全部 semantic_id 矩阵
        item_id_list: 与 sid_array 对齐的 item_id 列表
        k:            返回 top-k
        beam_width:   beam 宽度

    Returns:
        list of item_id, 长度 ≤ k
    """
    model.eval()
    input_tokens   = seq_to_t5_tokens(history_seq, semantic_ids, maxlen=20)
    input_ids      = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = (input_ids != PAD_TOKEN).long()

    constrained = LevelConstrainedLogitsProcessor(K_LEVELS, LEVEL_OFFSETS)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=len(K_LEVELS) + 1,   # +1 for decoder_start_token
            num_beams=beam_width,
            num_return_sequences=min(beam_width, k * 5),
            early_stopping=False,
            pad_token_id=PAD_TOKEN,
            logits_processor=LogitsProcessorList([constrained]),
        )

    recommended = []
    seen_items  = set()

    for output in outputs:
        # 跳过 decoder_start_token (位置 0)
        new_tokens = output[1:1 + len(K_LEVELS)].tolist()
        if len(new_tokens) < len(K_LEVELS):
            continue

        try:
            candidate_sid = tokens_to_semantic_id(new_tokens)
            if any(not (0 <= code < kk) for code, kk in zip(candidate_sid, K_LEVELS)):
                continue
        except Exception:
            continue

        item_id = sid_to_item.get(candidate_sid)
        if item_id is None:
            item_id = hamming_nearest(candidate_sid, sid_array, item_id_list, seen_items)

        if item_id not in seen_items:
            recommended.append(item_id)
            seen_items.add(item_id)

        if len(recommended) >= k:
            break

    return recommended
