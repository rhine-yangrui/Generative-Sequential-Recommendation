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


def _decode_one(output_seq, sid_to_item, sid_array, item_id_list, k, seen_items):
    """从一条 beam 输出还原 item_id；负责合法性检查、Hamming 兜底、去重。"""
    new_tokens = output_seq[1:1 + len(K_LEVELS)].tolist()
    if len(new_tokens) < len(K_LEVELS):
        return None
    try:
        candidate_sid = tokens_to_semantic_id(new_tokens)
        if any(not (0 <= c < kk) for c, kk in zip(candidate_sid, K_LEVELS)):
            return None
    except Exception:
        return None
    item_id = sid_to_item.get(candidate_sid)
    if item_id is None:
        item_id = hamming_nearest(candidate_sid, sid_array, item_id_list, seen_items)
    return item_id


def predict_topk_batch(model, history_seqs, semantic_ids, sid_to_item,
                       sid_array, item_id_list, k=10, beam_width=50, device='cpu'):
    """
    批量版 predict_topk：一次 T5 generate 处理多个用户的历史。

    Args:
        history_seqs: List[List[item_id]]，每个用户一条历史
    Returns:
        List[List[item_id]]，长度 = len(history_seqs)，每条 ≤ k
    """
    model.eval()
    token_lists = [seq_to_t5_tokens(h, semantic_ids, maxlen=20) for h in history_seqs]
    input_ids   = torch.tensor(token_lists, dtype=torch.long, device=device)
    attn_mask   = (input_ids != PAD_TOKEN).long()

    constrained = LevelConstrainedLogitsProcessor(K_LEVELS, LEVEL_OFFSETS)
    num_ret     = min(beam_width, k * 5)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_length=len(K_LEVELS) + 1,
            num_beams=beam_width,
            num_return_sequences=num_ret,
            early_stopping=False,
            pad_token_id=PAD_TOKEN,
            logits_processor=LogitsProcessorList([constrained]),
        )

    B = len(history_seqs)
    outputs = outputs.view(B, num_ret, -1)

    results = []
    for b in range(B):
        recs = []
        seen = set()
        for j in range(num_ret):
            item_id = _decode_one(outputs[b, j], sid_to_item, sid_array,
                                  item_id_list, k, seen)
            if item_id is None or item_id in seen:
                continue
            recs.append(item_id)
            seen.add(item_id)
            if len(recs) >= k:
                break
        results.append(recs)
    return results


def predict_topk(model, history_seq, semantic_ids, sid_to_item,
                 sid_array, item_id_list, k=10, beam_width=50, device='cpu'):
    """单用户版本，内部调用 predict_topk_batch。"""
    return predict_topk_batch(
        model, [history_seq], semantic_ids, sid_to_item,
        sid_array, item_id_list, k=k, beam_width=beam_width, device=device,
    )[0]
