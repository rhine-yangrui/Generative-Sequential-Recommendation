"""
Constrained beam-search inference for the T5 generative recommender.

Steps:
  1. encode the flat history token sequence
  2. beam-decode 4 Semantic ID tokens, restricting each step to the legal
     token range for that level
  3. look up the resulting (c0, c1, c2, c3) in the reverse index; drop
     any beam whose tuple does not correspond to a real item
"""

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from model.tokenizer import (
    K_LEVELS,
    LEVEL_OFFSETS,
    PAD_TOKEN,
    seq_to_t5_tokens,
    tokens_to_semantic_id,
)


class LevelConstrainedLogitsProcessor(LogitsProcessor):
    """Mask out every token that does not belong to the current level."""

    def __init__(self, k_levels, level_offsets):
        self.k_levels = k_levels
        self.level_offsets = level_offsets

    def __call__(self, input_ids, scores):
        # decoder_input_ids starts with decoder_start_token; the first prediction
        # corresponds to level 0, so gen_step = #generated so far.
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
    Build a (semantic_id -> item_id) lookup table plus aligned arrays for
    debugging / analysis.
    """
    sid_to_item  = {tuple(int(x) for x in sid): iid for iid, sid in semantic_ids.items()}
    item_id_list = list(semantic_ids.keys())
    sid_array    = np.array([semantic_ids[iid] for iid in item_id_list], dtype=np.int32)
    return sid_to_item, sid_array, item_id_list


def _decode_beam(output_seq, sid_to_item):
    """Recover an item_id from one beam output, or None if invalid / unknown."""
    new_tokens = output_seq[1:1 + len(K_LEVELS)].tolist()
    if len(new_tokens) < len(K_LEVELS):
        return None
    candidate_sid = tokens_to_semantic_id(new_tokens)
    if any(not (0 <= c < kk) for c, kk in zip(candidate_sid, K_LEVELS)):
        return None
    return sid_to_item.get(candidate_sid)


def predict_topk_batch(model, history_seqs, semantic_ids, sid_to_item,
                       sid_array, item_id_list, k=10, beam_width=50, device='cpu'):
    """
    Batched constrained beam search: one ``model.generate`` call covers all
    histories in ``history_seqs``.

    Returns a list of length ``len(history_seqs)``; each element is a list of
    at most ``k`` recommended item ids.
    """
    del sid_array, item_id_list  # accepted for backwards-compatible call sites

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
            item_id = _decode_beam(outputs[b, j], sid_to_item)
            if item_id is None or item_id in seen:
                continue
            recs.append(item_id)
            seen.add(item_id)
            if len(recs) >= k:
                break
        results.append(recs)
    return results
