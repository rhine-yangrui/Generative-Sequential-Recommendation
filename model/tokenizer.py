"""
Token vocabulary for the Semantic ID sequence model.

Each item is represented by a 4-token Semantic ID: ``(c0, c1, c2, c3)``.
``c0/c1/c2`` come from the residual VQ codebooks, ``c3`` is appended only to
break collisions when two items quantize to the same triple.

Each level uses its own contiguous token range, so the decoder can be
constrained to legal tokens per step (see ``model/inference.py``).
"""

from itertools import accumulate

K_LEVELS = [256, 256, 256, 64]

# LEVEL_OFFSETS[level] = first flat token id reserved for that level.
LEVEL_OFFSETS = [0, *accumulate(K_LEVELS)][:-1]

VOCAB_SIZE = sum(K_LEVELS) + 3
BOS_TOKEN  = sum(K_LEVELS)
EOS_TOKEN  = sum(K_LEVELS) + 1
PAD_TOKEN  = sum(K_LEVELS) + 2


def item_to_tokens(semantic_id):
    """Map an item's semantic_id to its flat token ids."""
    return [int(code) + LEVEL_OFFSETS[level]
            for level, code in enumerate(semantic_id)]


def tokens_to_semantic_id(tokens):
    """Inverse of ``item_to_tokens``; used during decoding."""
    assert len(tokens) == len(K_LEVELS)
    return tuple(token - LEVEL_OFFSETS[level] for level, token in enumerate(tokens))


def seq_to_t5_tokens(item_seq, semantic_ids, maxlen=20):
    """
    Flatten a user history into the encoder input: a fixed-length sequence
    of Semantic ID tokens, left-padded with PAD.

    No BOS/EOS/separators between items — the decoder learns the level layout
    from position alone.
    """
    tokens = []
    for item_id in item_seq[-maxlen:]:
        if item_id in semantic_ids:
            tokens.extend(item_to_tokens(semantic_ids[item_id]))
    target_len = maxlen * len(K_LEVELS)
    if len(tokens) < target_len:
        tokens = [PAD_TOKEN] * (target_len - len(tokens)) + tokens
    return tokens
