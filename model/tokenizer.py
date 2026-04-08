"""
Token 词表设计
==============

每个 item 用 4 个 token 表示其 Semantic ID：

    (c0, c1, c2, c3)

- c0/c1/c2 由 RQ-VAE 学习得到（每级 256 维 codebook）
- c3 仅用于 collision resolution；无冲突时为 0，c3 容量 64

每一级使用各自独立的 token 区间，所以下游 T5 解码时只要按 level 限制
合法 token 范围即可（见 ``model/inference.py``）。
"""

from itertools import accumulate

K_LEVELS = [256, 256, 256, 64]

# Cumulative sum of K_LEVELS, used to map (level, code) → flat token id.
# LEVEL_OFFSETS[level] is the first token id reserved for that level.
LEVEL_OFFSETS = [0, *accumulate(K_LEVELS)][:-1]

VOCAB_SIZE = sum(K_LEVELS) + 3
BOS_TOKEN  = sum(K_LEVELS)
EOS_TOKEN  = sum(K_LEVELS) + 1
PAD_TOKEN  = sum(K_LEVELS) + 2


def item_to_tokens(semantic_id):
    """把一个 item 的 semantic_id 转成对应 token 编号。"""
    return [int(code) + LEVEL_OFFSETS[level]
            for level, code in enumerate(semantic_id)]


def tokens_to_semantic_id(tokens):
    """把 token 编号还原成 semantic_id，用于推理阶段解码。"""
    assert len(tokens) == len(K_LEVELS)
    return tuple(token - LEVEL_OFFSETS[level] for level, token in enumerate(tokens))


def seq_to_t5_tokens(item_seq, semantic_ids, maxlen=20):
    """
    T5 encoder 输入：拍平的 Semantic ID token 序列，左侧 PAD 到固定长度。

    没有 BOS / EOS / 分隔符，对齐 TIGER 参考实现 (../TIGER/model/dataloader.py)。

    Args:
        item_seq:     用户交互的 item ID 列表
        semantic_ids: dict, item_id -> (c0, c1, c2, c3)
        maxlen:       最多保留最近多少个 item

    Returns:
        长度恒为 ``maxlen * len(K_LEVELS)`` 的 token 列表，左 PAD
    """
    tokens = []
    for item_id in item_seq[-maxlen:]:
        if item_id in semantic_ids:
            tokens.extend(item_to_tokens(semantic_ids[item_id]))
    target_len = maxlen * len(K_LEVELS)
    if len(tokens) < target_len:
        tokens = [PAD_TOKEN] * (target_len - len(tokens)) + tokens
    return tokens
