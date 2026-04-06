"""
Token 词表设计（分层码本 4/16/256，粗到细）：
- Level 1 codes: token   0 ~   3  (4 个，直接用 c1)
- Level 2 codes: token   4 ~  19  (16 个，c2 + 4)
- Level 3 codes: token  20 ~ 275  (256 个，c3 + 20)
- [BOS]: token 276
- [EOS]: token 277
- [PAD]: token 278

码本结构：c1 ∈ {0..3}（4 大类），c2 ∈ {0..15}（子类），c3 ∈ {0..255}（细分）。
总容量 4×16×256 = 16,384 > 12,101 items。

偏移设计避免层间 token 冲突：
  c1=2  → token 2
  c2=2  → token 6   (2 + 4)
  c3=2  → token 22  (2 + 20)
"""

K_LEVELS = [4, 16, 256]                          # 每层码本大小
LEVEL_OFFSETS = [0, K_LEVELS[0],                 # [0, 4, 20]
                 K_LEVELS[0] + K_LEVELS[1]]

VOCAB_SIZE = sum(K_LEVELS) + 3                    # = 279
BOS_TOKEN  = sum(K_LEVELS)                        # = 276
EOS_TOKEN  = sum(K_LEVELS) + 1                    # = 277
PAD_TOKEN  = sum(K_LEVELS) + 2                    # = 278


def item_to_tokens(semantic_id):
    """把一个 item 的 (c1, c2, c3) 转成 3 个 token 编号。"""
    c1, c2, c3 = semantic_id
    return [int(c1) + LEVEL_OFFSETS[0],
            int(c2) + LEVEL_OFFSETS[1],
            int(c3) + LEVEL_OFFSETS[2]]


def tokens_to_semantic_id(tokens):
    """把 3 个 token 编号还原成 (c1, c2, c3)，用于推理阶段解码。"""
    assert len(tokens) == 3
    c1 = tokens[0] - LEVEL_OFFSETS[0]
    c2 = tokens[1] - LEVEL_OFFSETS[1]
    c3 = tokens[2] - LEVEL_OFFSETS[2]
    return (c1, c2, c3)


def seq_to_tokens(item_seq, semantic_ids, maxlen=50):
    """
    把用户的 item ID 序列转成 token 序列（含 BOS，每个 item 后加 EOS 分隔符）。

    Args:
        item_seq:     用户交互的 item ID 列表，如 [3, 17, 42, ...]
        semantic_ids: dict，item_id -> (c1, c2, c3)
        maxlen:       最多保留最近多少个 item

    Returns:
        token 列表，如 [276, 2, 5, 22, 277, 0, 4, 20, 277, ...]
    """
    tokens = [BOS_TOKEN]
    for item_id in item_seq[-maxlen:]:
        if item_id in semantic_ids:
            tokens.extend(item_to_tokens(semantic_ids[item_id]))
            tokens.append(EOS_TOKEN)
    return tokens
