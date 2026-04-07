"""
Token 词表设计：
- RQ-VAE 学到的三级 Semantic ID: 256 / 256 / 256
- 若前三码发生 collision，则追加第 4 个去冲突 token（0~63）

最终用于生成式模型的 Semantic ID 长度为 4：
  (c1, c2, c3, c4)

其中：
- c1, c2, c3 由 RQ-VAE 学习得到
- c4 仅用于 collision resolution；无冲突时统一为 0
- c4 容量设为 64，与 embedding/generate_rqvae_ids.py 的 COLLISION_K 对齐
"""

K_LEVELS = [256, 256, 256, 64]

LEVEL_OFFSETS = []
_offset = 0
for _k in K_LEVELS:
    LEVEL_OFFSETS.append(_offset)
    _offset += _k

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


def seq_to_tokens(item_seq, semantic_ids, maxlen=50):
    """
    把用户的 item ID 序列转成 token 序列（含 BOS，每个 item 后加 EOS 分隔符）。

    Args:
        item_seq:     用户交互的 item ID 列表，如 [3, 17, 42, ...]
        semantic_ids: dict，item_id -> (c1, c2, c3)
        maxlen:       最多保留最近多少个 item

    Returns:
        token 列表，如 [BOS, c1, c2, c3, c4, EOS, ...]
    """
    tokens = [BOS_TOKEN]
    for item_id in item_seq[-maxlen:]:
        if item_id in semantic_ids:
            tokens.extend(item_to_tokens(semantic_ids[item_id]))
            tokens.append(EOS_TOKEN)
    return tokens
