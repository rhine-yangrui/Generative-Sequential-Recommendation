"""
Token 词表设计：
- Level 1 codes: token   0 ~ 255  (256 个，直接用 c1)
- Level 2 codes: token 256 ~ 511  (256 个，c2 + 256)
- Level 3 codes: token 512 ~ 767  (256 个，c3 + 512)
- [BOS]: token 768  （序列开始）
- [EOS]: token 769  （序列结束）
- [PAD]: token 770  （padding）

每个 item 展开为 3 个连续 token：[c1, c2+256, c3+512]
用户序列 [item1, item2, item3] 展开为：
[BOS, c1¹, c2¹+256, c3¹+512, c1², c2²+256, c3²+512, c1³, c2³+256, c3³+512]

偏移设计的目的：避免三层 token 之间的编号冲突。
例如 c1=5 和 c2=5 是完全不同的语义，编码后分别是 token 5 和 token 261，模型能区分。
"""

VOCAB_SIZE = 256 * 3 + 3  # = 771
BOS_TOKEN  = 768
EOS_TOKEN  = 769
PAD_TOKEN  = 770

LEVEL_OFFSETS = [0, 256, 512]  # c1 不加偏移，c2 加 256，c3 加 512


def item_to_tokens(semantic_id):
    """把一个 item 的 (c1, c2, c3) 转成 3 个 token 编号。"""
    c1, c2, c3 = semantic_id
    return [int(c1), int(c2) + 256, int(c3) + 512]


def tokens_to_semantic_id(tokens):
    """把 3 个 token 编号还原成 (c1, c2, c3)，用于推理阶段解码。"""
    assert len(tokens) == 3
    c1 = tokens[0]
    c2 = tokens[1] - 256
    c3 = tokens[2] - 512
    return (c1, c2, c3)


def seq_to_tokens(item_seq, semantic_ids, maxlen=50):
    """
    把用户的 item ID 序列转成 token 序列（含 BOS）。

    Args:
        item_seq:     用户交互的 item ID 列表，如 [3, 17, 42, ...]
        semantic_ids: dict，item_id -> (c1, c2, c3)
        maxlen:       最多保留最近多少个 item（截断过长的历史）

    Returns:
        token 列表，如 [768, 5, 261, 519, 12, 280, 534, ...]
    """
    tokens = [BOS_TOKEN]
    for item_id in item_seq[-maxlen:]:
        if item_id in semantic_ids:
            tokens.extend(item_to_tokens(semantic_ids[item_id]))
    return tokens


if __name__ == '__main__':
    import numpy as np
    import pickle

    semantic_ids = np.load('../embedding/semantic_ids.npy', allow_pickle=True).item()
    data = pickle.load(open('../data/beauty_data.pkl', 'rb'))
    train = data['train']

    # 取一个样本用户演示
    user = list(train.keys())[0]
    seq = train[user]
    tokens = seq_to_tokens(seq, semantic_ids)

    print(f'用户序列 (item IDs): {seq[:5]}{"..." if len(seq) > 5 else ""}')
    print(f'展开后 tokens:       {tokens[:16]}{"..." if len(tokens) > 16 else ""}')
    print()
    print(f'前 4 个 item 的解析：')
    for item_id in seq[:4]:
        if item_id in semantic_ids:
            sid = semantic_ids[item_id]
            toks = item_to_tokens(sid)
            print(f'  item {item_id:5d} → semantic_id={sid} → tokens={toks}')
    print()
    print(f'token 范围验证：')
    print(f'  BOS={BOS_TOKEN}, EOS={EOS_TOKEN}, PAD={PAD_TOKEN}')
    print(f'  VOCAB_SIZE={VOCAB_SIZE}')
    print(f'  所有 token 均 < VOCAB_SIZE: {all(t < VOCAB_SIZE for t in tokens)}')

    # 验证 round-trip：token -> semantic_id -> token
    sid_orig = semantic_ids[seq[0]]
    toks = item_to_tokens(sid_orig)
    sid_back = tokens_to_semantic_id(toks)
    assert sid_orig == sid_back, 'round-trip 失败！'
    print(f'  round-trip 验证通过: {sid_orig} → {toks} → {sid_back}')
