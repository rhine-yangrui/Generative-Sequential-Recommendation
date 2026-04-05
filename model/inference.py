"""
Beam Search 推理：给定用户历史，生成 top-k 推荐 item。

流程：
  1. 历史序列 → token 序列
  2. GPT-2 beam search 生成 3 个新 token
  3. 还原成 (c1, c2, c3)
  4. 查反向索引找对应 item（精确匹配 or Hamming 距离最近邻）
"""

import torch
import numpy as np
from model.tokenizer import seq_to_tokens, tokens_to_semantic_id, PAD_TOKEN


def build_reverse_index(semantic_ids):
    """
    构建 (c1, c2, c3) → item_id 的反向索引，用于 beam search 后查找 item。

    Args:
        semantic_ids: dict，item_id -> (c1, c2, c3)

    Returns:
        sid_to_item: dict，(c1, c2, c3) -> item_id
        sid_array:   np.array，shape (N, 3)，所有 item 的 semantic_id 矩阵（用于近似匹配）
        item_id_list: list，和 sid_array 行对齐的 item_id 列表
    """
    sid_to_item  = {tuple(int(x) for x in sid): iid for iid, sid in semantic_ids.items()}
    item_id_list = list(semantic_ids.keys())
    sid_array    = np.array([semantic_ids[iid] for iid in item_id_list], dtype=np.int32)
    return sid_to_item, sid_array, item_id_list


def hamming_nearest(candidate_sid, sid_array, item_id_list, exclude_ids=None):
    """
    当 beam search 生成的 (c1,c2,c3) 没有精确匹配时，
    用 Hamming 距离找最近邻 item。

    Args:
        candidate_sid: tuple (c1, c2, c3)
        sid_array:     (N, 3) 所有 item 的 semantic_id 矩阵
        item_id_list:  和 sid_array 行对齐的 item_id
        exclude_ids:   需要排除的 item_id 集合（已推荐过的）

    Returns:
        最近邻的 item_id
    """
    cand = np.array(candidate_sid, dtype=np.int32)
    distances = (sid_array != cand).sum(axis=1)  # Hamming 距离，每行 0~3

    if exclude_ids:
        for i, iid in enumerate(item_id_list):
            if iid in exclude_ids:
                distances[i] = 999  # 排除已推荐的 item

    nearest_idx = distances.argmin()
    return item_id_list[nearest_idx]


def predict_topk(model, history_seq, semantic_ids, sid_to_item,
                 sid_array, item_id_list, k=10, beam_width=50, device='cpu'):
    """
    给定用户历史序列，用 beam search 生成 top-k 推荐。

    Args:
        model:        训练好的 GPT-2 模型
        history_seq:  用户历史 item_id 列表（不含 target）
        semantic_ids: item_id -> (c1, c2, c3)
        sid_to_item:  (c1, c2, c3) -> item_id 反向索引
        sid_array:    (N, 3) 所有 item 的 semantic_id 矩阵
        item_id_list: 和 sid_array 对齐的 item_id 列表
        k:            返回 top-k 个推荐
        beam_width:   beam search 宽度，越大候选越多但越慢（建议 ≥ k*2）

    Returns:
        recommended_items: list of item_id，长度 ≤ k
    """
    model.eval()
    input_tokens = seq_to_tokens(history_seq, semantic_ids, maxlen=50)
    input_ids    = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=3,           # 生成 c1, c2, c3 共 3 个 token
            num_beams=beam_width,
            num_return_sequences=min(beam_width, k * 3),  # 多生成一些，防止重复后不够 k 个
            early_stopping=True,
            pad_token_id=PAD_TOKEN,
        )

    # 解码每条 beam 的输出，映射到 item
    recommended = []
    seen_items  = set()

    for output in outputs:
        new_tokens = output[len(input_tokens):].tolist()
        if len(new_tokens) < 3:
            continue

        # 还原 semantic_id，做范围检查
        c1 = new_tokens[0]
        c2 = new_tokens[1] - 256
        c3 = new_tokens[2] - 512
        if not (0 <= c1 <= 255 and 0 <= c2 <= 255 and 0 <= c3 <= 255):
            continue

        candidate_sid = (c1, c2, c3)

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


if __name__ == '__main__':
    import pickle
    import os
    from model.generative_rec import build_model

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')

    # 加载数据
    data         = pickle.load(open(os.path.join(base_dir, 'data/beauty_data.pkl'), 'rb'))
    semantic_ids = np.load(os.path.join(base_dir, 'embedding/semantic_ids.npy'),
                           allow_pickle=True).item()

    # 加载模型
    model = build_model().to(device)
    model.load_state_dict(torch.load(
        os.path.join(base_dir, 'checkpoints/best_model.pt'),
        map_location=device, weights_only=True))
    model.eval()
    print('模型加载成功')

    # 构建反向索引
    sid_to_item, sid_array, item_id_list = build_reverse_index(semantic_ids)
    print(f'反向索引构建完成，共 {len(sid_to_item)} 个精确映射')

    # 取一个测试用户演示
    test_seqs = data['test']
    user      = list(test_seqs.keys())[0]
    full_seq  = test_seqs[user]
    history   = full_seq[:-1]   # 历史
    target    = full_seq[-1]    # 真实目标

    print(f'\n用户 {user}：历史长度={len(history)}，目标 item={target}')
    recs = predict_topk(model, history, semantic_ids, sid_to_item,
                        sid_array, item_id_list, k=10, device=device)
    print(f'Top-10 推荐: {recs}')
    print(f'命中（HR@10）: {"✅ 是" if target in recs else "❌ 否"}')
