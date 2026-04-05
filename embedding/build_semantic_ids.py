"""
层次化 K-means 构建 Semantic ID。

码本结构（与 TIGER RQ-VAE 对齐）：
  Level 1: K=4   粗类（如大品类）
  Level 2: K=16  子类
  Level 3: K=256 细分

总容量 4×16×256 = 16,384 > 12,101 items，Beauty 数据集完全覆盖。

用法：
    python embedding/build_semantic_ids.py
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import os

K_LEVELS = [4, 32, 256]   # 与 model/tokenizer.py 保持一致


if __name__ == '__main__':
    emb_dir     = os.path.dirname(os.path.abspath(__file__))
    raw_path    = os.path.join(emb_dir, 'item_embeddings_raw.npy')
    output_path = os.path.join(emb_dir, 'semantic_ids.npy')

    raw = np.load(raw_path, allow_pickle=True).item()
    item_ids   = sorted(raw.keys())
    emb_matrix = np.stack([raw[i] for i in item_ids]).astype(np.float32)
    print(f"加载 embedding: {emb_matrix.shape}  ({len(item_ids)} items)")

    K1, K2, K3 = K_LEVELS

    # Layer 1：全量 k-means，分成 K1=4 个粗类
    print(f"\nLayer 1 k-means (K={K1})...")
    km1 = MiniBatchKMeans(n_clusters=K1, random_state=42, n_init=20, batch_size=4096)
    labels_l1 = km1.fit_predict(emb_matrix)
    print(f"Layer 1 完成，各簇大小: {np.bincount(labels_l1).tolist()}")

    semantic_ids = {}   # item_id -> (c1, c2, c3)

    # Layer 2 & 3：在每个 L1 簇内做嵌套 k-means
    for c1 in range(K1):
        mask_l1      = labels_l1 == c1
        sub_items_l1 = [item_ids[i] for i in range(len(item_ids)) if mask_l1[i]]
        sub_emb_l1   = emb_matrix[mask_l1]

        if len(sub_emb_l1) == 0:
            continue

        k2 = min(K2, len(sub_emb_l1))
        if k2 < 2:
            semantic_ids[sub_items_l1[0]] = (c1, 0, 0)
            continue

        print(f"  L1={c1}: {len(sub_emb_l1)} items → L2 k-means (K={k2})...")
        km2 = MiniBatchKMeans(n_clusters=k2, random_state=42, n_init=10,
                              batch_size=min(4096, len(sub_emb_l1)))
        labels_l2 = km2.fit_predict(sub_emb_l1)

        for c2 in range(k2):
            mask_l2      = labels_l2 == c2
            sub_items_l2 = [sub_items_l1[i] for i in range(len(sub_items_l1)) if mask_l2[i]]
            sub_emb_l2   = sub_emb_l1[mask_l2]

            if len(sub_emb_l2) == 0:
                continue

            k3 = min(K3, len(sub_emb_l2))
            if k3 < 2:
                semantic_ids[sub_items_l2[0]] = (c1, c2, 0)
                continue

            km3 = MiniBatchKMeans(n_clusters=k3, random_state=42, n_init=5,
                                  batch_size=min(4096, len(sub_emb_l2)))
            labels_l3 = km3.fit_predict(sub_emb_l2)

            for i, item_id in enumerate(sub_items_l2):
                semantic_ids[item_id] = (c1, c2, labels_l3[i])

    np.save(output_path, semantic_ids)
    print(f"\nSemantic IDs built for {len(semantic_ids)} items")
    print(f"已保存至 {output_path}")

    # 分布验证
    c1_vals = [v[0] for v in semantic_ids.values()]
    c2_vals = [v[1] for v in semantic_ids.values()]
    c3_vals = [v[2] for v in semantic_ids.values()]
    print(f"\nc1 分布 (期望 4 个簇均匀): {sorted(Counter(c1_vals).items())}")
    print(f"c2 使用了 {len(set(c2_vals))}/{K2} 个子类")
    print(f"c3 使用了 {len(set(c3_vals))}/{K3} 个细类")

    # 冲突检查
    sid_counts  = Counter(tuple(v) for v in semantic_ids.values())
    collisions  = {k: v for k, v in sid_counts.items() if v > 1}
    n_collision_items = sum(v for v in collisions.values())
    print(f"\n唯一 Semantic ID 数: {len(sid_counts)}")
    if collisions:
        print(f"冲突 Semantic ID 数: {len(collisions)}，涉及 item 数: {n_collision_items}")
        print(f"（冲突 item 将在推理时共享同一 Semantic ID，通过 Hamming 最近邻区分）")
    else:
        print("无冲突，每个 item 有唯一 Semantic ID")
