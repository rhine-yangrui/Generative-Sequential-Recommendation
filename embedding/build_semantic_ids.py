"""
层次化 K-means 构建 Semantic ID，保证零冲突。

码本结构：
  Level 1: K=4   粗类（全量 k-means）
  Level 2: K=32  子类（在每个 L1 簇内 k-means，自适应加倍保证子簇 ≤ 256）
  Level 3: K=256 细分（按到 L2 簇心的距离排序赋序号，保证零冲突）

L3 不做 k-means，改为距离排序：
  每个 (c1, c2) 子簇内，计算每个 item 到 L2 簇心的欧氏距离，
  按距离从小到大赋 c3 = 0, 1, 2, ...
  优点：零冲突（序号唯一），且距离相近的 item 获得相近的 c3 值。

用法：
    python embedding/build_semantic_ids.py
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import os

K_LEVELS = [4, 64, 256]   # 与 model/tokenizer.py 保持一致


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

    for c1 in range(K1):
        mask_l1      = labels_l1 == c1
        sub_items_l1 = [item_ids[i] for i in range(len(item_ids)) if mask_l1[i]]
        sub_emb_l1   = emb_matrix[mask_l1]
        n_l1         = len(sub_emb_l1)

        if n_l1 == 0:
            continue
        if n_l1 == 1:
            semantic_ids[sub_items_l1[0]] = (c1, 0, 0)
            continue

        # Layer 2：在 L1 子集内做 k-means
        # 若有子簇 > K3=256，自动将 k2 加倍重试，直到所有子簇 ≤ 256
        k2 = min(K2, n_l1)
        while True:
            print(f"  L1={c1}: {n_l1} items → L2 k-means (K={k2})...")
            km2       = MiniBatchKMeans(n_clusters=k2, random_state=42, n_init=10,
                                        batch_size=min(4096, n_l1))
            labels_l2 = km2.fit_predict(sub_emb_l1)

            cluster_sizes = np.bincount(labels_l2, minlength=k2)
            max_size      = cluster_sizes.max()
            if max_size <= K3:
                break   # 所有子簇 ≤ 256，可以用距离排序安全赋 c3

            new_k2 = k2 * 2
            if new_k2 > K2:
                raise RuntimeError(
                    f"L1={c1}（{n_l1} items）：L2 最大子簇 {max_size} > K3={K3}，"
                    f"加倍后 k2={new_k2} 超出 K_LEVELS[1]={K2}。"
                    f"请增大 K_LEVELS[1]。"
                )
            print(f"    警告: 最大子簇 {max_size} > {K3}，k2 加倍: {k2} → {new_k2}")
            k2 = new_k2

        # Layer 3：距离排序赋序号，保证零冲突
        for c2 in range(k2):
            mask_l2      = labels_l2 == c2
            sub_items_l2 = [sub_items_l1[i] for i in range(n_l1) if mask_l2[i]]
            sub_emb_l2   = sub_emb_l1[mask_l2]

            if len(sub_emb_l2) == 0:
                continue

            # 按到 L2 簇心的欧氏距离排序，距离最近的 item 获得 c3=0
            centroid  = km2.cluster_centers_[c2]            # (D,)
            distances = np.linalg.norm(sub_emb_l2 - centroid, axis=1)
            order     = np.argsort(distances)               # 从小到大

            for rank, idx in enumerate(order):
                semantic_ids[sub_items_l2[idx]] = (c1, c2, rank)

    np.save(output_path, semantic_ids)
    print(f"\nSemantic IDs built for {len(semantic_ids)} items")
    print(f"已保存至 {output_path}")

    # 分布验证
    c1_vals = [v[0] for v in semantic_ids.values()]
    c2_vals = [v[1] for v in semantic_ids.values()]
    c3_vals = [v[2] for v in semantic_ids.values()]
    print(f"\nc1 分布: {sorted(Counter(c1_vals).items())}")
    print(f"c2 使用了 {len(set(c2_vals))}/{K2} 个子类")
    print(f"c3 最大值: {max(c3_vals)}（上限 {K3 - 1}）")

    # 冲突检查
    all_sids      = [tuple(v) for v in semantic_ids.values()]
    unique_count  = len(set(all_sids))
    total_count   = len(all_sids)
    print(f"\n唯一 Semantic ID 数: {unique_count}  /  item 总数: {total_count}")
    if unique_count == total_count:
        print("✓ 零冲突，每个 item 有唯一 Semantic ID")
    else:
        collisions = {k: v for k, v in Counter(all_sids).items() if v > 1}
        print(f"✗ 仍有冲突：{len(collisions)} 个 Semantic ID 被多个 item 共用")
