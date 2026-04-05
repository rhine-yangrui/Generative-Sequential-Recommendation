import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os

NUM_LEVELS = 3
K = 256  # 每层簇数（vocab size per level）


if __name__ == '__main__':
    emb_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(emb_dir, 'item_embeddings_raw.npy')
    output_path = os.path.join(emb_dir, 'semantic_ids.npy')

    raw = np.load(raw_path, allow_pickle=True).item()
    item_ids = sorted(raw.keys())
    emb_matrix = np.stack([raw[i] for i in item_ids]).astype(np.float32)  # (N, D)
    print(f"加载 embedding: {emb_matrix.shape}")

    # Layer 1: 全量 k-means
    print("Layer 1 k-means...")
    km1 = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=10, batch_size=4096)
    labels_l1 = km1.fit_predict(emb_matrix)
    print(f"Layer 1 完成，各簇大小: min={np.bincount(labels_l1).min()}, max={np.bincount(labels_l1).max()}")

    semantic_ids = {}  # item_id -> (c1, c2, c3)

    # Layer 2 & 3: 在每个 L1 簇内做嵌套 k-means
    for c1 in range(K):
        mask_l1 = labels_l1 == c1
        sub_items_l1 = [item_ids[i] for i in range(len(item_ids)) if mask_l1[i]]
        sub_emb_l1 = emb_matrix[mask_l1]

        if len(sub_emb_l1) == 0:
            continue

        k2 = min(K, len(sub_emb_l1))
        if k2 < 2:
            # 只有一个 item，直接赋值
            semantic_ids[sub_items_l1[0]] = (c1, 0, 0)
            continue

        km2 = MiniBatchKMeans(n_clusters=k2, random_state=42, n_init=5, batch_size=min(4096, len(sub_emb_l1)))
        labels_l2 = km2.fit_predict(sub_emb_l1)

        for c2 in range(k2):
            mask_l2 = labels_l2 == c2
            sub_items_l2 = [sub_items_l1[i] for i in range(len(sub_items_l1)) if mask_l2[i]]
            sub_emb_l2 = sub_emb_l1[mask_l2]

            if len(sub_emb_l2) == 0:
                continue

            k3 = min(K, len(sub_emb_l2))
            if k3 < 2:
                semantic_ids[sub_items_l2[0]] = (c1, c2, 0)
                continue

            km3 = MiniBatchKMeans(n_clusters=k3, random_state=42, n_init=5, batch_size=min(4096, len(sub_emb_l2)))
            labels_l3 = km3.fit_predict(sub_emb_l2)

            for i, item_id in enumerate(sub_items_l2):
                semantic_ids[item_id] = (c1, c2, labels_l3[i])

        if c1 % 32 == 0:
            print(f"  L1 cluster {c1}/256 done, total assigned: {len(semantic_ids)}")

    np.save(output_path, semantic_ids)
    print(f"\nSemantic IDs built for {len(semantic_ids)} items")
    print(f"已保存至 {output_path}")

    # 简单验证：检查 semantic_ids 分布
    c1_vals = [v[0] for v in semantic_ids.values()]
    unique_c1 = len(set(c1_vals))
    print(f"一级编码 (c1) 使用了 {unique_c1}/256 个簇")
    print("验证：请检查同品类商品的 c1 是否相同（抽样查看 beauty_data.pkl 中 item_metas）")
