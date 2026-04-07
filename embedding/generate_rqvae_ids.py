"""
RQ-VAE ID Generation: loads the best checkpoint and produces a semantic IDs npy.

Loads:  checkpoints/rqvae_{TAG}_best_collision.pt   (TAG taken from rqvae.OUTPUT_TAG)
Saves:  embedding/semantic_ids_rqvae_{TAG}.npy

Each item gets a 4-tuple (c0, c1, c2, c3):
  - c0/c1/c2: learned RQ-VAE codes (ranges 256 / 256 / 256)
  - c3: collision-resolution index (0 when no collision, 0..N-1 within a group)

Usage:
    python embedding/generate_rqvae_ids.py
"""

import os
import sys
from collections import defaultdict

import numpy as np
import torch

# Allow importing sibling module rqvae.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rqvae import (
    RQVAE,
    K_LEVELS,
    BATCH_SIZE,
    EMBEDDING_FILE,
    OUTPUT_TAG,
    select_device,
)

# c3 collision-resolution capacity; must match model/tokenizer.py K_LEVELS[3]
COLLISION_K = 64


def resolve_collisions(semantic_ids_raw):
    """
    Append a 4th collision-resolution code to each item's (c0, c1, c2) triple.

    Items with unique (c0,c1,c2) get c3=0.
    Items in a collision group get c3=0,1,2,... within the group.

    Args:
        semantic_ids_raw: dict  item_id -> (c0, c1, c2)

    Returns:
        dict  item_id -> (c0, c1, c2, c3)
    """
    sid_to_items = defaultdict(list)
    for item_id, sid in semantic_ids_raw.items():
        sid_to_items[sid].append(item_id)

    n_collisions = sum(1 for items in sid_to_items.values() if len(items) > 1)
    n_collision_items = sum(
        len(items) for items in sid_to_items.values() if len(items) > 1
    )
    print(f'冲突 Semantic ID 组数: {n_collisions}  涉及 item 数: {n_collision_items}')

    max_group = max(len(items) for items in sid_to_items.values())
    print(f'最大 collision group 大小: {max_group}  (c3 容量: {COLLISION_K})')
    if max_group > COLLISION_K:
        raise RuntimeError(
            f'最大 collision group={max_group} 超过 c3 容量 {COLLISION_K}，'
            '请增大 COLLISION_K 或调整码本大小'
        )

    resolved = {}
    for sid, items in sid_to_items.items():
        for c3, item_id in enumerate(items):
            resolved[item_id] = (*sid, c3)

    return resolved


def generate_ids():
    device = select_device()
    print(f'使用设备: {device}')

    emb_dir  = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(emb_dir)

    # Load embeddings
    raw = np.load(
        os.path.join(emb_dir, EMBEDDING_FILE),
        allow_pickle=True,
    ).item()
    print(f'Embedding 源: {EMBEDDING_FILE}')

    item_ids   = sorted(raw.keys())
    emb_matrix = np.stack([raw[i] for i in item_ids]).astype(np.float32)
    # 不做 L2 normalize：与 rqvae.py 训练时保持一致
    print(f'加载 embedding: {emb_matrix.shape}')

    in_dim = emb_matrix.shape[1]
    data_tensor = torch.tensor(emb_matrix, dtype=torch.float32, device=device)
    n_items = len(data_tensor)

    # Load best checkpoint (collision-rate based, matching TIGER's generate_code.py)
    ckpt_path = os.path.join(
        proj_dir, 'checkpoints', f'rqvae_{OUTPUT_TAG}_best_collision.pt'
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = RQVAE(in_dim=ckpt.get('in_dim', in_dim)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(
        f'加载 checkpoint: {ckpt_path}\n'
        f'  epoch={ckpt.get("epoch", "?")}  '
        f'unique_rate={ckpt.get("unique_rate", float("nan")):.1%}  '
        f'best_collision_rate={ckpt.get("best_collision_rate", float("nan")):.4f}'
    )

    # Extract 3-level codes
    print('提取 Semantic IDs...')
    chunks = []
    with torch.no_grad():
        for i in range(0, n_items, BATCH_SIZE):
            batch = data_tensor[i:i + BATCH_SIZE]
            indices = model.get_indices(batch, use_sk=False)  # (B, n_levels)
            chunks.append(indices.cpu())
    all_codes = torch.cat(chunks, dim=0).numpy()  # (N, n_levels)

    semantic_ids_raw = {
        item_id: tuple(int(c) for c in all_codes[idx])
        for idx, item_id in enumerate(item_ids)
    }

    # Resolve collisions → add c3
    semantic_ids = resolve_collisions(semantic_ids_raw)

    # Verify uniqueness
    all_sids = [tuple(v) for v in semantic_ids.values()]
    unique_count = len(set(all_sids))
    print(f'解决后唯一 Semantic ID 数: {unique_count} / {len(all_sids)}')
    assert unique_count == len(all_sids), '仍有冲突，请检查 COLLISION_K'

    # Save
    output_path = os.path.join(emb_dir, f'semantic_ids_rqvae_{OUTPUT_TAG}.npy')
    np.save(output_path, semantic_ids)
    print(f'已保存至 {output_path}')

    # Sanity checks
    print('\n--- Sanity Check ---')
    print(f'总 item 数: {len(semantic_ids)}')
    for lvl in range(len(K_LEVELS)):
        used = len({v[lvl] for v in semantic_ids.values()})
        print(f'c{lvl} 使用码数: {used} / {K_LEVELS[lvl]}')
    c3_max = max(v[3] for v in semantic_ids.values())
    print(f'c3 最大值: {c3_max}  (容量: {COLLISION_K})')


if __name__ == '__main__':
    generate_ids()
