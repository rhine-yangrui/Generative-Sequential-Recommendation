"""
RQ-VAE ID Generation: loads the best checkpoint and produces a semantic IDs npy.

Loads:  checkpoints/rqvae_{TAG}_best.pt   (TAG taken from rqvae.OUTPUT_TAG)
Saves:  embedding/semantic_ids_rqvae_{TAG}.npy

Each item gets a 4-tuple (c0, c1, c2, c3):
  - c0/c1/c2: learned RQ-VAE codes (ranges 256 / 256 / 256)
  - c3: collision-resolution index (0 when no collision, 0..N-1 within a group)

Usage:
    python embedding/generate_rqvae_ids.py
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

# Allow importing sibling module rqvae.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rqvae import RQVAE, K_LEVELS, BATCH_SIZE, select_device, OUTPUT_TAG

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
        os.path.join(emb_dir, 'item_embeddings_raw_nomic.npy'),
        allow_pickle=True,
    ).item()

    item_ids   = sorted(raw.keys())
    emb_matrix = np.stack([raw[i] for i in item_ids]).astype(np.float32)
    emb_matrix /= np.clip(np.linalg.norm(emb_matrix, axis=1, keepdims=True), 1e-12, None)
    print(f'加载 embedding: {emb_matrix.shape}')

    data_tensor = torch.tensor(emb_matrix, dtype=torch.float32, device=device)
    n_items = len(data_tensor)

    # Load best checkpoint
    ckpt_path = os.path.join(proj_dir, 'checkpoints', f'rqvae_{OUTPUT_TAG}_best.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = RQVAE().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(
        f'加载 checkpoint: {ckpt_path}\n'
        f'  epoch={ckpt["epoch"]}  '
        f'recon_loss={ckpt.get("recon_loss", float("nan")):.4f}  '
        f'unique_rate={ckpt["unique_rate"]:.1%}'
    )

    # Extract 3-level codes
    print('提取 Semantic IDs...')
    all_codes_per_level = [[] for _ in range(len(K_LEVELS))]
    with torch.no_grad():
        for i in range(0, n_items, BATCH_SIZE):
            batch = data_tensor[i:i + BATCH_SIZE]
            z = model.encoder(batch)
            codes, _, _ = model.quantizer(z)
            for level, code_tensor in enumerate(codes):
                all_codes_per_level[level].extend(code_tensor.cpu().tolist())

    semantic_ids_raw = {
        item_id: tuple(all_codes_per_level[lvl][idx] for lvl in range(len(K_LEVELS)))
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
    c0_used = len(set(v[0] for v in semantic_ids.values()))
    print(f'c0 使用码数: {c0_used} / {K_LEVELS[0]}')
    c1_used = len(set(v[1] for v in semantic_ids.values()))
    print(f'c1 使用码数: {c1_used} / {K_LEVELS[1]}')
    c2_used = len(set(v[2] for v in semantic_ids.values()))
    print(f'c2 使用码数: {c2_used} / {K_LEVELS[2]}')
    c3_max = max(v[3] for v in semantic_ids.values())
    print(f'c3 最大值: {c3_max}  (容量: {COLLISION_K})')


if __name__ == '__main__':
    generate_ids()
