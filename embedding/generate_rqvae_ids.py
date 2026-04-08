"""
Run inference with the trained RQ-VAE and write
``embedding/semantic_ids_rqvae.npy``.

Each item gets a 4-tuple ``(c0, c1, c2, c3)``:
  - c0/c1/c2: argmin codes from the 3 residual codebooks (range 0..255)
  - c3: collision-resolution index (0 if unique, else 0..N-1 within the group)

    python embedding/generate_rqvae_ids.py
"""

import os
import sys
from collections import defaultdict

import numpy as np
import torch

# Make both this directory (for ``rqvae``) and the project root (for
# ``model.tokenizer``) importable when invoked from the project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))

from rqvae import (
    BATCH_SIZE,
    CODEBOOK_SIZES,
    EMBEDDING_FILE,
    RQVAE,
    select_device,
)
from model.tokenizer import K_LEVELS as TOKEN_LAYOUT  # 4-token layout incl. c4

# Single source of truth for the c4 collision-resolution capacity.
COLLISION_K = TOKEN_LAYOUT[3]
assert TOKEN_LAYOUT[:3] == CODEBOOK_SIZES, (
    'tokenizer K_LEVELS[:3] must match rqvae CODEBOOK_SIZES; '
    f'got {TOKEN_LAYOUT[:3]} vs {CODEBOOK_SIZES}'
)


def resolve_collisions(semantic_ids_raw):
    """
    Append a 4th code so that every item has a unique 4-tuple.

    Items with a unique (c0, c1, c2) get c3=0; items in a collision group get
    c3 = 0, 1, 2, ... within the group.
    """
    sid_to_items = defaultdict(list)
    for item_id, sid in semantic_ids_raw.items():
        sid_to_items[sid].append(item_id)

    n_collisions = sum(1 for items in sid_to_items.values() if len(items) > 1)
    n_collision_items = sum(
        len(items) for items in sid_to_items.values() if len(items) > 1
    )
    print(f'collision groups: {n_collisions}  items in collisions: {n_collision_items}')

    max_group = max(len(items) for items in sid_to_items.values())
    print(f'max group size: {max_group}  (c3 capacity: {COLLISION_K})')
    if max_group > COLLISION_K:
        raise RuntimeError(
            f'max collision group {max_group} exceeds c3 capacity {COLLISION_K}; '
            'increase COLLISION_K or change codebook sizes'
        )

    resolved = {}
    for sid, items in sid_to_items.items():
        for c3, item_id in enumerate(items):
            resolved[item_id] = (*sid, c3)

    return resolved


def generate_ids():
    device = select_device()
    print(f'Device: {device}')

    emb_dir  = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(emb_dir)

    raw = np.load(
        os.path.join(emb_dir, EMBEDDING_FILE),
        allow_pickle=True,
    ).item()
    print(f'Embeddings: {EMBEDDING_FILE}')

    item_ids   = sorted(raw.keys())
    emb_matrix = np.stack([raw[i] for i in item_ids]).astype(np.float32)
    # Same preprocessing as training: no L2 normalisation.
    print(f'Loaded embedding matrix: {emb_matrix.shape}')

    in_dim = emb_matrix.shape[1]
    data_tensor = torch.from_numpy(emb_matrix).to(device)
    n_items = len(data_tensor)

    ckpt_path = os.path.join(proj_dir, 'checkpoints', 'rqvae_best.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = RQVAE(in_dim=ckpt.get('in_dim', in_dim)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'Loaded ckpt: {ckpt_path}  epoch={ckpt.get("epoch", "?")}  '
          f'unique_rate={ckpt.get("unique_rate", "?")}')

    print('Extracting Semantic IDs...')
    chunks = []
    with torch.no_grad():
        for i in range(0, n_items, BATCH_SIZE):
            batch = data_tensor[i:i + BATCH_SIZE]
            indices = model.get_indices(batch, use_sk=False)
            chunks.append(indices.cpu())
    all_codes = torch.cat(chunks, dim=0).numpy()  # (N, n_levels)

    semantic_ids_raw = {
        item_id: tuple(int(c) for c in all_codes[idx])
        for idx, item_id in enumerate(item_ids)
    }

    semantic_ids = resolve_collisions(semantic_ids_raw)

    all_sids = [tuple(v) for v in semantic_ids.values()]
    unique_count = len(set(all_sids))
    print(f'unique semantic ids after collision resolution: '
          f'{unique_count} / {len(all_sids)}')
    assert unique_count == len(all_sids), 'collisions remain — check COLLISION_K'

    output_path = os.path.join(emb_dir, 'semantic_ids_rqvae.npy')
    np.save(output_path, semantic_ids)
    print(f'Saved to {output_path}')

    print('\n--- sanity check ---')
    print(f'#items: {len(semantic_ids)}')
    for lvl in range(len(CODEBOOK_SIZES)):
        used = len({v[lvl] for v in semantic_ids.values()})
        print(f'c{lvl} used: {used} / {CODEBOOK_SIZES[lvl]}')
    c3_max = max(v[3] for v in semantic_ids.values())
    print(f'c3 max value: {c3_max}  (capacity: {COLLISION_K})')


if __name__ == '__main__':
    generate_ids()
