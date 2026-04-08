"""
Random ID baseline：为 thesis 的 null hypothesis 生成随机 4-token ID。

每级按 tokenizer 的 K_LEVELS = [256, 256, 256, 64] 均匀采样，保证 item 间唯一
（拒绝采样；12k item vs 268M 联合槽位，基本不会重试）。与 semantic_ids_rqvae.npy
完全同 shape，下游 train.py / evaluate.py 通过 --semantic-ids 切换即可。

item 列表直接从 data/beauty_data.pkl 取，不依赖任何 embedding 文件 —— 随机 ID
baseline 本来就不该碰语义信号。

    python embedding/generate_random_ids.py
"""

import os
import pickle
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJ_DIR)

from model.tokenizer import K_LEVELS

SEED = 42
DATA_FILE   = os.path.join(_PROJ_DIR, 'data', 'beauty_data.pkl')
OUTPUT_FILE = 'semantic_ids_random.npy'


def generate_random_ids():
    emb_dir = _THIS_DIR

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    item_ids = sorted(data['item2id'].values())
    print(f'item 数: {len(item_ids)}  (来源: data/beauty_data.pkl item2id)')
    print(f'K_LEVELS: {K_LEVELS}  联合槽位: {int(np.prod(K_LEVELS)):,}')

    rng = np.random.default_rng(SEED)
    semantic_ids = {}
    seen = set()
    retries = 0

    for item_id in item_ids:
        while True:
            sid = tuple(int(rng.integers(0, k)) for k in K_LEVELS)
            if sid not in seen:
                seen.add(sid)
                semantic_ids[item_id] = sid
                break
            retries += 1

    print(f'拒绝采样重试次数: {retries}')
    assert len(semantic_ids) == len(item_ids)
    assert len(seen) == len(item_ids), '仍有冲突，请检查'

    output_path = os.path.join(emb_dir, OUTPUT_FILE)
    np.save(output_path, semantic_ids)
    print(f'已保存至 {output_path}')

    # Sanity：每级使用码数
    print('\n--- Sanity Check ---')
    for lvl in range(len(K_LEVELS)):
        used = len({v[lvl] for v in semantic_ids.values()})
        print(f'c{lvl} 使用码数: {used} / {K_LEVELS[lvl]}')


if __name__ == '__main__':
    generate_random_ids()
