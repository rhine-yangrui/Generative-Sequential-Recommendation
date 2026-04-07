"""
Sentence-T5 embedding extraction (对齐 TIGER 参考实现)。

用 `sentence-transformers/sentence-t5-base` 为每个 item 提取 768d embedding，
与 TIGER 论文 / 参考实现使用的 embedding 来源一致。

输入:   data/beauty_data.pkl
输出:   embedding/item_embeddings_raw_st5.npy

用法:
    pip install sentence-transformers
    python embedding/extract_embeddings_t5.py
"""

import os
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME   = 'sentence-transformers/sentence-t5-base'
BATCH_SIZE   = 64
OUTPUT_NAME  = 'item_embeddings_raw_st5.npy'


def build_item_prompt(meta):
    """
    对齐 TIGER 参考实现 (../TIGER/data/process.ipynb)：
    使用 title / price / salesRank / brand / categories 五个字段，
    dict-like 格式，无 instruction 后缀（Sentence-T5 非 instruction-tuned）。
    """
    description = meta.get('description', '')
    if isinstance(description, list):
        description = ' '.join(description)
    description = description[:300]
    return (
        f"'title':{meta.get('title', '')}\n"
        f" 'price':{meta.get('price', '')}\n"
        f" 'salesRank':{meta.get('salesRank', '')}\n"
        f" 'brand':{meta.get('brand', '')}\n"
        f" 'categories':{meta.get('categories', '')}\n"
        f" 'description':{description}"
    )


def main():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emb_dir     = os.path.dirname(os.path.abspath(__file__))
    data_path   = os.path.join(base_dir, 'data', 'beauty_data.pkl')
    output_path = os.path.join(emb_dir, OUTPUT_NAME)

    device = 'cuda' if torch.cuda.is_available() else (
             'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'加载模型: {MODEL_NAME}')
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f'Embedding 维度: {model.get_sentence_embedding_dimension()}')

    data = pickle.load(open(data_path, 'rb'))
    item2id    = data['item2id']
    item_metas = data['item_metas']

    # Collect (item_id, prompt) pairs, skip items without metadata
    pairs = []
    for asin, meta in item_metas.items():
        iid = item2id.get(asin)
        if iid is None:
            continue
        pairs.append((iid, build_item_prompt(meta)))
    pairs.sort(key=lambda x: x[0])
    print(f'待处理 item 数: {len(pairs)}')

    embeddings = {}
    for start in tqdm(range(0, len(pairs), BATCH_SIZE), desc='Encoding'):
        chunk    = pairs[start:start + BATCH_SIZE]
        iids     = [p[0] for p in chunk]
        prompts  = [p[1] for p in chunk]
        vecs     = model.encode(
            prompts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # rqvae.py 会自己做 L2 norm
        )
        for iid, vec in zip(iids, vecs):
            embeddings[iid] = vec.astype(np.float32)

    np.save(output_path, embeddings)
    print(f'\n✅ 保存至 {output_path}  ({len(embeddings)} items)')
    sample_vec = next(iter(embeddings.values()))
    print(f'   示例 shape: {sample_vec.shape}  dtype: {sample_vec.dtype}')


if __name__ == '__main__':
    main()
