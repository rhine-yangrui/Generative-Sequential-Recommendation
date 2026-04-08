"""
Item embedding 提取：用 Ollama 本地推理调用 ``nomic-embed-text``，把每个 item
的 6 字段元数据（title / brand / category / description / price / popularity）
拼成 prompt，得到 768 维 embedding。

输入：``data/beauty_data.pkl`` 里的 item 元数据
输出：``embedding/item_embeddings_raw.npy``  (dict[item_id -> np.ndarray])

Prompt 6-field 设计（E13）：
  - Brand / Price range / Popularity 缺失时跳过该行（不写 "unknown"）
  - Description 截断从 300 → 1000 字符，覆盖 ~p90 的描述长度
  - Price / salesRank 做离散 bucket，避免文本 embedding 对数字不敏感的问题

支持断点续传：已存在的 embedding 不会重跑，每 500 条落盘一次。
**换 prompt 模板后必须 --force，否则续传会跳过所有已处理 item。**

用法：
    ollama serve &                           # 启动后台服务（一次即可）
    ollama pull nomic-embed-text
    python embedding/extract_embeddings.py --force
"""

import argparse
import html
import os
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import ollama
from tqdm import tqdm


MODEL     = 'nomic-embed-text'   # 768d，instruction-tuned，速度快
N_WORKERS = 2                    # 并发数：2 个 worker 速度最优，>2 没有额外提升
SAVE_EVERY = 500                 # 每多少条 embedding 落盘一次（断点续传粒度）
DESCRIPTION_MAX_CHARS = 1000     # 覆盖 ~p90 的描述长度（p50≈292，p90≈1030）


def _price_bucket(price):
    """USD 价格 → 4 档离散标签；数据分位见 Progress.md。"""
    if price is None:
        return None
    if price < 10:
        return 'budget'
    if price < 25:
        return 'mid'
    if price < 50:
        return 'premium'
    return 'luxury'


def _rank_bucket(sales_rank):
    """Beauty 类目 sales rank → 4 档离散标签（log 切分）。"""
    if sales_rank is None:
        return None
    if sales_rank < 1_000:
        return 'bestseller'
    if sales_rank < 10_000:
        return 'popular'
    if sales_rank < 100_000:
        return 'niche'
    return 'long-tail'


def _get_sales_rank(meta):
    """salesRank 是 dict（如 {'Beauty': 10486}），优先取 Beauty 条目。"""
    sr = meta.get('salesRank')
    if not isinstance(sr, dict) or not sr:
        return None
    if 'Beauty' in sr:
        return sr['Beauty']
    return next(iter(sr.values()), None)


def _clean(s):
    """Unescape HTML entities in a string field. 原始 meta 里混有 ``&#39;`` / ``&nbsp;`` 等。"""
    return html.unescape(s) if isinstance(s, str) else s


def build_item_prompt(meta):
    """
    把 item 元数据拼成 6-field prompt。缺失字段整行跳过（不写 "unknown"），
    避免给缺失值一个可被模型学到的"无信息"token 向量。

    所有文本字段先 html.unescape() 清洗，防止 ``&#39;`` / ``&nbsp;`` 之类的
    实体污染 embedding 输入。
    """
    title = _clean(meta.get('title', ''))
    description = meta.get('description', '')
    if isinstance(description, list):
        description = ' '.join(description)
    description = _clean(description)
    categories = meta.get('categories', [[]])
    category_str = _clean(' > '.join(categories[0])) if categories else ''

    brand = _clean(meta.get('brand'))
    price_bucket = _price_bucket(meta.get('price'))
    rank_bucket  = _rank_bucket(_get_sales_rank(meta))

    lines = [f'Product: {title}']
    if brand:
        lines.append(f'Brand: {brand}')
    lines.append(f'Category: {category_str}')
    lines.append(f'Description: {description[:DESCRIPTION_MAX_CHARS]}')
    if price_bucket:
        lines.append(f'Price range: {price_bucket}')
    if rank_bucket:
        lines.append(f'Popularity: {rank_bucket}')

    return '\n'.join(lines) + '\n\nRepresent this product for semantic retrieval.'


def extract_all(force=False):
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path   = os.path.join(base_dir, 'data', 'beauty_data.pkl')
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'item_embeddings_raw.npy'
    )

    data = pickle.load(open(data_path, 'rb'))

    # 先 ping 一次 Ollama 确认服务可用
    try:
        test_resp = ollama.embeddings(model=MODEL, prompt='test')
        print(f"Ollama 连接成功，模型: {MODEL}，"
              f"embedding 维度: {len(test_resp['embedding'])}")
    except Exception as e:
        print(f"Ollama 连接失败: {e}")
        print(f"请确保已运行: ollama serve 并已拉取模型: ollama pull {MODEL}")
        sys.exit(1)

    # 断点续传：如果已有部分结果则继续（--force 时绕过）
    if os.path.exists(output_path) and not force:
        embeddings = np.load(output_path, allow_pickle=True).item()
        print(f"加载已有 embedding: {len(embeddings)} 个")
    else:
        if force and os.path.exists(output_path):
            print('--force：忽略已存在的 embedding，全量重算')
        embeddings = {}

    items_to_process = [
        (asin, meta) for asin, meta in data['item_metas'].items()
        if data['item2id'].get(asin) not in embeddings
    ]
    print(f"待处理 item 数: {len(items_to_process)}，并发数: {N_WORKERS}")

    lock         = threading.Lock()
    failed       = 0
    save_counter = 0

    def fetch_one(args):
        asin, meta = args
        item_id = data['item2id'].get(asin)
        if item_id is None:
            return None, None, None
        try:
            resp = ollama.embeddings(model=MODEL, prompt=build_item_prompt(meta))
            return item_id, np.array(resp['embedding']), None
        except Exception as e:
            return item_id, None, str(e)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(fetch_one, item): item for item in items_to_process}
        pbar = tqdm(total=len(items_to_process))

        for future in as_completed(futures):
            item_id, emb, err = future.result()
            pbar.update(1)

            if err:
                failed += 1
                if failed <= 5:
                    tqdm.write(f"Skip item {item_id}: {err}")
                continue
            if item_id is None or emb is None:
                continue

            with lock:
                embeddings[item_id] = emb
                save_counter += 1
                if save_counter % SAVE_EVERY == 0:
                    np.save(output_path, embeddings)

        pbar.close()

    np.save(output_path, embeddings)
    print(f"\nDone: {len(embeddings)} items，失败: {failed} 个")
    print(f"已保存至 {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='忽略已存在的 embedding，全量重算（换 prompt 模板时必加）')
    args = parser.parse_args()
    extract_all(force=args.force)
