"""
Item embedding 提取：用 Ollama 本地推理调用 ``nomic-embed-text``，把每个 item
的 ``title / category / description`` 拼成 prompt，得到 768 维 embedding。

输入：``data/beauty_data.pkl`` 里的 item 元数据
输出：``embedding/item_embeddings_raw.npy``  (dict[item_id -> np.ndarray])

支持断点续传：已存在的 embedding 不会重跑，每 500 条落盘一次。

用法：
    ollama serve &           # 启动后台服务（一次即可）
    ollama pull nomic-embed-text
    python embedding/extract_embeddings.py
"""

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


def build_item_prompt(meta):
    """把 item 元数据拼成给 nomic 的 prompt（含 instruction 后缀）。"""
    title = meta.get('title', '')
    description = meta.get('description', '')
    if isinstance(description, list):
        description = ' '.join(description)
    categories = meta.get('categories', [[]])
    categories = ' > '.join(categories[0]) if categories else ''
    return (
        f"Product: {title}\n"
        f"Category: {categories}\n"
        f"Description: {description[:300]}\n\n"
        "Represent this product for semantic retrieval."
    )


def extract_all():
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

    # 断点续传：如果已有部分结果则继续
    if os.path.exists(output_path):
        embeddings = np.load(output_path, allow_pickle=True).item()
        print(f"加载已有 embedding: {len(embeddings)} 个")
    else:
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
    extract_all()
