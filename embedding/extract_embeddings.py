import ollama
import pickle
import numpy as np
import os
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def build_item_prompt(meta):
    title = meta.get('title', '')
    description = meta.get('description', '')
    if isinstance(description, list):
        description = ' '.join(description)
    categories = meta.get('categories', [[]])
    if categories:
        categories = ' > '.join(categories[0])
    else:
        categories = ''
    return f"""Product: {title}
Category: {categories}
Description: {description[:300]}

Represent this product for semantic retrieval."""


if __name__ == '__main__':
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path   = os.path.join(base_dir, 'data', 'beauty_data.pkl')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'item_embeddings_raw.npy')

    # 当前使用 qwen2:7b（3584 维，语义质量更高）
    # 可替换为更大的 LLM（如 qwen2.5:7b）或轻量模型（nomic-embed-text，768 维）
    # 换模型只需改这一行，其余代码不用动
    MODEL      = 'qwen2:7b'
    N_WORKERS  = 2   # 并发数：2 个 worker 速度最优（~36 分钟），4 个无额外提升

    data = pickle.load(open(data_path, 'rb'))

    # 先检查 Ollama 是否可用
    try:
        test_resp = ollama.embeddings(model=MODEL, prompt='test')
        print(f"Ollama 连接成功，模型: {MODEL}，embedding 维度: {len(test_resp['embedding'])}")
    except Exception as e:
        print(f"Ollama 连接失败: {e}")
        print(f"请确保已运行: ollama serve 并已拉取模型: ollama pull {MODEL}")
        exit(1)

    # 断点续传：如果已有部分结果则继续
    if os.path.exists(output_path):
        embeddings = np.load(output_path, allow_pickle=True).item()
        print(f"加载已有 embedding: {len(embeddings)} 个")
    else:
        embeddings = {}

    items_to_process = [(asin, meta) for asin, meta in data['item_metas'].items()
                        if data['item2id'].get(asin) not in embeddings]
    print(f"待处理 item 数: {len(items_to_process)}，并发数: {N_WORKERS}")

    # 线程安全的写入锁
    lock    = threading.Lock()
    failed  = 0
    save_counter = [0]  # 用 list 以便在闭包中修改

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
                save_counter[0] += 1
                # 每 500 个保存一次（断点续传）
                if save_counter[0] % 500 == 0:
                    np.save(output_path, embeddings)

        pbar.close()

    np.save(output_path, embeddings)
    print(f"\nDone: {len(embeddings)} items，失败: {failed} 个")
    print(f"已保存至 {output_path}")
    # 预期：qwen2:7b，2 workers，~36 分钟，embedding 维度 3584
