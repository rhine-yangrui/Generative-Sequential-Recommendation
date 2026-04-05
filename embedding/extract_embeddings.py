import ollama
import pickle
import numpy as np
import os
from tqdm import tqdm

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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'beauty_data.pkl')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'item_embeddings_raw.npy')

    data = pickle.load(open(data_path, 'rb'))

    # 先检查 Ollama 是否可用
    try:
        test_resp = ollama.embeddings(model='qwen2:7b', prompt='test')
        print(f"Ollama 连接成功，embedding 维度: {len(test_resp['embedding'])}")
    except Exception as e:
        print(f"Ollama 连接失败: {e}")
        print("请确保已运行: ollama serve 并已拉取模型: ollama pull qwen2:7b")
        exit(1)

    # 断点续传：如果已有部分结果则继续
    if os.path.exists(output_path):
        embeddings = np.load(output_path, allow_pickle=True).item()
        print(f"加载已有 embedding: {len(embeddings)} 个")
    else:
        embeddings = {}

    items_to_process = [(asin, meta) for asin, meta in data['item_metas'].items()
                        if data['item2id'].get(asin) not in embeddings]
    print(f"待处理 item 数: {len(items_to_process)}")

    failed = 0
    for asin, meta in tqdm(items_to_process):
        item_id = data['item2id'].get(asin)
        if item_id is None:
            continue
        try:
            resp = ollama.embeddings(model='qwen2:7b', prompt=build_item_prompt(meta))
            embeddings[item_id] = np.array(resp['embedding'])
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"\nSkip {asin}: {e}")

        # 每 500 个保存一次（断点续传）
        if len(embeddings) % 500 == 0:
            np.save(output_path, embeddings)

    np.save(output_path, embeddings)
    print(f"\nDone: {len(embeddings)} items，失败: {failed} 个")
    print(f"已保存至 {output_path}")
    # 预期：~10 分钟，12,101 个 item，embedding 维度 3584
