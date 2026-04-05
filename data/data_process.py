import json
import ast
import gzip
import pickle
from collections import defaultdict

def load_and_process(review_path, meta_path, min_interactions=5):
    # 1. 读取评论
    reviews = []
    with gzip.open(review_path, 'rb') as f:
        for line in f:
            d = json.loads(line)
            reviews.append((d['reviewerID'], d['asin'], d['unixReviewTime']))

    print(f"原始交互数: {len(reviews)}")

    # 2. 5-core 过滤
    user_cnt = defaultdict(int)
    item_cnt = defaultdict(int)
    for u, i, t in reviews:
        user_cnt[u] += 1
        item_cnt[i] += 1
    filtered = [(u, i, t) for u, i, t in reviews
                if user_cnt[u] >= min_interactions and item_cnt[i] >= min_interactions]

    print(f"5-core 过滤后交互数: {len(filtered)}")

    # 3. 构建映射（item ID 从 1 开始，0 留给 padding）
    items = sorted(set(i for _, i, _ in filtered))
    item2id = {item: idx + 1 for idx, item in enumerate(items)}

    # 4. 按用户排序，leave-one-out 划分
    user_seqs = defaultdict(list)
    for u, i, t in sorted(filtered, key=lambda x: x[2]):
        user_seqs[u].append(item2id[i])

    train, val, test = {}, {}, {}
    for u, seq in user_seqs.items():
        if len(seq) < 3:
            continue
        train[u] = seq[:-2]
        val[u]   = seq[:-1]   # 训练序列 + 倒数第二个作为 val target
        test[u]  = seq        # 训练序列 + 最后一个作为 test target

    # 5. 读取 item 元数据（用于 LLM embedding 提取）
    # meta 文件是 Python dict 格式（单引号），需用 ast.literal_eval 解析
    item_metas = {}
    with gzip.open(meta_path, 'rb') as f:
        for line in f:
            try:
                d = ast.literal_eval(line.decode('utf-8'))
                if d['asin'] in item2id:
                    item_metas[d['asin']] = d
            except:
                continue

    print(f"用户数: {len(train)}, item 数: {len(item2id)}")
    print(f"有元数据的 item 数: {len(item_metas)}")
    return train, val, test, item2id, item_metas


if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    review_path = os.path.join(script_dir, 'reviews_Beauty_5.json.gz')
    meta_path   = os.path.join(script_dir, 'meta_Beauty.json.gz')
    output_path = os.path.join(script_dir, 'beauty_data.pkl')

    train, val, test, item2id, item_metas = load_and_process(review_path, meta_path)
    pickle.dump({
        'train': train, 'val': val, 'test': test,
        'item2id': item2id, 'item_metas': item_metas
    }, open(output_path, 'wb'))
    print(f"已保存至 {output_path}")
    # 预期：用户数 ~22,363，item 数 ~12,101
