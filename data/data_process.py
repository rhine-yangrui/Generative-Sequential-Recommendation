"""
Process the Amazon Beauty 5-core dataset into ``data/beauty_data.pkl``.

The output pickle contains:
    train / val / test : dict[user_id -> List[item_id]], leave-one-out split
    item2id            : dict[asin -> int], item ids start at 1 (0 is padding)
    item_metas         : dict[asin -> meta dict], used downstream by the
                         embedding extractor
"""

import ast
import gzip
import json
import pickle
from collections import defaultdict


def load_and_process(review_path, meta_path, min_interactions=5):
    # 1. Read raw reviews.
    reviews = []
    with gzip.open(review_path, 'rb') as f:
        for line in f:
            d = json.loads(line)
            reviews.append((d['reviewerID'], d['asin'], d['unixReviewTime']))

    print(f'#raw interactions: {len(reviews)}')

    # 2. 5-core filtering on both users and items.
    user_cnt = defaultdict(int)
    item_cnt = defaultdict(int)
    for u, i, t in reviews:
        user_cnt[u] += 1
        item_cnt[i] += 1
    filtered = [(u, i, t) for u, i, t in reviews
                if user_cnt[u] >= min_interactions and item_cnt[i] >= min_interactions]

    print(f'#interactions after 5-core filter: {len(filtered)}')

    # 3. Item ids start at 1 (0 is reserved for padding).
    items = sorted(set(i for _, i, _ in filtered))
    item2id = {item: idx + 1 for idx, item in enumerate(items)}

    # 4. Sort by timestamp, leave-one-out split per user.
    user_seqs = defaultdict(list)
    for u, i, t in sorted(filtered, key=lambda x: x[2]):
        user_seqs[u].append(item2id[i])

    train, val, test = {}, {}, {}
    for u, seq in user_seqs.items():
        if len(seq) < 3:
            continue
        train[u] = seq[:-2]
        val[u]   = seq[:-1]   # train + last-but-one as val target
        test[u]  = seq        # train + last as test target

    # 5. Item metadata. The Amazon meta files use Python dict literal syntax
    # (single quotes), so json.loads doesn't work — use ast.literal_eval.
    item_metas = {}
    with gzip.open(meta_path, 'rb') as f:
        for line in f:
            try:
                d = ast.literal_eval(line.decode('utf-8'))
            except (SyntaxError, ValueError):
                continue
            if d.get('asin') in item2id:
                item_metas[d['asin']] = d

    print(f'#users: {len(train)}  #items: {len(item2id)}')
    print(f'#items with metadata: {len(item_metas)}')
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
    print(f'Saved to {output_path}')
    # Expected: ~22,363 users, ~12,101 items.
