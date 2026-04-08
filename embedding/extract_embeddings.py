"""
Extract item embeddings with Ollama's local ``nomic-embed-text`` model.

For each item we build a 6-field prompt
(title / brand / category / description / price bucket / popularity bucket)
and call the embedding API. The result is written as
``embedding/item_embeddings_raw.npy``  (dict[item_id -> np.ndarray]).

The script supports resuming: existing rows are kept and only missing items
are processed. **Pass --force whenever the prompt template changes**, otherwise
old embeddings will be silently reused.

    ollama serve &
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


MODEL                 = 'nomic-embed-text'   # 768d, instruction-tuned
N_WORKERS             = 2                    # 2 workers gives the best throughput here
SAVE_EVERY            = 500                  # checkpoint frequency for resume support
DESCRIPTION_MAX_CHARS = 1000                 # ~p90 of description length in this dataset


def _price_bucket(price):
    """USD price → 4 discrete buckets."""
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
    """Beauty-category sales rank → 4 discrete buckets (log split)."""
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
    """``salesRank`` is a category dict; prefer the Beauty entry."""
    sr = meta.get('salesRank')
    if not isinstance(sr, dict) or not sr:
        return None
    if 'Beauty' in sr:
        return sr['Beauty']
    return next(iter(sr.values()), None)


def _clean(s):
    """Unescape HTML entities (``&#39;``, ``&nbsp;``, ...) in text fields."""
    return html.unescape(s) if isinstance(s, str) else s


def build_item_prompt(meta):
    """
    Build a 6-field prompt for one item. Missing fields are skipped instead of
    being written as ``unknown``, so the embedding model never has to learn a
    placeholder vector.
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

    # Ping Ollama once before launching workers.
    try:
        test_resp = ollama.embeddings(model=MODEL, prompt='test')
        print(f'Ollama OK, model={MODEL}, dim={len(test_resp["embedding"])}')
    except Exception as e:
        print(f'Ollama connection failed: {e}')
        print(f'Run: ollama serve  &&  ollama pull {MODEL}')
        sys.exit(1)

    if os.path.exists(output_path) and not force:
        embeddings = np.load(output_path, allow_pickle=True).item()
        print(f'Resuming from existing file: {len(embeddings)} items')
    else:
        if force and os.path.exists(output_path):
            print('--force: ignoring existing embeddings, recomputing all')
        embeddings = {}

    items_to_process = [
        (asin, meta) for asin, meta in data['item_metas'].items()
        if data['item2id'].get(asin) not in embeddings
    ]
    print(f'#items to process: {len(items_to_process)}  workers: {N_WORKERS}')

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
                    tqdm.write(f'skip item {item_id}: {err}')
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
    print(f'\nDone: {len(embeddings)} items, {failed} failed')
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='ignore existing embeddings (use after prompt changes)')
    args = parser.parse_args()
    extract_all(force=args.force)
