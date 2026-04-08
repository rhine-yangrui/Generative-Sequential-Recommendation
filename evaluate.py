"""
Evaluate the trained T5 generative recommender with the all-rank protocol.

For every test user we run constrained beam search to get top-K recommendations
and report Recall@K / NDCG@K with K in {5, 10}.

    python evaluate.py
"""

import argparse
import math
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from model.generative_rec import build_model
from model.inference import build_reverse_index, predict_topk_batch

K_LIST     = [5, 10]
BEAM_WIDTH = 50
BATCH_SIZE = 256        # batched beam search; lower if VRAM is tight
DEFAULT_SEMANTIC_IDS_FILE = 'semantic_ids_rqvae.npy'
DEFAULT_CKPT              = 'checkpoints/best_model_t5.pt'


def compute_metrics(recommended_items, target, k_list):
    """Per-user metric snapshot for one prediction."""
    results = {}
    for k in k_list:
        topk = recommended_items[:k]
        if target in topk:
            rank = topk.index(target) + 1
            results[k] = {'Recall': 1, 'NDCG': 1 / math.log2(rank + 1)}
        else:
            results[k] = {'Recall': 0, 'NDCG': 0.0}
    return results


def evaluate(model, test_seqs, semantic_ids, sid_to_item, sid_array,
             item_id_list, device, k_list=K_LIST,
             beam_width=BEAM_WIDTH, batch_size=BATCH_SIZE):
    """Run batched beam search across the whole test split."""
    model.eval()
    metrics = {k: {'Recall': [], 'NDCG': []} for k in k_list}
    max_k   = max(k_list)

    histories_buf = []
    targets_buf   = []
    pbar = tqdm(test_seqs.items(), desc='Evaluating')

    def _flush():
        if not histories_buf:
            return
        results = predict_topk_batch(
            model, histories_buf, semantic_ids, sid_to_item,
            sid_array, item_id_list,
            k=max_k, beam_width=beam_width, device=device,
        )
        for recs, target in zip(results, targets_buf):
            result = compute_metrics(recs, target, k_list)
            for k in k_list:
                metrics[k]['Recall'].append(result[k]['Recall'])
                metrics[k]['NDCG'].append(result[k]['NDCG'])
        histories_buf.clear()
        targets_buf.clear()

    for user, full_seq in pbar:
        histories_buf.append(full_seq[:-1])
        targets_buf.append(full_seq[-1])
        if len(histories_buf) >= batch_size:
            _flush()
    _flush()

    return {k: {m: np.mean(v) for m, v in mv.items()} for k, mv in metrics.items()}


def print_results(summary, model_name='Our Model'):
    print(f'\n{"="*60}')
    print(f'  {model_name}')
    print(f'  Protocol: all-rank Recall@K / NDCG@K')
    print(f'{"="*60}')
    print(f'  {"K":>4}  {"Recall@K":>10}  {"NDCG@K":>10}')
    print(f'  {"-"*35}')
    for k in sorted(summary.keys()):
        print(f'  {k:>4}  {summary[k]["Recall"]:>10.4f}  {summary[k]["NDCG"]:>10.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--semantic-ids', default=DEFAULT_SEMANTIC_IDS_FILE,
                        help='Semantic ID file (relative to embedding/)')
    parser.add_argument('--ckpt', default=DEFAULT_CKPT,
                        help='checkpoint path (relative to project root)')
    args = parser.parse_args()

    np.random.seed(42)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    data         = pickle.load(open(os.path.join(base_dir, 'data/beauty_data.pkl'), 'rb'))
    semantic_ids = np.load(os.path.join(base_dir, 'embedding', args.semantic_ids),
                           allow_pickle=True).item()
    print(f'Semantic IDs: {args.semantic_ids}')

    ckpt_path = os.path.join(base_dir, args.ckpt)
    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f'Loaded checkpoint: {args.ckpt}')

    sid_to_item, sid_array, item_id_list = build_reverse_index(semantic_ids)

    print(f'\nEvaluating on {len(data["test"])} test users...')
    summary = evaluate(
        model,
        test_seqs    = data['test'],
        semantic_ids = semantic_ids,
        sid_to_item  = sid_to_item,
        sid_array    = sid_array,
        item_id_list = item_id_list,
        device       = device,
        beam_width   = BEAM_WIDTH,
    )

    print_results(summary, model_name='Generative Rec (Semantic ID + T5)')
