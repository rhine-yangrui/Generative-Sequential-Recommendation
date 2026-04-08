"""
Train the SASRec baseline on the same data split and evaluation protocol as
the generative model (all-rank Recall@K / NDCG@K).

    python baseline/sasrec_train.py

Saves ``checkpoints/sasrec_best.pt`` and prints test metrics at the end.
"""

import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from baseline.sasrec import SASRec
except ModuleNotFoundError:
    from sasrec import SASRec

CONFIG = {
    'maxlen':       50,
    'hidden_size':  64,
    'num_layers':   2,
    'num_heads':    1,
    'dropout':      0.5,
    'lr':           5e-4,
    'batch_size':   256,
    'num_epochs':   200,
    'patience':     20,
    'val_every':    5,
    'num_neg':      1,
    'weight_decay': 0,
}
K_LIST = [5, 10]


# ── Dataset ───────────────────────────────────────────────────────────────

class SASRecDataset(Dataset):
    """
    Sliding-window samples: at every position the prefix is the input, the
    next item is the positive, and one random uninteracted item is the
    negative for the BPR objective.
    """

    def __init__(self, user_seqs, num_items, maxlen=50, num_neg=1):
        self.samples   = []
        self.num_items = num_items
        self.num_neg   = num_neg

        for user, seq in user_seqs.items():
            if len(seq) < 1:
                continue
            interacted = set(seq)
            self.samples.append(([], seq[0], interacted))   # empty history
            for i in range(1, len(seq)):
                input_seq = seq[:i][-maxlen:]
                pos_item  = seq[i]
                self.samples.append((input_seq, pos_item, interacted))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, pos_item, interacted = self.samples[idx]
        neg_items = []
        while len(neg_items) < self.num_neg:
            neg = random.randint(1, self.num_items)
            if neg not in interacted:
                neg_items.append(neg)
        return input_seq, pos_item, neg_items[0]


def collate_sasrec(batch):
    maxlen = max(len(s) for s, _, _ in batch)
    maxlen = max(maxlen, 1)
    padded = []
    for s, _, _ in batch:
        pad_len = maxlen - len(s)
        padded.append([0] * pad_len + list(s))
    input_ids = torch.tensor(padded, dtype=torch.long)
    pos_items = torch.tensor([p for _, p, _ in batch], dtype=torch.long)
    neg_items = torch.tensor([n for _, _, n in batch], dtype=torch.long)
    return input_ids, pos_items, neg_items


# ── Evaluation (all-rank) ────────────────────────────────────────────────

def evaluate_sasrec_allrank(model, test_seqs, num_items, device,
                            maxlen=50, k_list=K_LIST):
    """All-rank evaluation: score every item, take top-K, compute Recall/NDCG."""
    model.eval()
    metrics = {k: {'Recall': [], 'NDCG': []} for k in k_list}
    max_k   = max(k_list)

    all_item_ids = torch.arange(1, num_items + 1,
                                dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for user, full_seq in tqdm(test_seqs.items(), desc='Evaluating SASRec'):
            target  = full_seq[-1]
            history = full_seq[:-1]

            input_seq = history[-maxlen:]
            pad_len   = maxlen - len(input_seq)
            input_seq_padded = [0] * pad_len + list(input_seq)
            input_ids = torch.tensor([input_seq_padded], dtype=torch.long).to(device)

            scores = model.predict(input_ids, all_item_ids).squeeze(0)
            _, topk_indices = torch.topk(scores, max_k)
            topk_items = (topk_indices + 1).tolist()   # back to 1-indexed item ids

            for k in k_list:
                topk = topk_items[:k]
                if target in topk:
                    rank = topk.index(target) + 1
                    metrics[k]['Recall'].append(1)
                    metrics[k]['NDCG'].append(1 / math.log2(rank + 1))
                else:
                    metrics[k]['Recall'].append(0)
                    metrics[k]['NDCG'].append(0.0)

    model.train()
    return {k: {m: np.mean(v) for m, v in mv.items()} for k, mv in metrics.items()}


# ── Training ─────────────────────────────────────────────────────────────

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data     = pickle.load(open(os.path.join(base_dir, 'data/beauty_data.pkl'), 'rb'))

    num_items  = len(data['item2id'])
    train_seqs = data['train']
    val_seqs   = data['val']
    test_seqs  = data['test']
    print(f'#items: {num_items}  #train users: {len(train_seqs)}')

    train_dataset = SASRecDataset(train_seqs, num_items,
                                  CONFIG['maxlen'], CONFIG['num_neg'])
    train_loader  = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                               shuffle=True, collate_fn=collate_sasrec, num_workers=2)

    model = SASRec(
        num_items   = num_items,
        hidden_size = CONFIG['hidden_size'],
        num_layers  = CONFIG['num_layers'],
        num_heads   = CONFIG['num_heads'],
        dropout     = CONFIG['dropout'],
        maxlen      = CONFIG['maxlen'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'#parameters: {total_params / 1e6:.2f}M')

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'],
                                 weight_decay=CONFIG['weight_decay'])
    # Use logsigmoid directly to avoid log-of-tiny-sigmoid underflow.
    bpr_loss  = lambda pos_s, neg_s: -F.logsigmoid(pos_s - neg_s).mean()

    os.makedirs(os.path.join(base_dir, 'checkpoints'), exist_ok=True)
    best_recall10  = 0.0
    patience_count = 0

    print(f'\nTraining for up to {CONFIG["num_epochs"]} epochs\n')
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        total_loss = 0.0
        for input_ids, pos_items, neg_items in train_loader:
            input_ids = input_ids.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            candidates = torch.stack([pos_items, neg_items], dim=1)  # (B, 2)
            scores     = model.predict(input_ids, candidates)         # (B, 2)
            pos_scores, neg_scores = scores[:, 0], scores[:, 1]

            loss = bpr_loss(pos_scores, neg_scores)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if epoch % CONFIG['val_every'] == 0 or epoch == 1:
            summary  = evaluate_sasrec_allrank(model, val_seqs, num_items, device,
                                               CONFIG['maxlen'])
            recall10 = summary[10]['Recall']
            print(f'Epoch {epoch:3d}  loss={avg_loss:.4f}  '
                  f'R@5={summary[5]["Recall"]:.4f}  N@5={summary[5]["NDCG"]:.4f}  '
                  f'R@10={recall10:.4f}  N@10={summary[10]["NDCG"]:.4f}')

            if recall10 > best_recall10:
                best_recall10 = recall10
                patience_count = 0
                torch.save(model.state_dict(),
                           os.path.join(base_dir, 'checkpoints/sasrec_best.pt'))
                print(f'  saved best (R@10={best_recall10:.4f})')
            else:
                patience_count += 1
                if patience_count >= CONFIG['patience']:
                    print(f'\nEarly stopping')
                    break
        else:
            print(f'Epoch {epoch:3d}  loss={avg_loss:.4f}')

    print(f'\n{"="*60}')
    print(f'  SASRec final (best val R@10 = {best_recall10:.4f})')
    print(f'{"="*60}')
    model.load_state_dict(torch.load(
        os.path.join(base_dir, 'checkpoints/sasrec_best.pt'),
        map_location=device, weights_only=True))
    final = evaluate_sasrec_allrank(model, test_seqs, num_items, device, CONFIG['maxlen'])
    for k in K_LIST:
        print(f'  R@{k:<2} = {final[k]["Recall"]:.4f}   N@{k:<2} = {final[k]["NDCG"]:.4f}')


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train()
