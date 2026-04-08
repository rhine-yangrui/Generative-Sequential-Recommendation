"""
Train the T5 generative recommender on Semantic IDs.

    python train.py

The trained model is saved to ``checkpoints/best_model_t5.pt`` (override
with ``--ckpt``). Early stopping is driven by validation NDCG@10.
"""

import argparse
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.generative_rec import build_model, count_parameters
from model.inference import build_reverse_index, predict_topk_batch
from model.tokenizer import (
    K_LEVELS,
    PAD_TOKEN,
    VOCAB_SIZE,
    item_to_tokens,
    seq_to_t5_tokens,
)

# ── Hyper-parameters ─────────────────────────────────────────────────────
CONFIG = {
    'maxlen':      20,
    'batch_size':  512,    # comfortable on a 40 GB A100; lower if OOM
    'lr':          1e-4,
    'num_epochs':  200,
    'val_every':   2,      # full validation is expensive; evaluate every 2 epochs
    'patience':    10,
    'val_beam':    30,
    'val_batch':   256,
}
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_SEMANTIC_IDS_FILE = 'semantic_ids_rqvae.npy'
DEFAULT_CKPT_FILE         = 'checkpoints/best_model_t5.pt'

TARGET_LEN = len(K_LEVELS)
ENC_LEN    = CONFIG['maxlen'] * TARGET_LEN


class RecDataset(Dataset):
    """
    Encoder input  = flattened history token sequence (left-padded)
    Decoder labels = next item's Semantic ID tokens

    ``augment=True``  : sliding-window augmentation, every position becomes a sample
    ``augment=False`` : one sample per user (predict the last item only)
    """

    def __init__(self, user_seqs, semantic_ids, maxlen=20, augment=True):
        self.samples = []
        skipped = 0

        for user, seq in user_seqs.items():
            if len(seq) < 2:
                skipped += 1
                continue

            if augment:
                for t in range(1, len(seq)):
                    target_item = seq[t]
                    if target_item not in semantic_ids:
                        continue
                    history       = seq[:t]
                    history_tokens = seq_to_t5_tokens(history, semantic_ids, maxlen)
                    target_tokens  = item_to_tokens(semantic_ids[target_item])
                    self.samples.append((history_tokens, target_tokens))
            else:
                target_item = seq[-1]
                if target_item not in semantic_ids:
                    skipped += 1
                    continue
                history_tokens = seq_to_t5_tokens(seq[:-1], semantic_ids, maxlen)
                target_tokens  = item_to_tokens(semantic_ids[target_item])
                self.samples.append((history_tokens, target_tokens))

        if skipped:
            print(f'  skipped {skipped} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    input_ids = torch.tensor([h for h, _ in batch], dtype=torch.long)
    labels    = torch.tensor([t for _, t in batch], dtype=torch.long)
    attention_mask = (input_ids != PAD_TOKEN).long()
    return input_ids, attention_mask, labels


def evaluate_full_val(model, val_seqs, semantic_ids, device,
                      k=10, beam_width=30, batch_size=64):
    """
    Full all-rank validation pass: identical protocol to ``evaluate.py``,
    just with a smaller beam to keep per-epoch wall time bounded.
    """
    sid_to_item, sid_array, item_id_list = build_reverse_index(semantic_ids)
    model.eval()

    users   = sorted(val_seqs.keys())
    recalls = 0
    ndcgs   = 0.0
    n       = 0

    histories_buf = []
    targets_buf   = []

    def _flush():
        nonlocal recalls, ndcgs, n
        if not histories_buf:
            return
        results = predict_topk_batch(
            model, histories_buf, semantic_ids, sid_to_item,
            sid_array, item_id_list, k=k, beam_width=beam_width, device=device,
        )
        for recs, target in zip(results, targets_buf):
            n += 1
            if target in recs:
                rank = recs.index(target) + 1
                recalls += 1
                ndcgs   += 1.0 / math.log2(rank + 1)
        histories_buf.clear()
        targets_buf.clear()

    with torch.no_grad():
        for user in users:
            full_seq = val_seqs[user]
            histories_buf.append(full_seq[:-1])
            targets_buf.append(full_seq[-1])
            if len(histories_buf) >= batch_size:
                _flush()
        _flush()

    model.train()
    return (recalls / n, ndcgs / n) if n else (0.0, 0.0)


def train(semantic_ids_file=DEFAULT_SEMANTIC_IDS_FILE,
          ckpt_file=DEFAULT_CKPT_FILE):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device     : {device}')
    print(f'Semantic IDs: {semantic_ids_file}')
    print(f'Checkpoint  : {ckpt_file}')

    base_dir     = os.path.dirname(os.path.abspath(__file__))
    data         = pickle.load(open(os.path.join(base_dir, 'data/beauty_data.pkl'), 'rb'))
    semantic_ids = np.load(os.path.join(base_dir, 'embedding', semantic_ids_file),
                           allow_pickle=True).item()

    train_seqs = data['train']
    val_seqs   = data['val']
    print(f'#train users: {len(train_seqs)}  #val users: {len(val_seqs)}')

    print('Building training set (sliding-window augmentation)...')
    train_dataset = RecDataset(train_seqs, semantic_ids, CONFIG['maxlen'], augment=True)
    print(f'  #train samples: {len(train_dataset)}')

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, collate_fn=collate_fn,
        num_workers=4, pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
    )

    model = build_model().to(device)
    print(f'\n#parameters: {count_parameters(model) / 1e6:.1f}M')
    print(f'vocab size : {VOCAB_SIZE}')

    # Plain Adam, no weight decay, no LR scheduler. We tried CosineAnnealingLR
    # earlier and it killed the LR while validation was still improving.
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    os.makedirs(os.path.join(base_dir, 'checkpoints'), exist_ok=True)
    ckpt_path = os.path.join(base_dir, ckpt_file)
    best_val_ndcg  = 0.0
    patience_count = 0

    print(f'\nTraining for up to {CONFIG["num_epochs"]} epochs\n')
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        model.train()
        total_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids      = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels         = labels.to(device, non_blocking=True)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        if epoch % CONFIG['val_every'] == 0 or epoch == 1:
            val_recall, val_ndcg = evaluate_full_val(
                model, val_seqs, semantic_ids, device,
                k=10, beam_width=CONFIG['val_beam'],
                batch_size=CONFIG['val_batch'],
            )
            print(f'Epoch {epoch:3d}/{CONFIG["num_epochs"]}  '
                  f'train_loss={avg_train_loss:.4f}  '
                  f'val_R@10={val_recall:.4f}  val_N@10={val_ndcg:.4f}')

            if val_ndcg > best_val_ndcg:
                best_val_ndcg  = val_ndcg
                patience_count = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f'  saved best (val_N@10={best_val_ndcg:.4f})')
            else:
                patience_count += 1
                if patience_count >= CONFIG['patience']:
                    print(f'\nEarly stopping after {CONFIG["patience"]} stalled evals')
                    break
        else:
            print(f'Epoch {epoch:3d}/{CONFIG["num_epochs"]}  train_loss={avg_train_loss:.4f}')

    print(f'\nDone. best val_NDCG@10 = {best_val_ndcg:.4f}')
    print(f'Saved to {ckpt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--semantic-ids', default=DEFAULT_SEMANTIC_IDS_FILE,
                        help='Semantic ID file (relative to embedding/)')
    parser.add_argument('--ckpt', default=DEFAULT_CKPT_FILE,
                        help='checkpoint output path (relative to project root)')
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train(semantic_ids_file=args.semantic_ids, ckpt_file=args.ckpt)
