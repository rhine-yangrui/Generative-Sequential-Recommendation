"""
训练脚本：T5 encoder-decoder 生成式推荐，对齐 TIGER 参考实现。

用法：
    python train.py

训练完成后模型保存至 checkpoints/best_model_t5_{num_epochs}ep.pt
"""

import os
import pickle
import random as _random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.tokenizer import (
    seq_to_t5_tokens, item_to_tokens, VOCAB_SIZE, PAD_TOKEN, K_LEVELS
)
from model.generative_rec import build_model, count_parameters
from model.inference import build_reverse_index, predict_topk

# ── 超参数（对齐 ../TIGER/model/main.py）─────────────────────────────────
CONFIG = {
    'maxlen':      20,
    'batch_size':  256,
    'lr':          1e-4,
    'num_epochs':  200,
    'val_every':   10,
    'patience':    6,
}
# ─────────────────────────────────────────────────────────────────────────

ACTIVE_SEMANTIC_IDS = 'semantic_ids_rqvae.npy'
TARGET_LEN = len(K_LEVELS)
ENC_LEN    = CONFIG['maxlen'] * TARGET_LEN


class RecDataset(Dataset):
    """
    T5 推荐数据集：history → encoder 输入，target item Semantic ID → decoder labels。

    augment=True（训练）：滑动窗口，每个位置都生成一条样本。
    augment=False（验证）：每用户只预测最后一个 item。

    每条样本固定形状：
        history_tokens: ENC_LEN 个 token，左 PAD
        target_tokens:  TARGET_LEN 个 token
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
            print(f'  跳过 {skipped} 个样本')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    输出：
      input_ids:      (B, ENC_LEN)
      attention_mask: (B, ENC_LEN)，PAD 位置为 0
      labels:         (B, TARGET_LEN)，T5 内部自动 shift right
    """
    input_ids = torch.tensor([h for h, _ in batch], dtype=torch.long)
    labels    = torch.tensor([t for _, t in batch], dtype=torch.long)
    attention_mask = (input_ids != PAD_TOKEN).long()
    return input_ids, attention_mask, labels


def evaluate_recall_subset(model, val_seqs, semantic_ids, device,
                           k=10, n_users=500, beam_width=20):
    """val 子集 Recall@k，用于 early stopping。"""
    sid_to_item, sid_array, item_id_list = build_reverse_index(semantic_ids)
    model.eval()
    users   = list(val_seqs.keys())
    sampled = _random.sample(users, min(n_users, len(users)))
    hits    = 0

    with torch.no_grad():
        for user in sampled:
            full_seq = val_seqs[user]
            target   = full_seq[-1]
            history  = full_seq[:-1]

            recs = predict_topk(
                model, history, semantic_ids, sid_to_item, sid_array,
                item_id_list, k=k, beam_width=beam_width, device=device
            )
            if target in recs:
                hits += 1

    model.train()
    return hits / len(sampled) if sampled else 0.0


def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'使用设备: {device}')

    base_dir     = os.path.dirname(os.path.abspath(__file__))
    data         = pickle.load(open(os.path.join(base_dir, 'data/beauty_data.pkl'), 'rb'))
    semantic_ids = np.load(os.path.join(base_dir, 'embedding', ACTIVE_SEMANTIC_IDS),
                           allow_pickle=True).item()

    train_seqs = data['train']
    val_seqs   = data['val']
    print(f'训练用户数: {len(train_seqs)},  验证用户数: {len(val_seqs)}')

    print('构建训练集（滑动窗口增强）...')
    train_dataset = RecDataset(train_seqs, semantic_ids, CONFIG['maxlen'], augment=True)
    print(f'  训练样本数: {len(train_dataset)}')

    print('构建验证集...')
    val_dataset = RecDataset(val_seqs, semantic_ids, CONFIG['maxlen'], augment=False)
    print(f'  验证样本数: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, collate_fn=collate_fn, num_workers=2)

    model = build_model().to(device)
    print(f'\n模型参数量: {count_parameters(model) / 1e6:.1f}M')
    print(f'词表大小: {VOCAB_SIZE}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-5)

    os.makedirs(os.path.join(base_dir, 'checkpoints'), exist_ok=True)
    ckpt_path = os.path.join(base_dir, f'checkpoints/best_model_t5_{CONFIG["num_epochs"]}ep.pt')
    best_val_recall = 0.0
    patience_count  = 0

    print(f'\n开始训练，共 {CONFIG["num_epochs"]} epochs\n')
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        model.train()
        total_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        if epoch % CONFIG['val_every'] == 0 or epoch == 1:
            val_recall = evaluate_recall_subset(
                model, val_seqs, semantic_ids, device, n_users=500, beam_width=20
            )
            print(f'Epoch {epoch:3d}/{CONFIG["num_epochs"]}  '
                  f'train_loss={avg_train_loss:.4f}  val_Recall@10={val_recall:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}')

            if val_recall > best_val_recall:
                best_val_recall = val_recall
                patience_count  = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f'  ✓ 保存最优模型 (val_Recall@10={best_val_recall:.4f})')
            else:
                patience_count += 1
                if patience_count >= CONFIG['patience']:
                    print(f'\nEarly stopping（连续 {CONFIG["patience"]} 次未提升）')
                    break
        else:
            print(f'Epoch {epoch:3d}/{CONFIG["num_epochs"]}  train_loss={avg_train_loss:.4f}')

    print(f'\n训练完成，最优 val_Recall@10={best_val_recall:.4f}')
    print(f'模型已保存至 {ckpt_path}')


if __name__ == '__main__':
    _random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train()
