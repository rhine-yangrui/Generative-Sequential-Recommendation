"""
训练脚本：在 Colab（T4/A100）上运行。

用法：
    python train.py

训练完成后模型保存至 checkpoints/best_model.pt
"""

import os
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from model.tokenizer import (
    seq_to_tokens, item_to_tokens, tokens_to_semantic_id,
    VOCAB_SIZE, PAD_TOKEN, BOS_TOKEN
)
from model.generative_rec import build_model, count_parameters

# ── 超参数 ────────────────────────────────────────────────────────────────
CONFIG = {
    'maxlen':      50,      # 用户历史最多保留多少个 item
    'batch_size':  128,     # 滑动窗口后样本量大幅增加，可用更大 batch
    'lr':          1e-3,
    'num_epochs':  30,      # 样本多了不需要那么多 epoch
    'val_every':   3,
    'patience':    10,      # early stopping：val loss 连续多少次不降则停止
}
# ─────────────────────────────────────────────────────────────────────────


class RecDataset(Dataset):
    """
    推荐数据集，支持滑动窗口数据增强。

    augment=True（训练集）：
      序列 [a, b, c, d] 生成 3 条样本：
        [a] → b,  [a,b] → c,  [a,b,c] → d
      约 5-10 倍于原始样本量。

    augment=False（验证集）：
      每个用户只预测最后一个 item（与测试集评估方式一致）。

    训练只对最后 3 个 token（目标 item 的 c1/c2/c3）计算 loss，
    历史部分的 loss 用 -100 mask 掉。
    """
    def __init__(self, user_seqs, semantic_ids, maxlen=50, augment=True):
        self.samples = []
        skipped = 0

        for user, seq in user_seqs.items():
            if len(seq) < 2:
                skipped += 1
                continue

            if augment:
                # 滑动窗口：每个位置都生成一条样本
                for t in range(1, len(seq)):
                    target_item = seq[t]
                    if target_item not in semantic_ids:
                        continue
                    history      = seq[:t]
                    input_tokens = seq_to_tokens(history, semantic_ids, maxlen)
                    target_tokens = item_to_tokens(semantic_ids[target_item])
                    full_tokens  = input_tokens + target_tokens
                    self.samples.append((full_tokens, len(input_tokens)))
            else:
                # 只预测最后一个 item
                target_item = seq[-1]
                if target_item not in semantic_ids:
                    skipped += 1
                    continue
                input_tokens  = seq_to_tokens(seq[:-1], semantic_ids, maxlen)
                target_tokens = item_to_tokens(semantic_ids[target_item])
                full_tokens   = input_tokens + target_tokens
                self.samples.append((full_tokens, len(input_tokens)))

        if skipped:
            print(f'  跳过 {skipped} 个样本（序列过短或缺少 semantic_id）')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    将一个 batch 的样本 padding 到相同长度。

    input_ids:  (B, max_len)，PAD_TOKEN 填充
    labels:     (B, max_len)，历史部分 mask 为 -100，只有目标 3 个 token 计算 loss
    """
    token_seqs   = [torch.tensor(tokens, dtype=torch.long) for tokens, _ in batch]
    history_lens = [hist_len for _, hist_len in batch]

    input_ids = pad_sequence(token_seqs, batch_first=True, padding_value=PAD_TOKEN)

    labels = torch.full_like(input_ids, -100)
    for i, (tokens, hist_len) in enumerate(zip(token_seqs, history_lens)):
        target_start = hist_len
        target_end   = hist_len + 3
        labels[i, target_start:target_end] = tokens[target_start:target_end]

    return input_ids, labels


def evaluate_val_loss(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels    = labels.to(device)
            outputs   = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            n_batches  += 1
    model.train()
    return total_loss / n_batches if n_batches > 0 else float('inf')


def train():
    # ── 设备 ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'使用设备: {device}')

    # ── 加载数据 ──────────────────────────────────────────────────────────
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    data         = pickle.load(open(os.path.join(base_dir, 'data/beauty_data.pkl'), 'rb'))
    semantic_ids = np.load(os.path.join(base_dir, 'embedding/semantic_ids.npy'),
                           allow_pickle=True).item()

    train_seqs = data['train']
    val_seqs   = data['val']
    print(f'训练用户数: {len(train_seqs)},  验证用户数: {len(val_seqs)}')

    # ── 构建 Dataset ──────────────────────────────────────────────────────
    print('构建训练集（滑动窗口增强）...')
    train_dataset = RecDataset(train_seqs, semantic_ids, CONFIG['maxlen'], augment=True)
    print(f'  训练样本数: {len(train_dataset)}')

    print('构建验证集...')
    val_dataset = RecDataset(val_seqs, semantic_ids, CONFIG['maxlen'], augment=False)
    print(f'  验证样本数: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    # ── 模型 ──────────────────────────────────────────────────────────────
    model = build_model().to(device)
    print(f'\n模型参数量: {count_parameters(model) / 1e6:.1f}M')
    print(f'词表大小: {VOCAB_SIZE}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-5)

    # ── 训练循环 ──────────────────────────────────────────────────────────
    os.makedirs(os.path.join(base_dir, 'checkpoints'), exist_ok=True)
    best_val_loss  = float('inf')
    patience_count = 0

    print(f'\n开始训练，共 {CONFIG["num_epochs"]} epochs\n')
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        model.train()
        total_loss = 0.0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        if epoch % CONFIG['val_every'] == 0 or epoch == 1:
            val_loss = evaluate_val_loss(model, val_loader, device)
            print(f'Epoch {epoch:3d}/{CONFIG["num_epochs"]}  '
                  f'train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                torch.save(model.state_dict(),
                           os.path.join(base_dir, 'checkpoints/best_model.pt'))
                print(f'  ✓ 保存最优模型 (val_loss={best_val_loss:.4f})')
            else:
                patience_count += 1
                if patience_count >= CONFIG['patience']:
                    print(f'\nEarly stopping（连续 {CONFIG["patience"]} 次 val loss 未下降）')
                    break
        else:
            print(f'Epoch {epoch:3d}/{CONFIG["num_epochs"]}  train_loss={avg_train_loss:.4f}')

    print(f'\n训练完成，最优 val_loss={best_val_loss:.4f}')
    print(f'模型已保存至 checkpoints/best_model.pt')


if __name__ == '__main__':
    train()
