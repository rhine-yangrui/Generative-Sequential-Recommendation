"""
训练脚本：在 Colab（T4/A100）上运行。

用法：
    python train.py

训练完成后模型保存至 checkpoints/best_model_rqvae_{num_epochs}ep.pt
"""

import os
import pickle
import random as _random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from model.tokenizer import (
    seq_to_tokens, item_to_tokens, VOCAB_SIZE, PAD_TOKEN, K_LEVELS
)
from model.generative_rec import build_model, count_parameters
from model.inference import build_reverse_index, predict_topk

# ── 超参数 ────────────────────────────────────────────────────────────────
CONFIG = {
    'maxlen':      20,      # 对齐 TIGER 参考实现
    'batch_size':  256,     # 对齐 TIGER 参考实现
    'lr':          1e-4,    # 对齐 TIGER 参考实现（旧 1e-3 太激进）
    'num_epochs':  200,     # 对齐 TIGER 参考实现
    'val_every':   10,      # 拉长以适应 200 epoch
    'patience':    6,       # 连续 6 次 val Recall@10 不提升则停止
}
# ─────────────────────────────────────────────────────────────────────────

ACTIVE_SEMANTIC_IDS = 'semantic_ids_rqvae.npy'
TARGET_LEN = len(K_LEVELS)


class RecDataset(Dataset):
    """
    推荐数据集，支持滑动窗口数据增强。

    augment=True（训练集）：
      序列 [a, b, c, d] 生成 3 条样本：
        [a] → b,  [a,b] → c,  [a,b,c] → d
      约 5-10 倍于原始样本量。

    augment=False（验证集）：
      每个用户只预测最后一个 item（与测试集评估方式一致）。

    训练只对最后 TARGET_LEN 个 token（目标 item 的 Semantic ID，不含 EOS）计算 loss，
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
                    # full_tokens = [BOS, ..., semantic_id^T]
                    # target 不追加 EOS，只监督 Semantic ID token。
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
                # full_tokens = [BOS, ..., semantic_id^T]
                # target 不追加 EOS，只监督 Semantic ID token。
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
    labels:     (B, max_len)，历史部分 mask 为 -100，只有目标 Semantic ID token 计算 loss
    """
    token_seqs   = [torch.tensor(tokens, dtype=torch.long) for tokens, _ in batch]
    history_lens = [hist_len for _, hist_len in batch]

    input_ids = pad_sequence(token_seqs, batch_first=True, padding_value=PAD_TOKEN)

    labels = torch.full_like(input_ids, -100)
    for i, (tokens, hist_len) in enumerate(zip(token_seqs, history_lens)):
        target_start = hist_len
        target_end   = hist_len + TARGET_LEN
        labels[i, target_start:target_end] = tokens[target_start:target_end]

    return input_ids, labels


def evaluate_recall_subset(model, val_seqs, semantic_ids, device,
                           k=10, n_users=500, beam_width=20):
    """
    在 val 集随机子集上计算 Recall@10，用于 early stopping。
    """
    sid_to_item, sid_array, item_id_list = build_reverse_index(semantic_ids)
    model.eval()
    users = list(val_seqs.keys())
    sampled = _random.sample(users, min(n_users, len(users)))
    hits = 0

    with torch.no_grad():
        for user in sampled:
            full_seq = val_seqs[user]
            target = full_seq[-1]
            history = full_seq[:-1]

            recs = predict_topk(
                model, history, semantic_ids, sid_to_item, sid_array,
                item_id_list, k=k, beam_width=beam_width, device=device
            )
            if target in recs:
                hits += 1

    model.train()
    return hits / len(sampled) if sampled else 0.0


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
    # Active Semantic ID file: RQ-VAE on top of nomic embeddings.
    semantic_ids = np.load(os.path.join(base_dir, 'embedding', ACTIVE_SEMANTIC_IDS),
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
    # ── 模型 ──────────────────────────────────────────────────────────────
    model = build_model().to(device)
    print(f'\n模型参数量: {count_parameters(model) / 1e6:.1f}M')
    print(f'词表大小: {VOCAB_SIZE}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-5)

    # ── 训练循环 ──────────────────────────────────────────────────────────
    os.makedirs(os.path.join(base_dir, 'checkpoints'), exist_ok=True)
    ckpt_path = os.path.join(base_dir, f'checkpoints/best_model_rqvae_{CONFIG["num_epochs"]}ep.pt')
    best_val_recall = 0.0
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
            val_recall = evaluate_recall_subset(
                model, val_seqs, semantic_ids, device, n_users=500, beam_width=20
            )
            print(f'Epoch {epoch:3d}/{CONFIG["num_epochs"]}  '
                  f'train_loss={avg_train_loss:.4f}  val_Recall@10={val_recall:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}')

            if val_recall > best_val_recall:
                best_val_recall = val_recall
                patience_count = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f'  ✓ 保存最优模型 (val_Recall@10={best_val_recall:.4f})')
            else:
                patience_count += 1
                if patience_count >= CONFIG['patience']:
                    print(f'\nEarly stopping（连续 {CONFIG["patience"]} 次 val Recall@10 未提升）')
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
