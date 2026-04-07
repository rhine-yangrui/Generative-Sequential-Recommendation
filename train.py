"""
训练脚本：T5 encoder-decoder 生成式推荐，对齐 TIGER 参考实现。

用法：
    python train.py

训练完成后模型保存至 checkpoints/best_model_t5_{num_epochs}ep.pt
"""

import math
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
from model.inference import build_reverse_index, predict_topk, predict_topk_batch

# ── 超参数（对齐 ../TIGER/model/main.py）─────────────────────────────────
CONFIG = {
    'maxlen':      20,
    'batch_size':  256,
    'lr':          1e-4,
    'num_epochs':  200,     # 对齐 TIGER；E8 (100ep+cosine) 证明在低 LR 下被截停
    'val_every':   2,       # 全量 val 较慢，每 2 epoch 评估一次
    'patience':    10,      # 对齐 TIGER
    'val_beam':    30,      # 对齐 TIGER 训练评估 beam_size
    'val_batch':   256,     # A100 40GB 够；OOM 就降到 128
}
# ─────────────────────────────────────────────────────────────────────────

ACTIVE_SEMANTIC_IDS = 'semantic_ids_rqvae_3kep.npy'

# Checkpoint 名自动带上 sids 的 tag，避免多套 sids 互相覆盖。
#   semantic_ids_rqvae.npy       → ''              → best_model_t5_200ep.pt
#   semantic_ids_rqvae_3kep.npy  → '_3kep'         → best_model_t5_200ep_3kep.pt
#   semantic_ids_rqvae_t5_3kep.npy → '_t5_3kep'    → best_model_t5_200ep_t5_3kep.pt
_sid_stem = os.path.splitext(ACTIVE_SEMANTIC_IDS)[0]          # semantic_ids_rqvae[_3kep]
CKPT_TAG  = _sid_stem[len('semantic_ids_rqvae'):]             # '' | '_3kep' | '_t5_3kep'

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


def evaluate_full_val(model, val_seqs, semantic_ids, device,
                      k=10, beam_width=30, batch_size=64):
    """
    全量 val 评估，对齐 SASRec / TIGER：固定顺序、所有用户、与 test 一致的 beam。
    使用 predict_topk_batch 批量化，避免单用户串行 generate。
    返回 (Recall@k, NDCG@k)。
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

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, collate_fn=collate_fn,
        num_workers=4, pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
    )

    model = build_model().to(device)
    print(f'\n模型参数量: {count_parameters(model) / 1e6:.1f}M')
    print(f'词表大小: {VOCAB_SIZE}')

    # 对齐 TIGER：Adam，无 weight_decay，常数 LR（不用 scheduler）
    # E8 教训：CosineAnnealingLR 把 LR 退到 1e-5 后 val 仍单调上升，纯被掐死
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    os.makedirs(os.path.join(base_dir, 'checkpoints'), exist_ok=True)
    ckpt_path = os.path.join(base_dir, f'checkpoints/best_model_t5_{CONFIG["num_epochs"]}ep{CKPT_TAG}.pt')
    best_val_ndcg  = 0.0
    patience_count = 0

    print(f'\n开始训练，共 {CONFIG["num_epochs"]} epochs\n')
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
                print(f'  ✓ 保存最优模型 (val_N@10={best_val_ndcg:.4f})')
            else:
                patience_count += 1
                if patience_count >= CONFIG['patience']:
                    print(f'\nEarly stopping（连续 {CONFIG["patience"]} 次未提升）')
                    break
        else:
            print(f'Epoch {epoch:3d}/{CONFIG["num_epochs"]}  train_loss={avg_train_loss:.4f}')

    print(f'\n训练完成，最优 val_NDCG@10={best_val_ndcg:.4f}')
    print(f'模型已保存至 {ckpt_path}')


if __name__ == '__main__':
    _random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train()
