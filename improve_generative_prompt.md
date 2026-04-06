# Task: Improve Generative Recommender — Four Changes

We are improving an existing generative sequential recommendation system based on the TIGER (NeurIPS 2023) paper. The codebase uses GPT-2 decoder-only model with Semantic IDs. Please implement the four improvements below **in order**. Read each section fully before editing any file.

---

## Context: Current Architecture

- `model/tokenizer.py`: defines K_LEVELS, VOCAB_SIZE, token offsets, `seq_to_tokens()`, `item_to_tokens()`
- `model/inference.py`: beam search via `model.generate()` — currently **unconstrained**
- `train.py`: training loop with early stopping on **val_loss** (LM loss)
- `embedding/build_semantic_ids.py`: hierarchical k-means (currently uses `item_embeddings_raw.npy`)
- `embedding/item_embeddings_raw_nomic.npy`: **already extracted**, 768-dim nomic-embed-text embeddings

Current K_LEVELS in tokenizer = [4, 64, 256], VOCAB_SIZE = 327. We will change to [4, 16, 256].

---

## Change 1: Rebuild Semantic IDs with nomic embeddings and K_LEVELS = [4, 16, 256]

**Motivation**: The current embedding (qwen2:7b) gives cluster purity ~70.6%. nomic-embed-text gives ~91.65%, and K_LEVELS [4, 16, 256] matches the original TIGER paper.

### 1a. Update `model/tokenizer.py`

Change K_LEVELS from `[4, 64, 256]` to `[4, 16, 256]`. Everything else (LEVEL_OFFSETS, VOCAB_SIZE, token functions) derives automatically from K_LEVELS, so no other changes needed in this file.

New values after this change:
- `K_LEVELS = [4, 16, 256]`
- `LEVEL_OFFSETS = [0, 4, 20]`
- `VOCAB_SIZE = 4 + 16 + 256 + 3 = 279`
- `BOS_TOKEN = 276`, `EOS_TOKEN = 277`, `PAD_TOKEN = 278`

### 1b. Create `embedding/build_semantic_ids_nomic.py`

Create a new file (do not modify the existing `build_semantic_ids.py`). This new script:
- Reads from `embedding/item_embeddings_raw_nomic.npy` (already exists)
- Uses `K_LEVELS = [4, 16, 256]`
- Uses the same hierarchical MiniBatchKMeans approach as the original script
- **Removes** the L3 distance-sorting hack (the original used distance sorting only because L2=64 was needed to avoid overflow; with nomic's better clustering, standard k-means at L3 is fine, but since K3=256 is the capacity we still need zero-collision guarantee — keep distance sorting for safety)
- Saves output to `embedding/semantic_ids_nomic.npy`

The script should be a copy of `build_semantic_ids.py` with these changes:
```python
# Change these lines:
K_LEVELS = [4, 16, 256]   # match tokenizer
raw_path    = os.path.join(emb_dir, 'item_embeddings_raw_nomic.npy')
output_path = os.path.join(emb_dir, 'semantic_ids_nomic.npy')
```

### 1c. Update `train.py` and `evaluate.py`

Change the semantic_ids loading path from `embedding/semantic_ids.npy` to `embedding/semantic_ids_nomic.npy` in both files. Add a comment indicating which file is active.

---

## Change 2: Add EOS separator tokens between items in the sequence

**Motivation**: Without EOS separators, the model must infer item boundaries from positional patterns alone (every 3 tokens = one item). Explicit EOS tokens make this unambiguous and match TIGER's sequence design.

### 2a. Update `model/tokenizer.py` — `seq_to_tokens()`

Change `seq_to_tokens()` to insert `EOS_TOKEN` after each item's 3 tokens:

```python
def seq_to_tokens(item_seq, semantic_ids, maxlen=50):
    """
    把用户的 item ID 序列转成 token 序列（含 BOS，每个 item 后加 EOS 分隔符）。

    序列结构：[BOS, c1¹, c2¹, c3¹, EOS, c1², c2², c3², EOS, ...]
    每个 item 占 4 个 token（含 EOS），最大序列长度 = 1 + 4 * maxlen
    """
    tokens = [BOS_TOKEN]
    for item_id in item_seq[-maxlen:]:
        if item_id in semantic_ids:
            tokens.extend(item_to_tokens(semantic_ids[item_id]))
            tokens.append(EOS_TOKEN)   # item 边界标记
    return tokens
```

### 2b. Update `train.py` — `RecDataset.__init__()` and `collate_fn()`

In `RecDataset.__init__()`, the `hist_len` (length of input_tokens) now includes EOS tokens. No change needed to how it's computed — `len(input_tokens)` is still correct since `seq_to_tokens()` now returns longer sequences automatically.

However, the target tokens are **still only 3**: `[c1, c2, c3]` of the target item (no EOS appended to target, because we only supervise item code prediction, not the EOS). So `item_to_tokens()` stays unchanged and `target_tokens` is still 3 tokens.

Update the `full_tokens` construction comment to clarify:
```python
# full_tokens = [BOS, c1¹, c2¹, c3¹, EOS, ..., c1ᵀ, c2ᵀ, c3ᵀ]
# The target has no trailing EOS — we only supervise the 3 code tokens.
full_tokens = input_tokens + target_tokens
self.samples.append((full_tokens, len(input_tokens)))
```

The `collate_fn()` and loss masking logic need no changes (labels still cover positions `hist_len` to `hist_len+2`).

### 2c. Update `model/generative_rec.py` — `n_positions`

With EOS separators, the max sequence length becomes `1 + 4 * 50 = 201` tokens (instead of 151). Update `n_positions` from 512 to 512 (already sufficient, no change needed, but add a comment):

```python
n_positions=512,   # 1 BOS + 4*maxlen items (c1/c2/c3/EOS each) = 201 max, 512 sufficient
```

### 2d. Update `model/inference.py` — `predict_topk()`

The input sequence now contains EOS separators, but we still generate only 3 new tokens (c1, c2, c3 of the target — no EOS generation needed since inference stops after 3 tokens). No change needed to `max_new_tokens=3`. The constrained beam search in Change 3 will handle the rest.

---

## Change 3: Implement constrained beam search (level-aware logits masking)

**Motivation**: The current `model.generate()` allows all 279 vocab tokens at every generation step. At the c1 position only 4 tokens are valid (0–3), yet the beam wastes capacity on the other 275. Constrained generation forces each generation step to only produce tokens valid for that level.

### Update `model/inference.py`

Add a `LevelConstrainedLogitsProcessor` class and use it in `predict_topk()`.

```python
from transformers import LogitsProcessor, LogitsProcessorList
import torch

class LevelConstrainedLogitsProcessor(LogitsProcessor):
    """
    在 beam search 每一步强制只生成当前层合法的 token。

    生成步骤：
      step 0 → c1: 只允许 token 0 ~ K_LEVELS[0]-1
      step 1 → c2: 只允许 token K_LEVELS[0] ~ K_LEVELS[0]+K_LEVELS[1]-1
      step 2 → c3: 只允许 token K_LEVELS[0]+K_LEVELS[1] ~ ... -1
    """
    def __init__(self, input_len, k_levels, level_offsets, vocab_size):
        self.input_len     = input_len      # 原始 input_ids 的长度
        self.k_levels      = k_levels
        self.level_offsets = level_offsets
        self.vocab_size    = vocab_size

    def __call__(self, input_ids, scores):
        # 当前已生成的新 token 数量
        gen_step = input_ids.shape[1] - self.input_len   # 0, 1, 2
        level    = gen_step % 3                           # 0=c1, 1=c2, 2=c3

        # 把所有 token 的 logit 设为 -inf，再把合法范围还原
        mask = torch.full_like(scores, float('-inf'))
        start = self.level_offsets[level]
        end   = start + self.k_levels[level]
        mask[:, start:end] = 0.0

        return scores + mask
```

Then update `predict_topk()` to use this processor:

```python
def predict_topk(model, history_seq, semantic_ids, sid_to_item,
                 sid_array, item_id_list, k=10, beam_width=50, device='cpu'):
    model.eval()
    input_tokens = seq_to_tokens(history_seq, semantic_ids, maxlen=50)
    input_ids    = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
    input_len    = len(input_tokens)

    # 约束处理器：每步只允许对应层的合法 token
    constrained_processor = LevelConstrainedLogitsProcessor(
        input_len     = input_len,
        k_levels      = K_LEVELS,
        level_offsets = LEVEL_OFFSETS,
        vocab_size    = PAD_TOKEN + 1,   # total vocab size
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens        = 3,
            num_beams             = beam_width,
            num_return_sequences  = min(beam_width, k * 5),
            early_stopping        = False,     # 不用 EOS 停止，生成满 3 个 token 即止
            pad_token_id          = PAD_TOKEN,
            logits_processor      = LogitsProcessorList([constrained_processor]),
        )

    # ... rest of the function unchanged (item lookup, dedup, etc.)
```

Also add the import at the top of `model/inference.py`:
```python
from model.tokenizer import seq_to_tokens, tokens_to_semantic_id, PAD_TOKEN, K_LEVELS, LEVEL_OFFSETS
from transformers import LogitsProcessor, LogitsProcessorList
```

Note: `LEVEL_OFFSETS` needs to be exported from `model/tokenizer.py` — it is already defined there, just make sure it's importable (it should be since it's a module-level variable).

---

## Change 4: Change early stopping from val_loss to Recall@10 on val set

**Motivation**: Minimizing language model loss ≠ maximizing recommendation accuracy. We should save the checkpoint that actually gives the best recommendation metrics.

### Update `train.py`

Add a fast val Recall evaluation function that runs beam search on a **random subset** of val users (e.g., 500 users) to keep it fast enough to run every few epochs:

```python
import random as _random
from model.inference import build_reverse_index, predict_topk

def evaluate_recall_subset(model, val_seqs, semantic_ids, device,
                           maxlen=50, k=10, n_users=500, beam_width=20):
    """
    在 val 集的随机子集上计算 Recall@k（beam_width 可以小一些以加快速度）。
    用于 early stopping，不是最终汇报数字。
    """
    sid_to_item, sid_array, item_id_list = build_reverse_index(semantic_ids)
    model.eval()

    users    = list(val_seqs.keys())
    sampled  = _random.sample(users, min(n_users, len(users)))
    hits     = 0

    with torch.no_grad():
        for user in sampled:
            full_seq = val_seqs[user]
            target   = full_seq[-1]
            history  = full_seq[:-1]

            recs = predict_topk(model, history, semantic_ids,
                                sid_to_item, sid_array, item_id_list,
                                k=k, beam_width=beam_width, device=device)
            if target in recs:
                hits += 1

    model.train()
    return hits / len(sampled)
```

Then in the training loop, change from:
```python
# Old: early stop on val_loss
val_loss = evaluate_val_loss(model, val_loader, device)
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(...)
```

To:
```python
# New: early stop on val Recall@10 (evaluated every val_every epochs)
val_recall = evaluate_recall_subset(model, val_seqs, semantic_ids, device,
                                    n_users=500, beam_width=20)
print(f'Epoch {epoch:3d}/{CONFIG["num_epochs"]}  '
      f'train_loss={avg_train_loss:.4f}  val_Recall@10={val_recall:.4f}  '
      f'lr={scheduler.get_last_lr()[0]:.2e}')

if val_recall > best_val_recall:
    best_val_recall = val_recall
    patience_count  = 0
    torch.save(model.state_dict(),
               os.path.join(base_dir, 'checkpoints/best_model.pt'))
    print(f'  ✓ 保存最优模型 (val_Recall@10={best_val_recall:.4f})')
else:
    patience_count += 1
    if patience_count >= CONFIG['patience']:
        print(f'\nEarly stopping')
        break
```

Also update the CONFIG dict to adjust `val_every` (beam search eval is slower):
```python
CONFIG = {
    ...
    'val_every':   5,     # 每 5 epoch 做一次 val Recall 评估（beam search 慢）
    'patience':    6,     # 连续 6 次（即 30 epoch）不提升则 early stop
    ...
}
```

Update `best_val_loss` variable to `best_val_recall = 0.0`.

Also need to load `val_seqs` and `semantic_ids` in the `train()` function (they're already loaded, just make sure they're accessible in the training loop).

---

## Change 5: Implement RQ-VAE for Semantic ID generation

**Motivation**: k-means is post-hoc clustering. RQ-VAE learns the codebook end-to-end to minimize reconstruction error + commitment loss, producing semantically richer Semantic IDs. This is TIGER's core contribution.

### Create `embedding/rqvae.py`

This is a new file implementing the full RQ-VAE pipeline. Follow the TIGER paper's architecture exactly.

```python
"""
RQ-VAE: Residual-Quantized Variational AutoEncoder for Semantic ID generation.
Follows TIGER (NeurIPS 2023) architecture.

Architecture (from paper):
  - Encoder: 768 → 512 → 256 → 128 → 32 (ReLU, except final layer)
  - Decoder: 32 → 128 → 256 → 512 → 768 (ReLU, except final layer)
  - 3 codebooks: sizes [4, 16, 256], each entry dim = 32
  - Loss: L_recon + sum_d( ||sg[r_d] - e_{c_d}||² + β*||r_d - sg[e_{c_d}]||² )
  - β = 0.25 (from paper)
  - Optimizer: Adagrad, lr=0.4, batch_size=1024
  - Codebook init: k-means on first batch

Usage:
    python embedding/rqvae.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────
INPUT_DIM    = 768           # nomic-embed-text output dimension
HIDDEN_DIMS  = [512, 256, 128]
LATENT_DIM   = 32            # dimension of quantized representation
K_LEVELS     = [4, 16, 256]  # codebook sizes per level
BETA         = 0.25          # commitment loss weight (from TIGER paper)
LR           = 0.4           # Adagrad learning rate (from TIGER paper)
BATCH_SIZE   = 1024
NUM_EPOCHS   = 500           # run until codebook usage ≥ 80% for all levels
MIN_USAGE    = 0.80          # minimum codebook usage to stop early


# ── Model ─────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, latent_dim=LATENT_DIM):
        super().__init__()
        dims = [input_dim] + hidden_dims + [latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:   # no activation on last layer
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=INPUT_DIM):
        super().__init__()
        dims = [latent_dim] + list(reversed(hidden_dims)) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class ResidualQuantizer(nn.Module):
    """
    3-level residual quantizer. Each level has an independent codebook.
    Uses straight-through estimator for gradients.
    """
    def __init__(self, latent_dim=LATENT_DIM, k_levels=K_LEVELS):
        super().__init__()
        self.k_levels = k_levels
        # One codebook per level: (K_i, latent_dim)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(k, latent_dim))
            for k in k_levels
        ])

    def quantize_level(self, r, codebook):
        """
        Find the nearest codebook entry for residual r.
        Returns: code index, quantized vector, updated residual.
        """
        # r: (B, D), codebook: (K, D)
        # distances: (B, K)
        dists = (r.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)
        codes = dists.argmin(dim=1)                    # (B,)
        quantized = codebook[codes]                    # (B, D)
        return codes, quantized

    def forward(self, z):
        """
        Residual quantization over 3 levels.
        Returns: code indices per level, total quantized vector, rq_loss.
        """
        residual = z
        all_codes = []
        quantized_total = torch.zeros_like(z)
        rq_loss = 0.0

        for level, codebook in enumerate(self.codebooks):
            codes, e = self.quantize_level(residual, codebook)
            all_codes.append(codes)

            # Commitment loss + codebook loss
            rq_loss += (residual.detach() - e).pow(2).mean()         # codebook update
            rq_loss += BETA * (residual - e.detach()).pow(2).mean()   # encoder push

            # Straight-through: copy gradients through quantization
            e_st      = residual + (e - residual).detach()
            quantized_total = quantized_total + e_st
            residual  = residual - e.detach()   # new residual for next level

        return all_codes, quantized_total, rq_loss

    def kmeans_init(self, z_samples):
        """Initialize codebooks with k-means on the first batch."""
        residual = z_samples.detach().cpu().numpy()
        with torch.no_grad():
            for level, (codebook, k) in enumerate(zip(self.codebooks, self.k_levels)):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(residual)
                codebook.data = torch.tensor(
                    km.cluster_centers_, dtype=torch.float32,
                    device=codebook.device)
                # Update residual for next level
                labels    = km.labels_
                centroids = km.cluster_centers_
                residual  = residual - centroids[labels]
        print("  Codebook initialized via k-means")


class RQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = Encoder()
        self.decoder   = Decoder()
        self.quantizer = ResidualQuantizer()

    def forward(self, x):
        z                          = self.encoder(x)
        all_codes, z_q, rq_loss   = self.quantizer(z)
        x_recon                    = self.decoder(z_q)
        recon_loss                 = (x - x_recon).pow(2).mean()
        total_loss                 = recon_loss + rq_loss
        return all_codes, total_loss, recon_loss, rq_loss


# ── Training ──────────────────────────────────────────────────────────────

def compute_codebook_usage(model, data_tensor, batch_size=1024):
    """Compute fraction of codebook entries used (usage rate per level)."""
    model.eval()
    all_codes_per_level = [[] for _ in range(len(K_LEVELS))]
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size]
            z     = model.encoder(batch)
            codes, _, _ = model.quantizer(z)
            for level, c in enumerate(codes):
                all_codes_per_level[level].extend(c.cpu().numpy().tolist())
    model.train()
    usages = []
    for level, (codes_list, k) in enumerate(zip(all_codes_per_level, K_LEVELS)):
        used  = len(set(codes_list))
        usage = used / k
        usages.append(usage)
    return usages


def train_rqvae():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')

    emb_dir  = os.path.dirname(os.path.abspath(__file__))
    raw      = np.load(os.path.join(emb_dir, 'item_embeddings_raw_nomic.npy'),
                       allow_pickle=True).item()

    item_ids   = sorted(raw.keys())
    emb_matrix = np.stack([raw[i] for i in item_ids]).astype(np.float32)
    print(f"加载 embedding: {emb_matrix.shape}")

    data_tensor = torch.tensor(emb_matrix, dtype=torch.float32).to(device)
    N           = len(data_tensor)

    model     = RQVAE().to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)

    # Codebook init: k-means on first full pass
    print("初始化码本（k-means）...")
    model.encoder.eval()
    with torch.no_grad():
        z_all = model.encoder(data_tensor)
    model.quantizer.kmeans_init(z_all)
    model.train()

    print(f'\n开始训练 RQ-VAE，最多 {NUM_EPOCHS} epochs\n')
    best_usage = [0.0] * len(K_LEVELS)

    for epoch in range(1, NUM_EPOCHS + 1):
        perm       = torch.randperm(N)
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, N, BATCH_SIZE):
            idx   = perm[i:i+BATCH_SIZE]
            batch = data_tensor[idx]

            _, loss, recon, rq = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches

        if epoch % 20 == 0 or epoch == 1:
            usages = compute_codebook_usage(model, data_tensor)
            usage_str = '  '.join([f'L{i}={u:.1%}' for i, u in enumerate(usages)])
            print(f'Epoch {epoch:4d}  loss={avg_loss:.4f}  codebook usage: {usage_str}')

            # Early stop when all levels reach MIN_USAGE
            if all(u >= MIN_USAGE for u in usages):
                print(f'\n所有层码本使用率 ≥ {MIN_USAGE:.0%}，提前结束训练')
                break

    # ── Extract Semantic IDs ───────────────────────────────────────────────
    print('\n提取 Semantic IDs...')
    model.eval()
    all_codes_levels = [[] for _ in range(len(K_LEVELS))]
    with torch.no_grad():
        for i in range(0, N, BATCH_SIZE):
            batch = data_tensor[i:i+BATCH_SIZE]
            z     = model.encoder(batch)
            codes, _, _ = model.quantizer(z)
            for level, c in enumerate(codes):
                all_codes_levels[level].extend(c.cpu().numpy().tolist())

    # Build semantic_ids dict: item_id -> (c0, c1, c2)
    semantic_ids = {}
    for idx, item_id in enumerate(item_ids):
        c0 = all_codes_levels[0][idx]
        c1 = all_codes_levels[1][idx]
        c2 = all_codes_levels[2][idx]
        semantic_ids[item_id] = (c0, c1, c2)

    # ── Collision resolution ───────────────────────────────────────────────
    # If multiple items share the same (c0, c1, c2), reassign c2 to make unique.
    # Strategy: within each collision group, keep the first item as-is,
    # and reassign others by incrementing c2 (mod K3), avoiding existing codes.
    from collections import defaultdict
    sid_to_items = defaultdict(list)
    for item_id, sid in semantic_ids.items():
        sid_to_items[sid].append(item_id)

    collisions = {sid: items for sid, items in sid_to_items.items() if len(items) > 1}
    print(f'冲突 Semantic ID 数: {len(collisions)}（共涉及 {sum(len(v) for v in collisions.values())} items）')

    if collisions:
        used_sids = set(semantic_ids.values())
        for sid, items in collisions.items():
            c0, c1, c2_base = sid
            # Keep first item as-is; reassign subsequent items
            for item_id in items[1:]:
                new_c2 = (c2_base + 1) % K_LEVELS[2]
                while (c0, c1, new_c2) in used_sids:
                    new_c2 = (new_c2 + 1) % K_LEVELS[2]
                semantic_ids[item_id] = (c0, c1, new_c2)
                used_sids.add((c0, c1, new_c2))

    # Final collision check
    all_sids     = [tuple(v) for v in semantic_ids.values()]
    unique_count = len(set(all_sids))
    print(f'解决后唯一 Semantic ID 数: {unique_count} / {len(all_sids)}')

    output_path = os.path.join(emb_dir, 'semantic_ids_rqvae.npy')
    np.save(output_path, semantic_ids)
    print(f'已保存至 {output_path}')

    # Distribution summary
    c0_vals = [v[0] for v in semantic_ids.values()]
    print(f'c0 分布: {sorted(Counter(c0_vals).items())}')


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    train_rqvae()
```

---

## Summary of files to create/modify

| File | Action |
|------|--------|
| `model/tokenizer.py` | Modify: K_LEVELS [4,64,256]→[4,16,256]; `seq_to_tokens()` adds EOS after each item |
| `model/generative_rec.py` | Modify: update n_positions comment |
| `model/inference.py` | Modify: add `LevelConstrainedLogitsProcessor`; update `predict_topk()` |
| `train.py` | Modify: add `evaluate_recall_subset()`; change early stopping to val Recall@10 |
| `evaluate.py` | Modify: update semantic_ids path to `semantic_ids_nomic.npy` |
| `embedding/build_semantic_ids_nomic.py` | **Create**: k-means with K_LEVELS=[4,16,256] on nomic embeddings |
| `embedding/rqvae.py` | **Create**: full RQ-VAE implementation |

## Important Notes

1. **Run order**: After all code changes, run in this order:
   ```bash
   python embedding/build_semantic_ids_nomic.py   # Build k-means nomic semantic IDs
   python train.py                                  # Train with nomic IDs + all improvements
   python evaluate.py                               # Final evaluation
   # Then separately for RQ-VAE:
   python embedding/rqvae.py                        # Train RQ-VAE → semantic_ids_rqvae.npy
   # Update train.py/evaluate.py to use semantic_ids_rqvae.npy, then retrain
   ```

2. **Do not modify** `embedding/build_semantic_ids.py` or `embedding/item_embeddings_raw_nomic.npy`.

3. **The constrained beam search** (`LevelConstrainedLogitsProcessor`) must correctly compute `gen_step = input_ids.shape[1] - self.input_len`. Since HuggingFace's beam search duplicates `input_ids` across beams, `input_ids.shape[1]` increases by 1 at each step while the batch dimension expands. The processor receives one row per beam, so `input_ids.shape[1]` correctly reflects the total sequence length including generated tokens.

4. After changing `tokenizer.py`, the old checkpoints (`checkpoints/best_model.pt`) are **incompatible** (vocab size changed from 327 to 279). Always retrain from scratch after these changes.
