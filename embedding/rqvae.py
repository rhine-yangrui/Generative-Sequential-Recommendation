"""
RQ-VAE Training: trains encoder/decoder/quantizer on nomic embeddings.

Architecture:
  - Encoder: 768 -> 512 -> 256 -> 128 -> 32
  - Decoder: 32 -> 128 -> 256 -> 512 -> 768
  - 3 codebooks: K_LEVELS=[256, 256, 256], code dim=32
  - Loss: reconstruction loss + commitment loss (beta=0.25)

Training recipe:
  1. Warm up encoder+decoder (no quantization, 50 epochs) so K-means init
     works on a meaningful latent space.
  2. K-means init codebooks on warmed-up encoder output.
  3. Full RQ-VAE training with AdamW + cosine LR + grad clip.
  4. Sinkhorn balanced assignment on L1 and L2 during training (sk_epsilons).
     This is the key mechanism preventing codebook collapse.
  5. Dead-code reset every RESET_EVERY epochs as a safety net.

Sinkhorn is DISABLED during inference (use_sk=False in get_indices()).

Best checkpoint selected by highest unique_rate (= 1 - collision_rate)
and saved to checkpoints/rqvae_best.pt.

Usage:
    python embedding/rqvae.py
    python embedding/generate_rqvae_ids.py
"""

import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


INPUT_DIM    = 768
HIDDEN_DIMS  = [512, 256, 128]
LATENT_DIM   = 32
K_LEVELS     = [256, 256, 256]
BETA         = 0.25

# Sinkhorn balanced assignment: only enabled on L2 (对齐 TIGER 默认配置)
# Prevents codebook collapse by forcing roughly equal code usage per batch.
SK_EPSILONS = [0.0, 0.0, 0.003]
SK_ITERS    = 50

LR            = 1e-3
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 1024
WARMUP_EPOCHS = 50      # encoder-only warm-up before K-means init
NUM_EPOCHS    = 3000    # post-warmup full RQ-VAE training（对齐 TIGER 3000）
EVAL_EVERY    = 50
RESET_EVERY   = 100     # dead-code reset interval
MAX_GRAD_NORM = 1.0

# 输出文件 tag，避免与现有 rqvae_best.pt / semantic_ids_rqvae.npy 冲突
# E9 train/evaluate 仍在用旧文件，新文件落到独立路径
OUTPUT_TAG    = '3kep'


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, latent_dim=LATENT_DIM):
        super().__init__()
        dims = [input_dim] + hidden_dims + [latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
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
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


@torch.no_grad()
def sinkhorn(distances, epsilon, n_iters):
    """
    Numerically stable Sinkhorn-Knopp in the log domain (float32 / MPS-safe).

    Standard Sinkhorn uses Q = exp(-d/eps) and then row/col normalize. With
    small epsilon (e.g. 0.003) and float32, exp(-d/eps) underflows to 0 for
    all but the closest code, defeating the balancing.

    Log-domain Sinkhorn uses logsumexp throughout, so precision is preserved
    even when "linear-domain" weights would be ~1e-300.

    Returns Q in linear domain for diagnostic purposes; for picking codes you
    can equivalently take argmax of log_Q.

    Args:
        distances: (B, K) distance matrix, already centered to [-1, 1]
        epsilon:   temperature; smaller = harder assignment
        n_iters:   number of Sinkhorn iterations

    Returns:
        Q: (B, K) soft assignment matrix
    """
    log_Q = -distances / epsilon                       # (B, K) logits

    B, K = log_Q.shape
    log_B = torch.log(torch.tensor(float(B), device=log_Q.device))
    log_K = torch.log(torch.tensor(float(K), device=log_Q.device))

    # Initial normalization to log-probabilities (sum to 1 in linear space)
    log_Q = log_Q - log_Q.flatten().logsumexp(0)

    for _ in range(n_iters):
        # Row normalize: target row sum (linear) = 1/B  →  logsumexp = -log B
        log_Q = log_Q - log_Q.logsumexp(dim=1, keepdim=True) - log_B
        # Column normalize: target col sum (linear) = 1/K  →  logsumexp = -log K
        log_Q = log_Q - log_Q.logsumexp(dim=0, keepdim=True) - log_K

    # Q *= B  (so columns sum to 1 ≡ hard-assignment interpretation)
    log_Q = log_Q + log_B
    return torch.exp(log_Q)


class ResidualQuantizer(nn.Module):
    """
    Three-level residual quantizer with straight-through gradients.
    Supports Sinkhorn balanced assignment per level (enabled during training,
    disabled during inference).
    """

    def __init__(self, latent_dim=LATENT_DIM, k_levels=K_LEVELS,
                 sk_epsilons=SK_EPSILONS, sk_iters=SK_ITERS):
        super().__init__()
        self.k_levels    = k_levels
        self.sk_epsilons = sk_epsilons
        self.sk_iters    = sk_iters
        self.codebooks   = nn.ParameterList([
            nn.Parameter(torch.randn(k, latent_dim))
            for k in k_levels
        ])

    def quantize_level(self, residual, codebook, sk_epsilon, use_sk):
        """
        Quantize residual against one codebook.

        During training (use_sk=True) and when sk_epsilon > 0, Sinkhorn
        replaces argmin with a balanced soft assignment.
        During inference (use_sk=False), plain argmin is used.
        """
        # L2 distances: (B, K)
        dists = (residual.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)

        if use_sk and sk_epsilon > 0:
            # Center distances to [-1, 1] for numerical stability
            d_max = dists.max()
            d_min = dists.min()
            mid   = (d_max + d_min) / 2
            amp   = (d_max - mid).clamp(min=1e-5)
            d_centered = (dists - mid) / amp  # stay in float32 for MPS

            Q = sinkhorn(d_centered, sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                # Sinkhorn diverged — fall back to argmin
                codes = dists.argmin(dim=-1)
            else:
                codes = Q.argmax(dim=-1).long()
        else:
            codes = dists.argmin(dim=-1)

        quantized = codebook[codes]
        return codes, quantized

    def forward(self, z, use_sk=True):
        residual = z
        all_codes = []
        quantized_total = torch.zeros_like(z)
        rq_loss = 0.0

        for codebook, sk_epsilon in zip(self.codebooks, self.sk_epsilons):
            codes, e = self.quantize_level(residual, codebook, sk_epsilon, use_sk)
            all_codes.append(codes)

            # Codebook loss: trains codebook to track encoder
            rq_loss = rq_loss + (residual.detach() - e).pow(2).mean()
            # Commitment loss: trains encoder to stay near codebook
            rq_loss = rq_loss + BETA * (residual - e.detach()).pow(2).mean()

            # Straight-through estimator
            e_st = residual + (e - residual).detach()
            quantized_total = quantized_total + e_st
            residual = residual - e.detach()

        return all_codes, quantized_total, rq_loss

    def kmeans_init(self, z_samples):
        """Initialize codebooks with k-means on residuals at each level."""
        residual = z_samples.detach().cpu().numpy()
        with torch.no_grad():
            for codebook, k in zip(self.codebooks, self.k_levels):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(residual)
                codebook.data = torch.tensor(
                    km.cluster_centers_, dtype=torch.float32, device=codebook.device
                )
                residual = residual - km.cluster_centers_[km.labels_]
        print('  Codebook initialized via k-means on warmed-up encoder output')


class RQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = Encoder()
        self.decoder   = Decoder()
        self.quantizer = ResidualQuantizer()

    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        all_codes, z_q, rq_loss = self.quantizer(z, use_sk=use_sk)
        x_recon = self.decoder(z_q)
        recon_loss = (x - x_recon).pow(2).mean()
        total_loss = recon_loss + rq_loss
        return all_codes, total_loss, recon_loss, rq_loss

    @torch.no_grad()
    def get_indices(self, x, use_sk=False):
        """Get discrete codes for items. use_sk=False for inference."""
        z = self.encoder(x)
        codes, _, _ = self.quantizer(z, use_sk=use_sk)
        return codes


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def compute_metrics(model, data_tensor):
    """
    Compute per-level codebook usage and unique_rate (= 1 - collision_rate).
    Uses use_sk=False (argmin) for a fair picture of actual collision state.
    """
    model.eval()
    all_codes_per_level = [[] for _ in range(len(K_LEVELS))]

    with torch.no_grad():
        for i in range(0, len(data_tensor), BATCH_SIZE):
            batch = data_tensor[i:i + BATCH_SIZE]
            codes = model.get_indices(batch, use_sk=False)
            for level, code_tensor in enumerate(codes):
                all_codes_per_level[level].extend(code_tensor.cpu().tolist())

    model.train()

    n_items    = len(all_codes_per_level[0])
    usages     = [len(set(c)) / k for c, k in zip(all_codes_per_level, K_LEVELS)]
    unique_rate = len(set(zip(*all_codes_per_level))) / n_items

    return usages, unique_rate


def warmup_encoder(model, data_tensor, n_items, device):
    """
    Pre-train encoder+decoder with reconstruction loss only (no quantization).
    Gives the encoder a meaningful latent space before K-means codebook init,
    preventing the K-means from clustering a random/degenerate space.
    """
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    print(f'Encoder 预热 {WARMUP_EPOCHS} epochs（无量化）...')
    for epoch in range(1, WARMUP_EPOCHS + 1):
        perm       = torch.randperm(n_items, device=device)
        total_loss = 0.0
        n_batches  = 0
        for i in range(0, n_items, BATCH_SIZE):
            batch = data_tensor[perm[i:i + BATCH_SIZE]]
            z     = model.encoder(batch)
            recon = model.decoder(z)
            loss  = (batch - recon).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        if epoch % 10 == 0:
            print(f'  warmup epoch {epoch:3d}: recon={total_loss / n_batches:.4f}')
    print('  预热完成\n')


def reset_dead_codes(model, data_tensor):
    """
    Reset unused codebook entries (count=0) to a random encoder output + noise.
    Acts as a safety net in addition to Sinkhorn.
    """
    model.eval()
    n_items    = len(data_tensor)
    code_counts = [Counter() for _ in range(len(K_LEVELS))]

    with torch.no_grad():
        for i in range(0, n_items, BATCH_SIZE):
            batch = data_tensor[i:i + BATCH_SIZE]
            codes = model.get_indices(batch, use_sk=False)
            for level, c in enumerate(codes):
                for val in c.cpu().tolist():
                    code_counts[level][val] += 1

    sample_idx = torch.randperm(n_items, device=data_tensor.device)[:512]
    with torch.no_grad():
        z_sample = model.encoder(data_tensor[sample_idx])

    n_reset = 0
    with torch.no_grad():
        for level, (codebook, k) in enumerate(zip(model.quantizer.codebooks, K_LEVELS)):
            dead = [i for i in range(k) if code_counts[level].get(i, 0) == 0]
            for j, code_idx in enumerate(dead):
                src = z_sample[j % len(z_sample)].detach()
                codebook.data[code_idx] = src + 0.01 * torch.randn_like(src)
            n_reset += len(dead)

    model.train()
    return n_reset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rqvae():
    device = select_device()
    print(f'使用设备: {device}')

    emb_dir = os.path.dirname(os.path.abspath(__file__))
    raw = np.load(
        os.path.join(emb_dir, 'item_embeddings_raw_nomic.npy'),
        allow_pickle=True,
    ).item()

    item_ids   = sorted(raw.keys())
    emb_matrix = np.stack([raw[i] for i in item_ids]).astype(np.float32)
    emb_matrix /= np.clip(np.linalg.norm(emb_matrix, axis=1, keepdims=True), 1e-12, None)
    print(f'加载 embedding: {emb_matrix.shape}\n')

    data_tensor = torch.tensor(emb_matrix, dtype=torch.float32, device=device)
    n_items     = len(data_tensor)

    torch.manual_seed(42)
    np.random.seed(42)
    model = RQVAE().to(device)

    # ── Phase 1: encoder warm-up ──────────────────────────────────────────
    warmup_encoder(model, data_tensor, n_items, device)

    # ── Phase 2: K-means init on warmed-up encoder output ────────────────
    print('K-means 初始化码本（在预热后的 encoder 输出上）...')
    with torch.no_grad():
        z_all = model.encoder(data_tensor)
    model.quantizer.kmeans_init(z_all)

    usages, unique_rate = compute_metrics(model, data_tensor)
    usage_str = '  '.join([f'L{i}={u:.1%}' for i, u in enumerate(usages)])
    print(f'  初始 unique={unique_rate:.1%}  usage: {usage_str}\n')

    # ── Phase 3: full RQ-VAE training with Sinkhorn ───────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )

    ckpt_dir  = os.path.join(os.path.dirname(emb_dir), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path  = os.path.join(ckpt_dir, f'rqvae_{OUTPUT_TAG}_best.pt')
    final_path = os.path.join(ckpt_dir, f'rqvae_{OUTPUT_TAG}_final.pt')

    best_recon_loss = float('inf')
    print(f'开始 RQ-VAE 训练，共 {NUM_EPOCHS} epochs')
    print(f'Sinkhorn 配置: sk_epsilons={SK_EPSILONS}  (0.0 = 禁用)\n')

    for epoch in range(1, NUM_EPOCHS + 1):
        perm = torch.randperm(n_items, device=device)
        total_loss = recon_sum = rq_sum = 0.0
        n_batches  = 0

        for i in range(0, n_items, BATCH_SIZE):
            idx   = perm[i:i + BATCH_SIZE]
            batch = data_tensor[idx]

            _, loss, recon_loss, rq_loss = model(batch, use_sk=True)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()
            recon_sum  += recon_loss.item()
            rq_sum     += rq_loss.item()
            n_batches  += 1

        scheduler.step()

        # Dead-code reset (safety net; Sinkhorn should handle most cases)
        if epoch % RESET_EVERY == 0:
            n_reset = reset_dead_codes(model, data_tensor)
            if n_reset > 0:
                print(f'  [reset] epoch {epoch}: 重置 {n_reset} 个死码')

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            usages, unique_rate = compute_metrics(model, data_tensor)
            usage_str = '  '.join([f'L{i}={u:.1%}' for i, u in enumerate(usages)])
            print(
                f'Epoch {epoch:4d}  '
                f'loss={total_loss/n_batches:.4f}  '
                f'recon={recon_sum/n_batches:.4f}  '
                f'rq={rq_sum/n_batches:.4f}  '
                f'unique={unique_rate:.1%}  '
                f'usage: {usage_str}'
            )

            # 与 unique 不同：3000 epoch 下 unique 早期就饱和（~99%），
            # 后期靠 recon 持续改进编码器/码本质量。按 recon 选 best。
            cur_recon = recon_sum / n_batches
            if cur_recon < best_recon_loss:
                best_recon_loss = cur_recon
                torch.save({
                    'epoch':       epoch,
                    'model_state': model.state_dict(),
                    'recon_loss':  cur_recon,
                    'unique_rate': unique_rate,
                    'usages':      usages,
                }, ckpt_path)
                print(f'  ✓ 保存最优 checkpoint (recon={cur_recon:.4f}, unique={unique_rate:.1%})')

    # 始终额外保存最后一个 epoch，便于对比 best vs final
    torch.save({
        'epoch':       NUM_EPOCHS,
        'model_state': model.state_dict(),
        'recon_loss':  recon_sum / n_batches,
        'unique_rate': unique_rate,
        'usages':      usages,
    }, final_path)

    print(f'\n训练完成，最优 recon_loss={best_recon_loss:.4f}')
    print(f'Best checkpoint:  {ckpt_path}')
    print(f'Final checkpoint: {final_path}')
    print(f'运行 python embedding/generate_rqvae_ids.py 生成 semantic_ids_rqvae_{OUTPUT_TAG}.npy')


if __name__ == '__main__':
    train_rqvae()
