"""
RQ-VAE Training — TIGER-aligned reference reproduction.

Mirrors `../TIGER/rqvae/{models,trainer,main}.py`:
  - MLPLayers with xavier init, dropout/bn options, no activation on last layer
  - VectorQuantizer + ResidualVectorQuantizer with lazy k-means init on the
    residual at first forward (no encoder warm-up phase)
  - 5-hidden encoder [768, 512, 256, 128, 64, 32]
  - AdamW + linear LR schedule with warmup (per-step .step())
  - Quant loss averaged across levels (not summed)
  - No L2 normalization on input embeddings (this kills the recon signal)
  - No dead-code reset (Sinkhorn is the only balancing mechanism, like TIGER)

Best checkpoint (highest unique-rate = lowest collision) saved to
`checkpoints/rqvae_best.pt`, matching TIGER's generate_code.py default.

Usage:
    python embedding/rqvae.py
    python embedding/generate_rqvae_ids.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup


# ── Architecture (TIGER defaults) ────────────────────────────────────────
HIDDEN_LAYERS  = [512, 256, 128, 64]   # encoder hidden sizes (5 hidden total: + e_dim)
LATENT_DIM     = 32                    # encoder bottleneck = codebook embedding dim
# Sizes of the 3 learned residual codebooks. The downstream tokenizer adds a
# 4th collision-resolution code on top of these (see model/tokenizer.py).
CODEBOOK_SIZES = [256, 256, 256]
BETA           = 0.25
QUANT_LOSS_WEIGHT = 1.0
DROPOUT        = 0.0
USE_BN         = False
LOSS_TYPE      = 'mse'

KMEANS_INIT    = True
KMEANS_ITERS   = 100

# Sinkhorn balanced assignment, only enabled on the last codebook (TIGER default)
SK_EPSILONS    = [0.0, 0.0, 0.003]
SK_ITERS       = 50

# ── Optimization ─────────────────────────────────────────────────────────
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
BATCH_SIZE     = 1024
WARMUP_EPOCHS  = 50         # linear LR warmup (TIGER 'linear' scheduler)
NUM_EPOCHS     = 3000       # full TIGER schedule; shorter runs leave R@10 on the table
EVAL_EVERY     = 50         # TIGER default
MAX_GRAD_NORM  = 1.0

# ── Input / output ───────────────────────────────────────────────────────
EMBEDDING_FILE = 'item_embeddings_raw.npy'


# ─────────────────────────────────────────────────────────────────────────
# Building blocks (mirror TIGER's models/layers.py + vq.py + rq.py)
# ─────────────────────────────────────────────────────────────────────────

class MLPLayers(nn.Module):
    """Dropout → Linear (→ BN) (→ ReLU) per hidden, no activation on last layer."""

    def __init__(self, layers, dropout=0.0, bn=False):
        super().__init__()
        modules = []
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            is_last = (idx == len(layers) - 2)
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(in_size, out_size))
            if bn and not is_last:
                modules.append(nn.BatchNorm1d(out_size))
            if not is_last:
                modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def sinkhorn_log(distances, epsilon, n_iters):
    """
    Numerically stable log-domain Sinkhorn-Knopp (MPS-safe; same semantics
    as TIGER's linear-domain version, but doesn't underflow under fp32).
    """
    log_Q = -distances / epsilon
    B, K  = log_Q.shape
    log_B = torch.log(torch.tensor(float(B), device=log_Q.device))
    log_K = torch.log(torch.tensor(float(K), device=log_Q.device))

    log_Q = log_Q - log_Q.flatten().logsumexp(0)
    for _ in range(n_iters):
        log_Q = log_Q - log_Q.logsumexp(dim=1, keepdim=True) - log_B
        log_Q = log_Q - log_Q.logsumexp(dim=0, keepdim=True) - log_K
    log_Q = log_Q + log_B
    return torch.exp(log_Q)


class VectorQuantizer(nn.Module):
    """Single-codebook VQ — mirrors `../TIGER/rqvae/models/vq.py`."""

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=True,
                 kmeans_iters=100, sk_epsilon=0.0, sk_iters=50):
        super().__init__()
        self.n_e          = n_e
        self.e_dim        = e_dim
        self.beta         = beta
        self.kmeans_init  = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon   = sk_epsilon
        self.sk_iters     = sk_iters
        self.embedding    = nn.Embedding(n_e, e_dim)

        if kmeans_init:
            self.initted = False
            self.embedding.weight.data.zero_()
        else:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    @torch.no_grad()
    def init_emb(self, latent):
        x_np = latent.detach().cpu().numpy()
        km = KMeans(
            n_clusters=self.n_e,
            n_init=10,
            max_iter=self.kmeans_iters,
            random_state=42,
        )
        km.fit(x_np)
        centers = torch.from_numpy(km.cluster_centers_).to(
            device=latent.device, dtype=latent.dtype
        )
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def _center_distance(d):
        d_max = d.max()
        d_min = d.min()
        mid   = (d_max + d_min) / 2
        amp   = (d_max - mid).clamp(min=1e-5)
        return (d - mid) / amp

    def forward(self, x, use_sk=True):
        latent = x.reshape(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # |a-b|^2 = |a|^2 + |b|^2 - 2 a·b
        d = (latent.pow(2).sum(dim=1, keepdim=True)
             + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
             - 2 * latent @ self.embedding.weight.t())

        if use_sk and self.sk_epsilon > 0:
            d_centered = self._center_distance(d)
            Q = sinkhorn_log(d_centered, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                indices = d.argmin(dim=-1)
            else:
                indices = Q.argmax(dim=-1).long()
        else:
            indices = d.argmin(dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # codebook_loss trains the codebook; commitment_loss trains the encoder
        codebook_loss   = F.mse_loss(x_q, x.detach())
        commitment_loss = F.mse_loss(x_q.detach(), x)
        loss = codebook_loss + self.beta * commitment_loss

        # straight-through estimator
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices


class ResidualVectorQuantizer(nn.Module):
    """Mirrors `../TIGER/rqvae/models/rq.py`."""

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta=0.25,
                 kmeans_init=True, kmeans_iters=100, sk_iters=50):
        super().__init__()
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(
                n_e, e_dim, beta=beta,
                kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                sk_epsilon=sk_eps, sk_iters=sk_iters,
            )
            for n_e, sk_eps in zip(n_e_list, sk_epsilons)
        ])

    def forward(self, x, use_sk=True):
        x_q         = 0
        residual    = x
        all_losses  = []
        all_indices = []
        for quantizer in self.vq_layers:
            x_res, loss, idx = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q      = x_q + x_res
            all_losses.append(loss)
            all_indices.append(idx)
        mean_loss   = torch.stack(all_losses).mean()       # ← average, not sum
        all_indices = torch.stack(all_indices, dim=-1)     # (B, n_levels)
        return x_q, mean_loss, all_indices


class RQVAE(nn.Module):
    """Top-level model — mirrors `../TIGER/rqvae/models/rqvae.py`."""

    def __init__(self, in_dim):
        super().__init__()
        encoder_dims = [in_dim] + HIDDEN_LAYERS + [LATENT_DIM]
        decoder_dims = encoder_dims[::-1]
        self.encoder = MLPLayers(encoder_dims, dropout=DROPOUT, bn=USE_BN)
        self.decoder = MLPLayers(decoder_dims, dropout=DROPOUT, bn=USE_BN)
        self.rq = ResidualVectorQuantizer(
            n_e_list=CODEBOOK_SIZES,
            e_dim=LATENT_DIM,
            sk_epsilons=SK_EPSILONS,
            beta=BETA,
            kmeans_init=KMEANS_INIT,
            kmeans_iters=KMEANS_ITERS,
            sk_iters=SK_ITERS,
        )

    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        z_q, rq_loss, indices = self.rq(z, use_sk=use_sk)
        out = self.decoder(z_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, x, use_sk=False):
        """Inference: pure argmin (use_sk=False), no Sinkhorn."""
        z = self.encoder(x)
        _, _, indices = self.rq(z, use_sk=use_sk)
        return indices

    def compute_loss(self, out, rq_loss, xs):
        if LOSS_TYPE == 'mse':
            recon = F.mse_loss(out, xs)
        elif LOSS_TYPE == 'l1':
            recon = F.l1_loss(out, xs)
        else:
            raise ValueError(f'unknown loss type {LOSS_TYPE}')
        return recon + QUANT_LOSS_WEIGHT * rq_loss, recon


# ─────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────

def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def compute_metrics(model, data_tensor, device):
    """Per-level codebook usage and unique-rate (= 1 - collision_rate)."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(data_tensor), BATCH_SIZE):
            batch = data_tensor[i:i + BATCH_SIZE].to(device)
            chunks.append(model.get_indices(batch, use_sk=False).cpu())
    all_codes = torch.cat(chunks, dim=0).numpy()    # (N, n_levels)
    model.train()

    n_items     = len(all_codes)
    usages      = [
        len(set(all_codes[:, lvl].tolist())) / CODEBOOK_SIZES[lvl]
        for lvl in range(len(CODEBOOK_SIZES))
    ]
    unique_rate = len({tuple(c) for c in all_codes}) / n_items
    return usages, unique_rate


# ─────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────

def train_rqvae():
    device = select_device()
    print(f'使用设备: {device}')

    emb_dir = os.path.dirname(os.path.abspath(__file__))
    raw = np.load(
        os.path.join(emb_dir, EMBEDDING_FILE),
        allow_pickle=True,
    ).item()
    print(f'Embedding 源: {EMBEDDING_FILE}')

    item_ids   = sorted(raw.keys())
    emb_matrix = np.stack([raw[i] for i in item_ids]).astype(np.float32)
    # 不做 L2 normalize：对齐 TIGER（保留幅度信息，否则 recon floor 极低）
    print(f'加载 embedding: {emb_matrix.shape}\n')

    in_dim = emb_matrix.shape[1]
    data_tensor = torch.from_numpy(emb_matrix)

    train_loader = DataLoader(
        TensorDataset(data_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    model = RQVAE(in_dim=in_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f'参数量: {n_params/1e6:.2f}M\n')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )

    # Linear warmup + linear decay (TIGER 'linear' scheduler), per-step
    steps_per_epoch = len(train_loader)
    total_steps     = NUM_EPOCHS * steps_per_epoch
    warmup_steps    = WARMUP_EPOCHS * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    ckpt_dir = os.path.join(os.path.dirname(emb_dir), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path  = os.path.join(ckpt_dir, 'rqvae_best.pt')
    final_ckpt_path = os.path.join(ckpt_dir, 'rqvae_final.pt')
    best_coll = float('inf')
    last_unique = None

    print(f'开始 RQ-VAE 训练，共 {NUM_EPOCHS} epochs '
          f'(warmup {WARMUP_EPOCHS} ep, total {total_steps} steps)')
    print(f'Sinkhorn 配置: sk_epsilons={SK_EPSILONS}\n')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_recon = 0.0
        epoch_quant = 0.0
        n_batches   = 0

        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            out, rq_loss, _ = model(batch, use_sk=True)
            loss, recon = model.compute_loss(out, rq_loss, batch)
            if torch.isnan(loss):
                raise RuntimeError(f'loss=nan at epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            epoch_recon += recon.item()
            epoch_quant += rq_loss.item()
            n_batches   += 1

        avg_loss  = epoch_loss  / n_batches
        avg_recon = epoch_recon / n_batches
        avg_quant = epoch_quant / n_batches

        is_eval = (epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS)
        if is_eval:
            usages, unique = compute_metrics(model, data_tensor, device)
            last_unique = unique
            usage_str = '  '.join(
                f'L{i}={u*100:.1f}%' for i, u in enumerate(usages)
            )
            current_lr = scheduler.get_last_lr()[0]
            print(
                f'Epoch {epoch:4d}  lr={current_lr:.2e}  '
                f'loss={avg_loss:.4f}  recon={avg_recon:.4f}  rq={avg_quant:.4f}  '
                f'unique={unique*100:.1f}%  usage: {usage_str}'
            )

            collision = 1.0 - unique
            if collision < best_coll:
                best_coll = collision
                torch.save(
                    {'state_dict': model.state_dict(),
                     'epoch': epoch,
                     'unique_rate': unique,
                     'in_dim': in_dim},
                    best_ckpt_path,
                )
                print(f'  ✓ best ckpt (collision={best_coll:.4f})')
        else:
            print(
                f'Epoch {epoch:4d}  loss={avg_loss:.4f}  '
                f'recon={avg_recon:.4f}  rq={avg_quant:.4f}'
            )

    # Always save the final-epoch state alongside best-by-collision so we can
    # A/B compare downstream. The "best" save criterion (1 - unique_rate) often
    # locks in the lazy k-means init at epoch 1; the final ckpt lets us check
    # whether subsequent training actually helps SID quality.
    torch.save(
        {'state_dict': model.state_dict(),
         'epoch': NUM_EPOCHS,
         'unique_rate': last_unique,
         'in_dim': in_dim},
        final_ckpt_path,
    )

    print(f'\n训练完成  best_collision={best_coll:.4f}')
    print(f'Best ckpt:  {best_ckpt_path}')
    print(f'Final ckpt: {final_ckpt_path}  '
          f'(epoch={NUM_EPOCHS}, unique={last_unique*100:.1f}%)')
    print('\n运行 python embedding/generate_rqvae_ids.py 生成 semantic_ids_rqvae.npy')
    print('   或 python embedding/generate_rqvae_ids.py --ckpt rqvae_final.pt '
          '--out semantic_ids_rqvae_final.npy')


if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    train_rqvae()
