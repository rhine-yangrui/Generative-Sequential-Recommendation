"""
RQ-VAE Training: trains encoder/decoder/quantizer on nomic embeddings.

Architecture:
  - Encoder: 768 -> 512 -> 256 -> 128 -> 32
  - Decoder: 32 -> 128 -> 256 -> 512 -> 768
  - 3 codebooks: K_LEVELS=[4, 16, 256], code dim=32
  - Loss: reconstruction loss + commitment loss (beta=0.25)

Best checkpoint is selected by highest unique_rate (= 1 - collision_rate)
and saved to checkpoints/rqvae_best.pt.

Usage:
    python embedding/rqvae.py
    python embedding/generate_rqvae_ids.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


INPUT_DIM   = 768
HIDDEN_DIMS = [512, 256, 128]
LATENT_DIM  = 32
K_LEVELS    = [4, 16, 256]
BETA        = 0.25

LR            = 1e-3
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 1024
NUM_EPOCHS    = 300
EVAL_EVERY    = 10
MAX_GRAD_NORM = 1.0


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


class ResidualQuantizer(nn.Module):
    """Three-level residual quantizer with straight-through gradients."""

    def __init__(self, latent_dim=LATENT_DIM, k_levels=K_LEVELS):
        super().__init__()
        self.k_levels = k_levels
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(k, latent_dim))
            for k in k_levels
        ])

    def quantize_level(self, residual, codebook):
        dists = (residual.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)
        codes = dists.argmin(dim=1)
        quantized = codebook[codes]
        return codes, quantized

    def forward(self, z):
        residual = z
        all_codes = []
        quantized_total = torch.zeros_like(z)
        rq_loss = 0.0

        for codebook in self.codebooks:
            codes, e = self.quantize_level(residual, codebook)
            all_codes.append(codes)

            rq_loss = rq_loss + (residual.detach() - e).pow(2).mean()
            rq_loss = rq_loss + BETA * (residual - e.detach()).pow(2).mean()

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
        print('  Codebook initialized via k-means')


class RQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder  = Encoder()
        self.decoder  = Decoder()
        self.quantizer = ResidualQuantizer()

    def forward(self, x):
        z = self.encoder(x)
        all_codes, z_q, rq_loss = self.quantizer(z)
        x_recon = self.decoder(z_q)
        recon_loss = (x - x_recon).pow(2).mean()
        total_loss = recon_loss + rq_loss
        return all_codes, total_loss, recon_loss, rq_loss


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def compute_metrics(model, data_tensor):
    """
    Compute per-level codebook usage and unique_rate over the full dataset.

    unique_rate = fraction of items with a unique (c0, c1, c2) triple.
    1 - unique_rate = collision_rate (lower is better).
    """
    model.eval()
    all_codes_per_level = [[] for _ in range(len(K_LEVELS))]

    with torch.no_grad():
        for i in range(0, len(data_tensor), BATCH_SIZE):
            batch = data_tensor[i:i + BATCH_SIZE]
            z = model.encoder(batch)
            codes, _, _ = model.quantizer(z)
            for level, code_tensor in enumerate(codes):
                all_codes_per_level[level].extend(code_tensor.cpu().tolist())

    model.train()

    n_items = len(all_codes_per_level[0])
    usages = [len(set(codes)) / k for codes, k in zip(all_codes_per_level, K_LEVELS)]
    all_sids = list(zip(*all_codes_per_level))
    unique_rate = len(set(all_sids)) / n_items

    return usages, unique_rate


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
    # L2-normalize: reduces RQ-VAE collapse on nomic embeddings
    emb_matrix /= np.clip(np.linalg.norm(emb_matrix, axis=1, keepdims=True), 1e-12, None)
    print(f'加载 embedding: {emb_matrix.shape}')

    data_tensor = torch.tensor(emb_matrix, dtype=torch.float32, device=device)
    n_items = len(data_tensor)

    torch.manual_seed(42)
    np.random.seed(42)
    model = RQVAE().to(device)

    print('K-means 初始化码本...')
    with torch.no_grad():
        z_all = model.encoder(data_tensor)
    model.quantizer.kmeans_init(z_all)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )

    ckpt_dir  = os.path.join(os.path.dirname(emb_dir), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'rqvae_best.pt')

    best_unique_rate = 0.0

    print(f'\n开始训练 RQ-VAE，共 {NUM_EPOCHS} epochs\n')
    for epoch in range(1, NUM_EPOCHS + 1):
        perm = torch.randperm(n_items, device=device)
        total_loss = recon_sum = rq_sum = 0.0
        n_batches  = 0

        for i in range(0, n_items, BATCH_SIZE):
            idx   = perm[i:i + BATCH_SIZE]
            batch = data_tensor[idx]

            _, loss, recon_loss, rq_loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()
            recon_sum  += recon_loss.item()
            rq_sum     += rq_loss.item()
            n_batches  += 1

        scheduler.step()

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

            if unique_rate > best_unique_rate:
                best_unique_rate = unique_rate
                torch.save({
                    'epoch':       epoch,
                    'model_state': model.state_dict(),
                    'unique_rate': unique_rate,
                    'usages':      usages,
                }, ckpt_path)
                print(f'  ✓ 保存最优 checkpoint (unique_rate={unique_rate:.1%})')

    print(f'\n训练完成，最优 unique_rate={best_unique_rate:.1%}')
    print(f'Checkpoint 保存至: {ckpt_path}')
    print('运行 python embedding/generate_rqvae_ids.py 生成 semantic_ids_rqvae.npy')


if __name__ == '__main__':
    train_rqvae()
