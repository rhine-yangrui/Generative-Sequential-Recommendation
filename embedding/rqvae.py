"""
RQ-VAE: Residual-Quantized Variational AutoEncoder for Semantic ID generation.

Architecture:
  - Encoder: 768 -> 512 -> 256 -> 128 -> 32
  - Decoder: 32 -> 128 -> 256 -> 512 -> 768
  - 3 codebooks: [4, 16, 256], code dim = 32
  - Loss: reconstruction loss + residual quantization loss

Usage:
    python embedding/rqvae.py
"""

import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
LATENT_DIM = 32
K_LEVELS = [4, 16, 256]
COLLISION_K = 512
BETA = 0.25
LR = 0.4
BATCH_SIZE = 1024
NUM_EPOCHS = 500
MIN_USAGE = 0.80
WARMUP_EPOCHS = 20
SEED_CANDIDATES = [40, 41, 42, 43, 44]


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
    """
    Three-level residual quantizer with straight-through gradients.
    """
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
        residual = z_samples.detach().cpu().numpy()
        with torch.no_grad():
            for codebook, k in zip(self.codebooks, self.k_levels):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(residual)
                codebook.data = torch.tensor(
                    km.cluster_centers_, dtype=torch.float32, device=codebook.device
                )
                labels = km.labels_
                centroids = km.cluster_centers_
                residual = residual - centroids[labels]
        print("  Codebook initialized via k-means")


class RQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
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


def compute_codebook_usage(model, data_tensor, batch_size=BATCH_SIZE):
    model.eval()
    all_codes_per_level = [[] for _ in range(len(K_LEVELS))]
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i + batch_size]
            z = model.encoder(batch)
            codes, _, _ = model.quantizer(z)
            for level, code_tensor in enumerate(codes):
                all_codes_per_level[level].extend(code_tensor.cpu().numpy().tolist())
    model.train()

    usages = []
    for codes_list, k in zip(all_codes_per_level, K_LEVELS):
        usage = len(set(codes_list)) / k
        usages.append(usage)
    return usages


def usage_score(usages):
    score = 0.0
    for level, usage in enumerate(usages):
        score += usage * (10 ** level)
    return float(score)


def train_epochs(model, optimizer, data_tensor, n_items, device, start_epoch, end_epoch):
    last_avg_loss = None
    for epoch in range(start_epoch, end_epoch + 1):
        perm = torch.randperm(n_items, device=device)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_items, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            batch = data_tensor[idx]

            _, loss, _, _ = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        last_avg_loss = total_loss / n_batches

    return last_avg_loss


def warmup_seed_search(data_tensor, n_items, device):
    best = None

    print('开始 warmup seed search...')
    for seed in SEED_CANDIDATES:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = RQVAE().to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)

        with torch.no_grad():
            z_all = model.encoder(data_tensor)
        model.quantizer.kmeans_init(z_all)
        model.train()

        avg_loss = train_epochs(
            model, optimizer, data_tensor, n_items, device, start_epoch=1, end_epoch=WARMUP_EPOCHS
        )
        usages = compute_codebook_usage(model, data_tensor)
        score = usage_score(usages)

        print(
            f'  seed={seed}  warmup_loss={avg_loss:.4f}  '
            f'usage=' + '  '.join([f'L{i}={u:.1%}' for i, u in enumerate(usages)])
        )

        candidate = {
            'seed': seed,
            'model': model,
            'optimizer': optimizer,
            'usages': usages,
            'score': score,
            'avg_loss': avg_loss,
        }
        if best is None or candidate['score'] > best['score']:
            best = candidate

    print(
        f'选择 seed={best["seed"]} 作为正式训练起点，'
        f'usage=' + '  '.join([f'L{i}={u:.1%}' for i, u in enumerate(best['usages'])])
    )
    return best


def resolve_collisions(semantic_ids):
    sid_to_items = defaultdict(list)
    for item_id, sid in semantic_ids.items():
        sid_to_items[sid].append(item_id)

    collisions = {sid: items for sid, items in sid_to_items.items() if len(items) > 1}
    print(
        f'冲突 Semantic ID 数: {len(collisions)}（共涉及 '
        f'{sum(len(v) for v in collisions.values())} items）'
    )

    if not collisions:
        return {item_id: (*sid, 0) for item_id, sid in semantic_ids.items()}

    max_group_size = max(len(items) for items in sid_to_items.values())
    if max_group_size > COLLISION_K:
        raise RuntimeError(
            f'最大 collision group 大小为 {max_group_size}，超过 c4 容量 {COLLISION_K}。'
        )

    resolved = {}
    for sid, items in sid_to_items.items():
        for collision_idx, item_id in enumerate(items):
            resolved[item_id] = (*sid, collision_idx)

    return resolved


def train_rqvae():
    device = select_device()
    print(f'使用设备: {device}')

    emb_dir = os.path.dirname(os.path.abspath(__file__))
    raw = np.load(
        os.path.join(emb_dir, 'item_embeddings_raw_nomic.npy'),
        allow_pickle=True,
    ).item()

    item_ids = sorted(raw.keys())
    emb_matrix = np.stack([raw[i] for i in item_ids]).astype(np.float32)
    # nomic embeddings 先做 L2 normalize，能显著降低 RQ-VAE collapse 的概率。
    emb_matrix /= np.clip(np.linalg.norm(emb_matrix, axis=1, keepdims=True), 1e-12, None)
    print(f"加载 embedding: {emb_matrix.shape}")

    data_tensor = torch.tensor(emb_matrix, dtype=torch.float32, device=device)
    n_items = len(data_tensor)

    best = warmup_seed_search(data_tensor, n_items, device)
    model = best['model']
    optimizer = best['optimizer']

    print(f'\n开始训练 RQ-VAE，最多 {NUM_EPOCHS} epochs\n')
    for epoch in range(WARMUP_EPOCHS + 1, NUM_EPOCHS + 1):
        avg_loss = train_epochs(
            model, optimizer, data_tensor, n_items, device, start_epoch=epoch, end_epoch=epoch
        )
        if epoch % 20 == 0:
            usages = compute_codebook_usage(model, data_tensor)
            usage_str = '  '.join([f'L{i}={u:.1%}' for i, u in enumerate(usages)])
            print(f'Epoch {epoch:4d}  loss={avg_loss:.4f}  codebook usage: {usage_str}')

            if all(u >= MIN_USAGE for u in usages):
                print(f'\n所有层码本使用率 >= {MIN_USAGE:.0%}，提前结束训练')
                break

    print('\n提取 Semantic IDs...')
    model.eval()
    all_codes_levels = [[] for _ in range(len(K_LEVELS))]
    with torch.no_grad():
        for i in range(0, n_items, BATCH_SIZE):
            batch = data_tensor[i:i + BATCH_SIZE]
            z = model.encoder(batch)
            codes, _, _ = model.quantizer(z)
            for level, code_tensor in enumerate(codes):
                all_codes_levels[level].extend(code_tensor.cpu().numpy().tolist())

    semantic_ids = {}
    for idx, item_id in enumerate(item_ids):
        c0 = all_codes_levels[0][idx]
        c1 = all_codes_levels[1][idx]
        c2 = all_codes_levels[2][idx]
        semantic_ids[item_id] = (c0, c1, c2)

    semantic_ids = resolve_collisions(semantic_ids)

    all_sids = [tuple(v) for v in semantic_ids.values()]
    unique_count = len(set(all_sids))
    print(f'解决后唯一 Semantic ID 数: {unique_count} / {len(all_sids)}')

    output_path = os.path.join(emb_dir, 'semantic_ids_rqvae.npy')
    np.save(output_path, semantic_ids)
    print(f'已保存至 {output_path}')

    c0_vals = [v[0] for v in semantic_ids.values()]
    print(f'c0 分布: {sorted(Counter(c0_vals).items())}')


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    train_rqvae()
