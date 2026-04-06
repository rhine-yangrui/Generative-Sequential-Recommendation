# Generative Sequential Recommendation via Semantic IDs

A simplified reproduction of [TIGER (NeurIPS 2023)](https://arxiv.org/abs/2305.05065).
This repository studies whether hierarchical Semantic IDs help autoregressive
next-item generation on Amazon Beauty.

The repo has three layers of information:

- Current code path: what the scripts do today (this README).
- Recorded experiments: numbers in [Progress.md](./Progress.md).
- Outstanding issues: [improve_generative_prompt.md](./improve_generative_prompt.md).

## Current Code Path

### Pipeline

1. `data/data_process.py` → `beauty_data.pkl`
2. `embedding/extract_embeddings.py` (default `nomic-embed-text`)
   → `embedding/item_embeddings_raw.npy`
3. `embedding/rqvae.py` trains the RQ-VAE on the nomic embeddings
   → `checkpoints/rqvae_best.pt`
4. `embedding/generate_rqvae_ids.py` loads the best checkpoint, runs argmin
   quantization, resolves collisions with a 4th code
   → `embedding/semantic_ids_rqvae.npy`
5. `train.py` trains a from-scratch GPT-2 decoder; early stopping on
   validation Recall@10 → `checkpoints/best_model.pt`
6. `evaluate.py` runs all-rank Recall@K / NDCG@K on the test set.

The `baseline/sasrec_train.py` SASRec baseline is independent of the
generative path.

### Tokenizer

`model/tokenizer.py` uses a **4-token Semantic ID** layout:

- `K_LEVELS = [4, 16, 256, 512]`
- first 3 levels are RQ-VAE codes; the 4th is collision resolution
- `seq_to_tokens()` inserts `EOS` between items
- vocab = `sum(K_LEVELS) + 3` (BOS / EOS / PAD)

### RQ-VAE training (`embedding/rqvae.py`)

- Encoder/decoder MLP, latent dim 32
- 3 residual VQ levels with K = `[4, 16, 256]`
- Sinkhorn balanced assignment on L1 / L2 (log-domain, MPS-safe)
- 50-epoch encoder warmup → k-means init → 300-epoch joint training
- AdamW + cosine LR + grad clip; periodic dead-code reset
- Saves the checkpoint with the highest `unique_rate`

### Inference

`model/inference.py` uses constrained beam search via
`LevelConstrainedLogitsProcessor`, generating exactly `len(K_LEVELS)` tokens
per item and falling back to Hamming nearest-neighbor for any unmatched ID.

## Results

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Notes |
|-----|----------|--------|-----------|---------|-------|
| SASRec (TIGER paper) | 0.0387 | 0.0249 | 0.0605 | 0.0318 | reference |
| TIGER (paper) | 0.0454 | 0.0321 | 0.0648 | 0.0384 | reference |
| SASRec baseline (ours, latest) | 0.0358 | 0.0180 | 0.0573 | 0.0250 | E3b |
| Generative + qwen2:7b kmeans `4/64/256` | 0.0197 | 0.0122 | 0.0322 | 0.0162 | E1, archived |
| Generative + Random ID ablation | 0.0025 | 0.0016 | 0.0042 | 0.0021 | E2, archived |
| Generative + RQ-VAE `4/16/256` (+c4) | TBD | TBD | TBD | TBD | E4, training in progress |

See [Progress.md](./Progress.md) for per-experiment detail.

## Known Issue

The current RQ-VAE setup `[4,16,256]` only reaches **unique_rate ≈ 41%**
on Beauty's 12,101 items (max collision group 24). All three codebooks are
fully used, so this is a capacity ceiling, not collapse: `4 × 16 × 256 = 16384`
joint slots are too few once the residual quantization is non-uniform.
See [improve_generative_prompt.md](./improve_generative_prompt.md) — the
recommended fix is `K_LEVELS=[256,256,256]` if downstream metrics are
unsatisfactory.

## Dataset

Amazon Beauty 5-core ([SNAP](http://snap.stanford.edu/data/amazon/)).

| Stat | Value |
|------|-------|
| Users | 22,363 |
| Items | 12,101 |
| Interactions | 198,502 |
| Split | Leave-one-out |
| Evaluation | all-rank Recall@K / NDCG@K |

## Quickstart

```bash
pip install -r requirements.txt
brew install ollama && ollama pull nomic-embed-text

python data/data_process.py
python embedding/extract_embeddings.py
python embedding/rqvae.py
python embedding/generate_rqvae_ids.py
python train.py
python evaluate.py

# Independent baseline
python baseline/sasrec_train.py
```

## References

- [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- [XiaoLongtaoo/TIGER reproduction](https://github.com/XiaoLongtaoo/TIGER)
