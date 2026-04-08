# Generative Sequential Recommendation via Semantic IDs

A simplified reproduction of [TIGER (NeurIPS 2023)](https://arxiv.org/abs/2305.05065).
This repository studies whether hierarchical Semantic IDs help autoregressive
next-item generation on Amazon Beauty.

The repo has three layers of information:

- Current code path: what the scripts do today (this README).
- Recorded experiments: numbers in [Progress.md](./Progress.md).
- Improvement plan: [plan.md](./plan.md).

## Current Code Path

### Pipeline

1. `data/data_process.py` → `beauty_data.pkl`
2. `embedding/extract_embeddings.py` (`nomic-embed-text` via Ollama)
   → `embedding/item_embeddings_raw.npy`
3. `embedding/rqvae.py` trains the RQ-VAE on the nomic embeddings
   → `checkpoints/rqvae_best.pt`
4. `embedding/generate_rqvae_ids.py` loads the best checkpoint, runs argmin
   quantization, resolves collisions with a 4th code
   → `embedding/semantic_ids_rqvae.npy`
5. `train.py` trains a from-scratch T5 encoder-decoder (~4.6M params); early
   stopping on validation NDCG@10 → `checkpoints/best_model_t5.pt`
6. `evaluate.py` runs all-rank Recall@K / NDCG@K on the test set.

The `baseline/sasrec_train.py` SASRec baseline is independent of the
generative path.

### Tokenizer

`model/tokenizer.py` uses a **4-token Semantic ID** layout:

- `K_LEVELS = [256, 256, 256, 64]`
- first 3 levels are RQ-VAE codes; the 4th is collision resolution
- `seq_to_t5_tokens()` flattens history into encoder input
- vocab = `sum(K_LEVELS) + 3` (BOS / EOS / PAD)

### RQ-VAE training (`embedding/rqvae.py`)

- 5-hidden encoder `[768→512→256→128→64→32]`, latent dim 32
- 3 residual VQ levels with K = `[256,256,256]`
- Lazy k-means init at first forward
- Sinkhorn balanced assignment on the last codebook only (`sk_epsilons=[0,0,0.003]`)
- AdamW + linear warmup/decay, 3000 epoch
- Saves the checkpoint with the highest `unique_rate` as `rqvae_best.pt`

### Inference

`model/inference.py` uses constrained beam search via
`LevelConstrainedLogitsProcessor`, generating exactly `len(K_LEVELS)` tokens
per item. Beam width 50, no Hamming fallback.

## Results

| Run | R@5 | N@5 | R@10 | N@10 |
|-----|-----|-----|------|------|
| SASRec baseline (ours) | 0.0358 | 0.0180 | 0.0573 | 0.0250 |
| **Ours (nomic + RQ-VAE 3kep + T5)** | **0.0369** | **0.0242** | **0.0589** | **0.0312** |
| TIGER (paper) | 0.0454 | 0.0321 | 0.0648 | 0.0384 |
| TIGER (community reimpl) | — | — | 0.0594 | 0.0321 |

Generative NDCG@10 is **+24.8%** over SASRec. Ours is within split noise of
the community TIGER reimplementation. See [Progress.md](./Progress.md) for
the tech-selection narrative.

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
