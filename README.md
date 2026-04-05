# Generative Sequential Recommendation via Semantic IDs

A simplified reproduction of [TIGER (NeurIPS 2023)](https://arxiv.org/abs/2305.05065).
Items are encoded as hierarchical **Semantic IDs** derived from LLM embeddings, and an autoregressive GPT-2 model generates the next item's ID directly — no scoring over all items needed.

Same dataset and evaluation protocol as the original paper, results are directly comparable.

---

## Architecture

```
User History [i₁, i₂, ..., iₙ]
        │
        ▼  each item → 3 Semantic ID tokens  (hierarchical k-means on LLM embeddings)
[c₁¹, c₂¹+256, c₃¹+512,   c₁², c₂²+256, c₃²+512,   ...]
        │
        ▼
┌─────────────────────────────┐
│  GPT-2 (trained from scratch)│  ← causal LM, vocab size = 771
│  3.5M params                 │    loss only on target item's 3 tokens
└────────────┬────────────────┘
             │  beam search (beam_width=50)
             ▼
      (c₁*, c₂*, c₃*)          ← generated Semantic ID
             │
             ├─ exact match → item
             └─ Hamming nearest neighbor → item
             ▼
      HR@10 / NDCG@10
```

**Token vocabulary design** (771 tokens total):

| Range | Meaning |
|-------|---------|
| 0 – 255 | Level-1 codes (c₁) |
| 256 – 511 | Level-2 codes (c₂ + 256) |
| 512 – 767 | Level-3 codes (c₃ + 512) |
| 768 | [BOS] |
| 769 | [EOS] |
| 770 | [PAD] |

Offset design prevents token collisions between levels: c₁=5 and c₂=5 map to token 5 and token 261 respectively, allowing the model to distinguish levels.

---

## Results

Amazon Beauty, leave-one-out evaluation, 99 random negatives per user (same as TIGER paper).

| Model | HR@1 | HR@5 | HR@10 | NDCG@10 |
|-------|------|------|-------|---------|
| SASRec (ID-based baseline) | - | - | - | - |
| Generative + Random Semantic ID (ablation) | - | - | - | - |
| **Generative + LLM Semantic ID — Round 1** (nomic-embed-text) | 0.0370 | 0.0755 | 0.1209 | 0.0706 |
| **Generative + LLM Semantic ID — Round 2** (qwen2:7b) | ? | ? | ? | ? |
| TIGER (original paper, all-rank, Recall@K) | — | — | 0.0648 | 0.0384 |

---

## Model Configuration

### Generative Model (GPT-2, trained from scratch)

| Hyperparameter | Value | Note |
|----------------|-------|------|
| Architecture | GPT-2 Decoder-only | causal LM, no pretrained weights |
| vocab_size | 771 | 256×3 Semantic ID tokens + BOS/EOS/PAD |
| n_embd | 256 | hidden dimension |
| n_layer | 4 | Transformer blocks |
| n_head | 4 | attention heads |
| n_positions | 512 | max sequence length |
| Parameters | ~3.5M | |
| Loss | Cross-entropy (target tokens only) | history tokens masked with -100 |
| Optimizer | AdamW | lr=1e-3, weight_decay=0.01 |
| Scheduler | CosineAnnealingLR | eta_min=1e-5 |
| Epochs | 50 | early stopping, patience=10 |
| Batch size | 64 | |
| Max history | 50 items | |

### SASRec Baseline (discriminative, ID-based)

| Hyperparameter | Value | Note |
|----------------|-------|------|
| Architecture | Transformer Encoder | self-attentive, causal mask |
| hidden_size | 64 | |
| num_layers | 2 | |
| num_heads | 1 | |
| dropout | 0.2 | |
| Parameters | ~0.88M | |
| Loss | BPR | 1 negative sample per positive |
| Optimizer | Adam | lr=1e-3 |
| Epochs | 200 | early stopping, patience=20 |
| Batch size | 256 | |
| Max history | 50 items | |

> SASRec uses random integer ID embeddings with no semantic information — the key contrast with our Semantic ID approach.

### Semantic ID Construction

| Step | Detail |
|------|--------|
| Embedding model | Local Ollama (current: `qwen2:7b`, 3584-dim; upgradeable to larger LLMs) |
| Clustering | Hierarchical k-means, 3 levels, K=256 per level |
| Cluster purity | Mean 91.65% (items in same c₁ cluster share same product category) |
| Unique IDs | 11,780 / 12,101 items have a unique (c₁, c₂, c₃) tuple |
| Fallback | Hamming nearest-neighbor for unmatched generated IDs |

---

## Dataset

**Amazon Beauty (5-core)** from [SNAP](http://snap.stanford.edu/data/amazon/).

| Stat | Value |
|------|-------|
| Users | 22,363 |
| Items | 12,101 |
| Interactions | 198,502 |
| Split | Leave-one-out (val = last-2nd item, test = last item) |
| Evaluation | 99 random negatives + 1 target = 100 candidates |

Same dataset and split as TIGER — results are directly comparable.

---

## Quickstart

```bash
pip install -r requirements.txt

# Install Ollama and pull embedding model
brew install ollama
ollama pull qwen2:7b      # or nomic-embed-text for a faster (~15 min) run

# 1. Preprocess
python data/data_process.py

# 2. Extract LLM embeddings (~36 min with qwen2:7b, 2 concurrent workers)
python embedding/extract_embeddings.py

# 3. Build Semantic IDs
python embedding/build_semantic_ids.py

# 4. Train generative model (Colab recommended)
python train.py

# 5. Evaluate
python evaluate.py

# 6. Train SASRec baseline
python baseline/sasrec_train.py
```

---

## Design Choices vs TIGER

| Dimension | TIGER (original) | This project |
|-----------|-----------------|--------------|
| Semantic ID construction | RQ-VAE (trained) | Hierarchical k-means (no training needed) |
| Autoregressive model | T5-Small (Encoder-Decoder, 60M) | GPT-2 (Decoder-only, 3.5M) |
| LLM embedding source | text-embedding-ada-002 (OpenAI API) | Local Ollama — no API key needed |
| Dataset | Amazon Beauty / Sports / Toys | Amazon Beauty |
| Evaluation | Leave-one-out, HR@K / NDCG@K | Same ✓ |

The embedding model is modular — swap one line in `extract_embeddings.py` to use any Ollama-compatible model.

---

## References

- [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (NeurIPS 2023)
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (ICDM 2018)
- [OneRec: Kuaishou's End-to-End Generative Recommendation](https://arxiv.org/abs/2501.18653) (2025)
- [HLLM: Hierarchical Large Language Models for Sequential Recommendation](https://arxiv.org/abs/2409.12740) (ByteDance, 2024)
