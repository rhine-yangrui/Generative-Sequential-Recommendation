# Generative Sequential Recommendation via Semantic IDs

A simplified reproduction of [TIGER (NeurIPS 2023)](https://arxiv.org/abs/2305.05065).
Items are encoded as hierarchical **Semantic IDs** derived from LLM embeddings, and an autoregressive GPT-2 model generates the next item's ID directly — no scoring over all items needed.

**Key finding**: The ablation experiment (random ID vs. semantic ID) is the core contribution — it isolates whether semantic structure in item IDs actually helps the autoregressive model, independent of the generative framework itself.

---

## Architecture

```
User History [i₁, i₂, ..., iₙ]
        │
        ▼  each item → 3 Semantic ID tokens  (hierarchical k-means on LLM embeddings)
        │  Codebook: K1=4 (coarse), K2=16 (mid), K3=256 (fine)
        │
[c₁¹, c₂¹+4, c₃¹+20,   c₁², c₂²+4, c₃²+20,   ...]
        │
        ▼
┌─────────────────────────────┐
│  GPT-2 (trained from scratch)│  ← causal LM, vocab size = 279
│  3.5M params                 │    loss only on target item's 3 tokens
└────────────┬────────────────┘
             │  beam search (beam_width=50)
             ▼
      (c₁*, c₂*, c₃*)          ← generated Semantic ID
             │
             ├─ exact match → item
             └─ Hamming nearest neighbor → item
             ▼
      Recall@5 / Recall@10 / NDCG@10
```

**Token vocabulary design** (279 tokens total):

| Range | Meaning |
|-------|---------|
| 0 – 3 | Level-1 codes (c₁, 4 coarse categories) |
| 4 – 19 | Level-2 codes (c₂ + 4, 16 subcategories) |
| 20 – 275 | Level-3 codes (c₃ + 20, 256 fine-grained) |
| 276 | [BOS] |
| 277 | [EOS] |
| 278 | [PAD] |

Codebook structure `4/16/256` aligns with TIGER's RQ-VAE design: coarse-to-fine hierarchy improves beam search accuracy at each step.

---

## Results

**Evaluation protocol**: All-rank Recall@K / NDCG@K — same as TIGER (NeurIPS 2023).
Beam search generates top-K recommendations; check if ground truth is in top-K.

| Model | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|-------|----------|--------|-----------|---------|
| SASRec (TIGER paper) | 0.0387 | 0.0249 | 0.0605 | 0.0318 |
| TIGER (original paper) | 0.0454 | 0.0321 | 0.0648 | 0.0384 |
| **SASRec (ours, all-rank)** | 0.0222 | 0.0114 | 0.0404 | 0.0172 |
| Generative + Random ID (ablation) | TBD | TBD | TBD | TBD |
| **Generative + nomic-embed-text** | TBD | TBD | TBD | TBD |

> Latest SASRec run: best val Recall@10 = 0.0543. Current target remains to close the gap to TIGER's SASRec Recall@10 = 0.0605.

---

## Model Configuration

### Generative Model (GPT-2, trained from scratch)

| Hyperparameter | Value | Note |
|----------------|-------|------|
| Architecture | GPT-2 Decoder-only | causal LM, no pretrained weights |
| vocab_size | 279 | 4+16+256 Semantic ID tokens + BOS/EOS/PAD |
| n_embd | 256 | hidden dimension |
| n_layer | 4 | Transformer blocks |
| n_head | 4 | attention heads |
| n_positions | 512 | max sequence length |
| Parameters | ~3.5M | |
| Loss | Cross-entropy (target tokens only) | history tokens masked with -100 |
| Optimizer | AdamW | lr=1e-3, weight_decay=0.01 |
| Scheduler | CosineAnnealingLR | eta_min=1e-5 |
| Epochs | 30 | early stopping, patience=10 |
| Batch size | 128 | |
| Training data | Sliding window augmentation | ~5-10x samples vs. single-target |

### SASRec Baseline (discriminative, ID-based)

| Hyperparameter | Value | Note |
|----------------|-------|------|
| Architecture | Transformer Encoder | self-attentive, causal mask |
| maxlen | 50 | max history length |
| hidden_size | 128 | latest reported run |
| num_layers | 2 | |
| num_heads | 1 | |
| dropout | 0.5 | latest reported run |
| Loss | BPR | 1 negative sample per positive |
| Optimizer | Adam | lr=1e-3 |
| Epochs | 200 | early stopping, patience=20 |
| Validation | every 10 epochs | model selection by Recall@10 |
| Batch size | 256 | |

### Semantic ID Construction

| Step | Detail |
|------|--------|
| Embedding model | Local Ollama — modular, swap one line to change model |
| Clustering | Hierarchical k-means, structure 4/16/256 |
| Capacity | 4×16×256 = 16,384 > 12,101 items |
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
| Evaluation | All-rank (no negative sampling) |

---

## Quickstart

```bash
pip install -r requirements.txt

# Install Ollama and pull embedding model
brew install ollama
ollama pull nomic-embed-text   # or qwen2:7b for potentially better embeddings

# 1. Preprocess
python data/data_process.py

# 2. Extract LLM embeddings (~15 min with nomic-embed-text)
python embedding/extract_embeddings.py

# 3. Build Semantic IDs (4/16/256 hierarchy)
python embedding/build_semantic_ids.py

# 4. Train generative model (Colab recommended)
python train.py

# 5. Evaluate (all-rank Recall@K / NDCG@K)
python evaluate.py

# 6. Train SASRec baseline
python baseline/sasrec_train.py
```

---

## Design Choices vs TIGER

| Dimension | TIGER (original) | This project |
|-----------|-----------------|--------------|
| Semantic ID construction | RQ-VAE (trained) | Hierarchical k-means (no training needed) |
| Codebook structure | 4/16/256 | 4/16/256 ✓ |
| Autoregressive model | T5-Small (Encoder-Decoder, 60M) | GPT-2 (Decoder-only, 3.5M) |
| LLM embedding source | text-embedding-ada-002 (OpenAI API) | Local Ollama — no API key needed |
| Evaluation | All-rank Recall@K / NDCG@K | All-rank Recall@K / NDCG@K ✓ |

---

## References

- [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (NeurIPS 2023)
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (ICDM 2018)
- [OneRec: Kuaishou's End-to-End Generative Recommendation](https://arxiv.org/abs/2501.18653) (2025)
- [HLLM: Hierarchical Large Language Models for Sequential Recommendation](https://arxiv.org/abs/2409.12740) (ByteDance, 2024)
