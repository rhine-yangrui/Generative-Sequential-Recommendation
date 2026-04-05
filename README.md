# Generative Sequential Recommendation via Semantic IDs

Inspired by [TIGER (NeurIPS 2023)](https://arxiv.org/abs/2305.05065). Items are encoded as hierarchical Semantic IDs derived from Qwen2.5-7B embeddings, and an autoregressive GPT-2 model generates the next item's ID directly — no scoring over all items needed.

## Architecture

```
User History [i₁, i₂, ..., iₙ]
        │
        ▼  (each item → 3 Semantic ID tokens via hierarchical k-means on LLM embeddings)
[c₁¹, c₂¹, c₃¹,  c₁², c₂², c₃²,  ...,  c₁ⁿ, c₂ⁿ, c₃ⁿ]
        │
        ▼
┌──────────────────┐
│  GPT-2 (custom)  │  ← autoregressive Transformer decoder
│  vocab: 771      │    trained with causal LM objective
└────────┬─────────┘
         │  beam search
         ▼
   (c₁*, c₂*, c₃*)      ← generated Semantic ID
         │
         ▼
   nearest item lookup   ← map back to item
         │
         ▼
   HR@10 / NDCG@10
```

## Results

| Model | HR@1 | HR@5 | HR@10 | NDCG@10 |
|-------|------|------|-------|---------|
| SASRec (ID-based baseline) | - | - | - | - |
| Generative + Random ID (ablation) | - | - | - | - |
| **Generative + LLM Semantic ID (ours)** | - | - | - | - |
| TIGER (original paper, RQ-VAE + T5) | 0.2134 | 0.4521 | 0.5498 | 0.3638 |

*Results on Amazon Beauty. To be filled after experiments.*

## Quickstart

```bash
# 1. Preprocess data
python data/data_process.py

# 2. Extract LLM embeddings (requires Ollama + qwen2.5:7b)
python embedding/extract_embeddings.py

# 3. Build Semantic IDs via hierarchical k-means
python embedding/build_semantic_ids.py

# 4. Train
python train.py

# 5. Evaluate
python evaluate.py
```

## Setup

```bash
pip install -r requirements.txt

# Install Ollama and pull Qwen2.5-7B for embedding extraction
brew install ollama
ollama pull qwen2.5:7b
```

## Key Design Choices

**Semantic ID vs Random ID**: Items are assigned hierarchical codes based on their LLM semantic embeddings (via k-means clustering), not arbitrary integers. This allows the autoregressive model to leverage semantic structure — generating `c₁=12` implicitly means "searching within skincare products."

**k-means vs RQ-VAE**: The original TIGER uses RQ-VAE for Semantic ID construction. We simplify this to hierarchical k-means, which requires no training and is easier to implement while preserving the core semantic structure.

**Qwen2.5-7B vs text-embedding-ada-002**: The original TIGER uses OpenAI's embedding API. We use a locally-hosted Qwen2.5-7B via Ollama, making the pipeline fully reproducible without API keys.

## Dataset

Amazon Beauty (5-core) from [SNAP](http://snap.stanford.edu/data/amazon/).
~22,363 users, ~12,101 items, ~198,502 interactions.
Same dataset as TIGER original paper — results directly comparable.

## Reference

- [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (NeurIPS 2023)
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- [OneRec: Kuaishou's End-to-End Generative Recommendation](https://arxiv.org/abs/2501.18653)
