# LLM-Enhanced Sequential Recommendation

Aligning LLM semantic embeddings with collaborative filtering via contrastive learning, inspired by [RLMRec (WWW 2024)](https://arxiv.org/abs/2310.15950).

## Architecture

```
User Interaction Sequence [i₁, i₂, ..., iₙ]
        │
        ▼
┌─────────────────┐        ┌──────────────────────┐
│   SASRec        │        │  Qwen2.5-7B (offline) │
│  (ID Embedding) │        │  Item Semantic Emb     │
│                 │        │  (pre-extracted)        │
└────────┬────────┘        └──────────┬─────────────┘
         │                            │
         │      ┌─────────────────────┘
         ▼      ▼
   ┌─────────────────┐
   │  InfoNCE Loss   │  ← Align ID emb and LLM emb
   │  (Contrastive)  │    for the same item
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Fusion Layer   │  ← Weighted sum of ID & LLM scores
   └────────┬────────┘
            ▼
     HR@10 / NDCG@10
```

## Results

| Model | HR@10 | NDCG@10 |
|-------|-------|---------|
| SASRec (ID only) | - | - |
| LLM Emb + KNN (no training) | - | - |
| SASRec + concat (no alignment loss) | - | - |
| **SASRec + LLM Alignment (ours)** | - | - |

*Results on Amazon Beauty dataset. To be filled after experiments.*

## Quickstart

```bash
# 1. Preprocess data
python data/data_process.py

# 2. Extract LLM embeddings (requires Ollama + qwen2.5:7b)
python embedding/extract_embeddings.py

# 3. Train
python train.py
```

## Setup

```bash
pip install -r requirements.txt

# Install Ollama and pull Qwen2.5-7B for embedding extraction
brew install ollama
ollama pull qwen2.5:7b
```

## Project Structure

```
├── data/
│   ├── data_process.py       # Data preprocessing (5-core filtering, leave-one-out split)
│   └── beauty_data.pkl       # Processed dataset (generated)
├── model/
│   ├── sasrec.py             # SASRec baseline
│   └── sasrec_align.py       # SASRec + LLM alignment
├── embedding/
│   ├── extract_embeddings.py # Qwen2.5-7B embedding extraction via Ollama
│   └── item_llm_embeddings.npy  # Pre-extracted embeddings (generated, not tracked)
├── notebooks/
│   └── analysis.ipynb        # Ablation results & t-SNE visualization
├── train.py                  # Training entry point
├── evaluate.py               # Evaluation (leave-one-out, 99 negative sampling)
└── requirements.txt
```

## Dataset

Amazon Beauty (5-core) from [SNAP](http://snap.stanford.edu/data/amazon/).
~22,363 users, ~12,101 items, ~198,502 interactions.

## Reference

- [RLMRec: Representation Learning with Large Language Models for Recommendation](https://arxiv.org/abs/2310.15950) (WWW 2024)
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
