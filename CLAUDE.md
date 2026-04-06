# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A simplified reproduction of [TIGER (NeurIPS 2023)](https://arxiv.org/abs/2305.05065) — generative sequential recommendation using Semantic IDs. Items are encoded as hierarchical `(c1, c2, c3)` tuples via k-means on LLM embeddings. A GPT-2 model trained from scratch autoregressively generates the next item's Semantic ID tokens instead of scoring over all items.

**Core thesis**: The ablation experiment (random ID vs. LLM Semantic ID, same model) isolates whether semantic structure in item IDs actually helps, independent of the generative framework.

**Target metric**: Surpass SASRec's Recall@10 ≈ 0.06 (all-rank protocol, TIGER Table 1).

---

## Pipeline Commands

Run each step in order:

```bash
# 1. Preprocess raw Amazon Beauty data
python data/data_process.py

# 2. Extract LLM embeddings via local Ollama
#    nomic-embed-text: ~15 min; qwen2:7b: ~36 min (2 workers)
python embedding/extract_embeddings.py

# 3. Build Semantic IDs via hierarchical k-means (4/64/256)
python embedding/build_semantic_ids.py

# 4. Train generative model (Colab T4/A100 recommended)
python train.py

# 5. Evaluate (all-rank Recall@K / NDCG@K)
python evaluate.py

# 6. Train SASRec discriminative baseline
python baseline/sasrec_train.py
```

---

## Architecture

### Token Vocabulary (327 tokens total)

| Range | Meaning |
|-------|---------|
| 0–3 | Level-1 codes (c1, offset=0) |
| 4–67 | Level-2 codes (c2 + 4, offset=4) |
| 68–323 | Level-3 codes (c3 + 68, offset=68) |
| 324 | [BOS] |
| 325 | [EOS] |
| 326 | [PAD] |

Codebook `K_LEVELS = [4, 64, 256]` — coarse-to-fine hierarchy matches TIGER's RQ-VAE structure.

### Data Flow
```
item text → Ollama embedding → hierarchical k-means (4/64/256) → (c1, c2, c3)
user history → seq_to_tokens() → [BOS, c1¹, c2¹+4, c3¹+68, c1², ...]
→ GPT-2 (trained from scratch, 3.5M params, vocab=327)
→ beam search (width=50) → top-k (c1*, c2*, c3*) tuples
→ exact match or Hamming nearest-neighbor → item_id
→ Recall@K / NDCG@K  (all-rank, no negative sampling)
```

### File Responsibilities

- `model/tokenizer.py` — `K_LEVELS`, `LEVEL_OFFSETS`, `VOCAB_SIZE`/`BOS`/`EOS`/`PAD` constants; `seq_to_tokens()`, `item_to_tokens()`, `tokens_to_semantic_id()`
- `model/generative_rec.py` — `build_model()` constructs GPT2Config from scratch using imported `VOCAB_SIZE`
- `model/inference.py` — `build_reverse_index()`, `hamming_nearest()`, `predict_topk()` with beam search; uses `tokens_to_semantic_id()` and `K_LEVELS` for range validation
- `embedding/extract_embeddings.py` — calls local Ollama; change `MODEL` variable to swap embedding source
- `embedding/build_semantic_ids.py` — 3-level MiniBatchKMeans with `K_LEVELS = [4, 64, 256]`
- `train.py` — `RecDataset(augment=True/False)`: sliding window for training, single-target for validation; AdamW + CosineAnnealingLR
- `evaluate.py` — **all-rank** Recall@K / NDCG@K; no negative sampling
- `baseline/sasrec_train.py` — SASRec with BPR loss; `evaluate_sasrec_allrank()` for all-rank eval

### Generated Artifacts (not in git)

- `data/beauty_data.pkl` — processed sequences: `train`/`val`/`test`/`item2id`/`item_metas` **(tracked in git)**
- `embedding/item_embeddings_raw.npy` — raw LLM embeddings, `{item_id: np.array}`
- `embedding/semantic_ids.npy` — `{item_id: (c1, c2, c3)}`
- `checkpoints/best_model.pt` — best generative model checkpoint
- `checkpoints/sasrec_best.pt` — best SASRec checkpoint

---

## Key Design Decisions

- **Evaluation protocol**: All-rank Recall@K / NDCG@K (TIGER paper standard). The old 99-neg HR@K was unfair to the generative model — beam search can't meaningfully compete when the target is randomly appended if missed.
- **Codebook 4/64/256**: Coarse-to-fine; c1 has only 4 choices, making beam search step 1 accurate. Total capacity 4×64×256 = 65,536 > 12,101 items.
- **Sliding window training**: `augment=True` generates all `t` sub-sequences per user, increasing training samples ~5-10x (~110K → ~500K+).
- **Training loss**: Only the last 3 tokens (target item's c1/c2/c3) contribute to loss — history tokens are masked to -100.
- **Beam search fallback**: When a generated `(c1, c2, c3)` has no exact match, `hamming_nearest()` finds the closest item.
- **Embedding model is modular**: Change `MODEL` in `extract_embeddings.py` only; all downstream code is unaffected.

---

## Ablation Experiment (most important)

To generate random IDs (control condition — same model, no semantic structure):

```python
import random
random_semantic_ids = {
    item_id: (random.randint(0, 3), random.randint(0, 63), random.randint(0, 255))
    for item_id in all_items
}
```

Save as `embedding/semantic_ids_random.npy`, retrain, evaluate. A significant Recall@10 drop vs. LLM Semantic ID proves semantic structure is the key contributor.

---

## TIGER Reference Numbers (all-rank, Amazon Beauty)

| Model | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|-------|----------|--------|-----------|---------|
| SASRec (TIGER paper) | 0.0387 | 0.0249 | 0.0605 | 0.0318 |
| TIGER | 0.0454 | 0.0321 | 0.0648 | 0.0384 |
| Our SASRec (all-rank) | TBD | TBD | TBD | TBD |
| Our Generative | TBD | TBD | TBD | TBD |
| Generative + Random ID | TBD | TBD | TBD | TBD |
