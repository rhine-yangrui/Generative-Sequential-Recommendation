# Generative Sequential Recommendation via Semantic IDs

A simplified reproduction of [TIGER (NeurIPS 2023)](https://arxiv.org/abs/2305.05065).
This repository studies whether hierarchical Semantic IDs help autoregressive next-item generation on Amazon Beauty.

The repo currently contains three different layers of information:

- Current code path: what the scripts do today.
- Recorded experiments: what has already been run and logged in [Progress.md](./Progress.md).
- Planned improvements: the next-step upgrade plan in [improve_generative_prompt.md](./improve_generative_prompt.md).

This README is intentionally aligned to the current codebase, not to planned changes.

## Current Status

### What is implemented in code now

- Semantic tokenizer uses `K_LEVELS = [4, 64, 256]` in [model/tokenizer.py](./model/tokenizer.py).
- Sequence format is `[BOS, c1, c2+4, c3+68, ...]` with no per-item `EOS` separator yet.
- The generative model is a GPT-2 decoder trained from scratch in [model/generative_rec.py](./model/generative_rec.py).
- Training still uses early stopping on `val_loss` in [train.py](./train.py).
- Inference still uses unconstrained beam search in [model/inference.py](./model/inference.py).
- The active semantic ID artifact path is `embedding/semantic_ids.npy`.
- Embedding extraction currently defaults to `nomic-embed-text` in [embedding/extract_embeddings.py](./embedding/extract_embeddings.py).

### What has already been completed experimentally

The most important completed result is the ablation:

- Generative + semantic IDs clearly beats Generative + random IDs.
- This supports the project thesis that semantic structure in item IDs matters, not just the autoregressive framework.

For exact experiment logs and notes, use [Progress.md](./Progress.md) as the source of truth.

### What is next

The next planned step is to improve the generative pipeline with:

- `nomic-embed-text` as the main embedding path
- `4/16/256` codebooks
- `EOS` separators between items
- constrained beam search
- early stopping on validation `Recall@10`

Those changes are documented in [improve_generative_prompt.md](./improve_generative_prompt.md) and are not implemented in the main code yet.

## Current Architecture

```text
User History [i1, i2, ..., in]
        |
        v  each item -> 3 Semantic ID tokens
        |  built by hierarchical k-means
        |  Codebook: 4 / 64 / 256
        |
[BOS, c1^1, c2^1+4, c3^1+68, c1^2, c2^2+4, c3^2+68, ...]
        |
        v
GPT-2 decoder-only model (trained from scratch)
        |
        v
Beam search generates 3 new tokens
        |
        v
(c1*, c2*, c3*)
        |
        +- exact match -> item
        +- otherwise -> Hamming nearest neighbor
        v
Recall@K / NDCG@K
```

### Token Vocabulary

Current token layout in code:

| Range | Meaning |
|-------|---------|
| 0-3 | Level-1 codes (`c1`) |
| 4-67 | Level-2 codes (`c2 + 4`) |
| 68-323 | Level-3 codes (`c3 + 68`) |
| 324 | `[BOS]` |
| 325 | `[EOS]` |
| 326 | `[PAD]` |

Although `EOS` is defined in the vocabulary, the current sequence builder does not insert item separators yet.

### Semantic ID Construction

The current build script in [embedding/build_semantic_ids.py](./embedding/build_semantic_ids.py):

- loads `embedding/item_embeddings_raw.npy`
- uses hierarchical MiniBatchKMeans with `K_LEVELS = [4, 64, 256]`
- performs L1 global clustering
- performs L2 clustering inside each L1 cluster
- assigns L3 by distance-to-centroid ranking to guarantee zero collisions
- saves `embedding/semantic_ids.npy`

The reason for `K2 = 64` in the current code is practical: with plain k-means, a larger L2 codebook helps keep each `(c1, c2)` bucket within the `K3 = 256` capacity.

## Results

Recorded results from completed runs:

| Run | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Notes |
|-----|----------|--------|-----------|---------|-------|
| SASRec (TIGER paper) | 0.0387 | 0.0249 | 0.0605 | 0.0318 | reference |
| TIGER (paper) | 0.0454 | 0.0321 | 0.0648 | 0.0384 | reference |
| Generative + qwen2:7b + `4/64/256` | 0.0197 | 0.0122 | 0.0322 | 0.0162 | completed |
| Generative + Random ID ablation | 0.0025 | 0.0016 | 0.0042 | 0.0021 | completed |
| SASRec baseline (latest recorded run) | 0.0222 | 0.0114 | 0.0404 | 0.0172 | completed |
| Generative + default nomic pipeline | TBD | TBD | TBD | TBD | next active direction |

Important note:

- The generative code now defaults to `nomic-embed-text` during embedding extraction.
- The completed generative numbers above come from the earlier `qwen2:7b` run recorded in [Progress.md](./Progress.md).
- `Progress.md` may therefore contain experiment settings that are newer or more specific than the current default script values.

## Dataset

Amazon Beauty 5-core from [SNAP](http://snap.stanford.edu/data/amazon/).

| Stat | Value |
|------|-------|
| Users | 22,363 |
| Items | 12,101 |
| Interactions | 198,502 |
| Split | Leave-one-out |
| Validation | sequence up to second-to-last item |
| Test | sequence up to last item |
| Evaluation | all-rank Recall@K / NDCG@K |

## Repository Guide

### Main files

- [data/data_process.py](./data/data_process.py): preprocess Amazon Beauty and build `beauty_data.pkl`
- [embedding/extract_embeddings.py](./embedding/extract_embeddings.py): extract item embeddings with Ollama
- [embedding/build_semantic_ids.py](./embedding/build_semantic_ids.py): build hierarchical semantic IDs
- [model/tokenizer.py](./model/tokenizer.py): token offsets, vocab layout, sequence encoding
- [model/generative_rec.py](./model/generative_rec.py): GPT-2 recommender model
- [model/inference.py](./model/inference.py): beam search and Hamming fallback
- [train.py](./train.py): train the generative model
- [evaluate.py](./evaluate.py): all-rank evaluation for the generative model
- [baseline/sasrec_train.py](./baseline/sasrec_train.py): SASRec baseline training and evaluation
- [Progress.md](./Progress.md): experiment log
- [improve_generative_prompt.md](./improve_generative_prompt.md): planned next-step improvements

### Artifact naming notes

Some files in the repository are archived experiment artifacts rather than the paths used by the default scripts:

- `embedding/item_embeddings_raw_qwen.npy`
- `embedding/item_embeddings_raw_nomic.npy`
- `checkpoints/llm_id_model.pt`
- `checkpoints/random_id_model.pt`
- `checkpoints/sasrec_best.pt`

By contrast, the current default script flow expects:

- `embedding/item_embeddings_raw.npy`
- `embedding/semantic_ids.npy`
- `checkpoints/best_model.pt`

## Quickstart

```bash
pip install -r requirements.txt

# Install Ollama and pull the current default embedding model
brew install ollama
ollama pull nomic-embed-text

# 1. Preprocess raw data
python data/data_process.py

# 2. Extract item embeddings
# Current default: MODEL = 'nomic-embed-text'
# Output: embedding/item_embeddings_raw.npy
python embedding/extract_embeddings.py

# 3. Build Semantic IDs with the current 4/64/256 pipeline
# Output: embedding/semantic_ids.npy
python embedding/build_semantic_ids.py

# 4. Train the generative model
python train.py

# 5. Evaluate the generative model
python evaluate.py

# 6. Train the SASRec baseline
python baseline/sasrec_train.py
```

## Current Limitations

- The generative model is still behind the latest recorded SASRec baseline.
- Current beam search is unconstrained and wastes probability mass on invalid token ranges.
- Current training selects checkpoints by `val_loss`, not by recommendation quality.
- The current tokenizer still uses the older `4/64/256` setup with no item-level `EOS` separators.
- The repository still contains a mix of active paths and archived artifact names from earlier runs.

## References

- [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
