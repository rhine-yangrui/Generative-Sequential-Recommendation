# CLAUDE.md

This file provides working guidance for agents editing this repository.

## Project Summary

This project is a simplified reproduction of TIGER-style generative sequential recommendation:

- items are mapped to hierarchical Semantic IDs
- a GPT-2 decoder predicts the next item's Semantic ID autoregressively
- evaluation uses all-rank Recall@K / NDCG@K on Amazon Beauty

The project thesis is tested mainly through the ablation:

- Semantic ID vs random ID
- same autoregressive model
- same evaluation protocol

## Documentation Rules

Before changing docs or code, treat the repository as having three distinct states:

- `README.md`: should describe the current codebase accurately
- `Progress.md`: is the experiment log and may include runs produced with settings not reflected in the current default scripts
- `improve_generative_prompt.md`: is a planned next-step change list, not the current implementation

Do not describe planned improvements as already implemented.

## Current Code State

### Active generative path

- `model/tokenizer.py`
  - `K_LEVELS = [4, 64, 256]`
  - token layout is `0-3`, `4-67`, `68-323`, then `BOS/EOS/PAD`
  - `seq_to_tokens()` uses `BOS` only and does not insert per-item `EOS`

- `embedding/extract_embeddings.py`
  - current default `MODEL = 'nomic-embed-text'`
  - writes to `embedding/item_embeddings_raw.npy`

- `embedding/build_semantic_ids.py`
  - reads `embedding/item_embeddings_raw.npy`
  - builds `embedding/semantic_ids.npy`
  - hierarchical MiniBatchKMeans with `4/64/256`
  - L3 uses centroid-distance ranking for zero-collision assignment

- `train.py`
  - loads `embedding/semantic_ids.npy`
  - trains GPT-2 from scratch
  - early stopping is based on `val_loss`
  - saves `checkpoints/best_model.pt`

- `model/inference.py`
  - uses unconstrained `model.generate()`
  - generates 3 new tokens
  - maps invalid or unmatched outputs through exact match or Hamming nearest neighbor

- `evaluate.py`
  - expects `checkpoints/best_model.pt`
  - evaluates with all-rank Recall@K / NDCG@K

### Baseline path

- `baseline/sasrec_train.py`
  - trains SASRec with all-rank evaluation
  - saves `checkpoints/sasrec_best.pt`

Note:

- `Progress.md` contains the latest recorded SASRec result.
- The default hyperparameters visible in `baseline/sasrec_train.py` may not exactly match that recorded best run.

## Files To Know

- `data/data_process.py`: build `beauty_data.pkl`
- `embedding/extract_embeddings.py`: Ollama embedding extraction
- `embedding/build_semantic_ids.py`: semantic ID construction
- `model/tokenizer.py`: tokenization and ID conversion
- `model/generative_rec.py`: GPT-2 model builder
- `model/inference.py`: beam search and item lookup
- `train.py`: generative training entry point
- `evaluate.py`: generative evaluation entry point
- `baseline/sasrec_train.py`: SASRec baseline
- `Progress.md`: experiment record
- `improve_generative_prompt.md`: planned improvements

## Artifact Naming Notes

The repository contains both active-path filenames and archived experimental artifacts.

Active-path filenames used by the current scripts:

- `embedding/item_embeddings_raw.npy`
- `embedding/semantic_ids.npy`
- `checkpoints/best_model.pt`

Archived files currently present in the repo:

- `embedding/item_embeddings_raw_qwen.npy`
- `embedding/item_embeddings_raw_nomic.npy`
- `checkpoints/llm_id_model.pt`
- `checkpoints/random_id_model.pt`
- `checkpoints/sasrec_best.pt`

Be careful not to confuse archived artifacts with the paths hardcoded in the active scripts.

## Current Target

Short term:

- keep docs aligned with the current code
- preserve `Progress.md` as the experiment ledger
- treat `improve_generative_prompt.md` as the next implementation plan

Next implementation phase:

- switch the active generative path to the planned nomic-based improvement flow
- only after code changes land should docs be updated to describe `4/16/256`, EOS separators, constrained beam search, and Recall@10-based early stopping as current behavior
