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
- `improve_generative_prompt.md`: outstanding TODOs (RQ-VAE capacity ceiling, c4 vocab shrink)

Do not describe planned improvements as already implemented.

## Current Code State

### Active generative path

- `model/tokenizer.py`
  - `K_LEVELS = [4, 16, 256, 512]` (4-token Semantic ID; first 3 are RQ-VAE codes, c4 is collision resolution)
  - `seq_to_tokens()` inserts `EOS` between items

- `embedding/extract_embeddings.py`
  - default `MODEL = 'nomic-embed-text'`, writes `embedding/item_embeddings_raw.npy`

- `embedding/rqvae.py`
  - trains RQ-VAE on nomic embeddings, saves `checkpoints/rqvae_best.pt`
  - 3 levels `[4,16,256]`, log-domain Sinkhorn, encoder warmup + k-means init + joint training

- `embedding/generate_rqvae_ids.py`
  - loads best checkpoint, argmin quantization, appends c4 for collisions
  - writes `embedding/semantic_ids_rqvae.npy`

- `train.py`
  - loads `embedding/semantic_ids_rqvae.npy`
  - trains GPT-2 from scratch, early stopping on validation Recall@10
  - saves `checkpoints/best_model.pt`

- `model/inference.py`
  - constrained beam search via `LevelConstrainedLogitsProcessor`
  - generates `len(K_LEVELS)` tokens, falls back to Hamming nearest neighbor

- `evaluate.py`
  - expects `checkpoints/best_model.pt`, all-rank Recall@K / NDCG@K

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
- `embedding/rqvae.py`: RQ-VAE training
- `embedding/generate_rqvae_ids.py`: RQ-VAE inference + collision resolution
- `embedding/build_semantic_ids.py`: legacy k-means semantic ID builder (not on active path)
- `model/tokenizer.py`: tokenization and ID conversion
- `model/generative_rec.py`: GPT-2 model builder
- `model/inference.py`: constrained beam search and item lookup
- `train.py`: generative training entry point
- `evaluate.py`: generative evaluation entry point
- `baseline/sasrec_train.py`: SASRec baseline
- `Progress.md`: experiment record
- `improve_generative_prompt.md`: outstanding TODOs

## Artifact Naming Notes

The repository contains both active-path filenames and archived experimental artifacts.

Active-path filenames used by the current scripts:

- `embedding/item_embeddings_raw.npy`
- `embedding/semantic_ids_rqvae.npy`
- `checkpoints/rqvae_best.pt`
- `checkpoints/best_model.pt`

Archived files currently present in the repo:

- `embedding/item_embeddings_raw_qwen.npy`
- `embedding/item_embeddings_raw_nomic.npy`
- `embedding/semantic_ids.npy` (legacy k-means output)
- `checkpoints/llm_id_model.pt`
- `checkpoints/random_id_model.pt`
- `checkpoints/sasrec_best.pt`

Be careful not to confuse archived artifacts with the paths hardcoded in the active scripts.

## Current Target

- keep docs aligned with the current code
- preserve `Progress.md` as the experiment ledger
- run downstream `train.py` / `evaluate.py` on the current RQ-VAE IDs (unique_rate ≈ 41%)
- if downstream Recall@10 underperforms SASRec, switch RQ-VAE to `K_LEVELS=[256,256,256]` (see `improve_generative_prompt.md`)
