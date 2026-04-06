# Next-Step Handoff: RQ-VAE-First TIGER Direction

This document replaces the old improvement checklist.
It is intended for the next CLI agent working on this repository.

The current direction is:

- keep the project on the `nomic` embedding path
- use `RQ-VAE` for Semantic ID generation
- keep the downstream generative model improvements that were already implemented
- do **not** spend more time on the `k-means + 4/16/256` path for now

## Current State

### Already changed in code

The downstream generative pipeline has already been upgraded:

- `model/tokenizer.py`
  - now uses a 4-token Semantic ID layout
  - current `K_LEVELS = [4, 16, 256, 512]`
  - first 3 levels are the learned RQ-VAE codes
  - the 4th level is reserved for collision resolution
  - `seq_to_tokens()` inserts `EOS` between items

- `model/inference.py`
  - constrained beam search is implemented via `LevelConstrainedLogitsProcessor`
  - generation length is tied to `len(K_LEVELS)`

- `train.py`
  - early stopping is based on validation `Recall@10`
  - active semantic ID file is `embedding/semantic_ids_rqvae.npy`

- `evaluate.py`
  - also reads `embedding/semantic_ids_rqvae.npy`

- `embedding/rqvae.py`
  - exists
  - trains an RQ-VAE on top of `embedding/item_embeddings_raw_nomic.npy`
  - includes warmup seed search and collision handling

### Important commit boundary

There is a clean commit before the latest uncommitted debugging work:

- commit: `fa28343`
- message: `Add RQ-VAE semantic ID pipeline and generative upgrades`

After that commit, there are additional local modifications in:

- `embedding/rqvae.py`
- `model/generative_rec.py`
- `model/inference.py`
- `model/tokenizer.py`
- `train.py`

These changes are **not** committed yet.

### Current blocker

The project is currently blocked at the RQ-VAE stage.

Observed failure modes:

- codebook usage often collapses, especially at the deepest level
- some trajectories finish training but produce large collision groups
- even after adding a 4th collision token, the current training recipe is still unstable

What was observed locally:

- raw `nomic` embeddings made collapse worse
- L2-normalizing the input embeddings helps
- some seeds on `mps` give materially better warmup trajectories than others
- however, the current training recipe is still not reliable enough to treat as solved

## Why The Current RQ-VAE Version Is Not Good Enough

The issue does not appear to be just one bug.
It is more likely that the current training recipe is too naive.

The current script still differs from a stronger TIGER-style reproduction in several practical ways:

- training and code generation are coupled in one script
- model selection is not based on `collision_rate`
- optimizer and scheduler recipe are not aligned with the stronger public reproduction
- warmup seed search is only a local stabilization hack, not a principled training setup

## External Reference To Borrow From

Use this repository as the main reference for the next implementation pass:

- `https://github.com/XiaoLongtaoo/TIGER`

Important note:

- do **not** blindly copy its default `RQ-VAE` codebook sizes
- that reproduction defaults to a different setting in places
- we still want to stay on the `Amazon Beauty + 4/16/256 semantic hierarchy` direction

What to borrow from that repo:

- separate `RQ-VAE` training from discrete code generation
- select checkpoints using `collision_rate`
- use a more stable optimizer / scheduler recipe
- follow its post-processing flow for duplicate resolution

## Recommended Next Tasks

Do these in order.

### 1. Refactor `embedding/rqvae.py`

Split the current script into two conceptual stages:

- train RQ-VAE
- generate Semantic IDs from the best checkpoint

Recommended structure:

- keep `embedding/rqvae.py` as the training entry point
- optionally add a second script such as `embedding/generate_rqvae_ids.py`

The training stage should:

- train the encoder / decoder / quantizer
- periodically compute:
  - reconstruction loss
  - codebook usage per level
  - collision rate on generated codes
- save the best checkpoint using collision-oriented criteria

The generation stage should:

- load the best checkpoint
- generate the first 3 learned codes
- append a 4th collision-resolution code only when needed
- save `embedding/semantic_ids_rqvae.npy`

### 2. Replace the current training recipe with a stronger one

Borrow the training flow from the XiaoLongtaoo reproduction instead of the current ad hoc setup.

Specifically investigate:

- `AdamW` instead of the current optimizer choice
- scheduler + warmup
- longer training with explicit eval intervals
- checkpoint selection by `collision_rate`

Do not assume the current `Adagrad + warmup seed search` path is the right final solution.
It was only introduced to diagnose instability.

### 3. Keep the `4/16/256` semantic hierarchy

Current intended semantic structure:

- level 1: `4`
- level 2: `16`
- level 3: `256`

The 4th token is not part of the semantic hierarchy itself.
It is only there to resolve collisions so the downstream generative model can map generated IDs back to unique items.

### 4. Re-check the tokenizer / downstream assumptions after the RQ-VAE refactor

Because the downstream pipeline now assumes 4-token IDs, verify these files together after changing RQ-VAE:

- `model/tokenizer.py`
- `model/generative_rec.py`
- `model/inference.py`
- `train.py`
- `evaluate.py`

Current assumption:

- each target item contributes 4 code tokens during prediction
- history items are encoded as `4 code tokens + EOS`

Make sure the final RQ-VAE output format matches that assumption exactly.

### 5. Only then rerun the full pipeline

Run order should be:

```bash
python embedding/rqvae.py
# or:
# python embedding/rqvae.py
# python embedding/generate_rqvae_ids.py

python train.py
python evaluate.py
```

Do not start another long `train.py` run until `embedding/semantic_ids_rqvae.npy` is produced successfully and passes a basic sanity check.

## Sanity Checks The Next Agent Should Perform

Before running GPT training, verify all of the following:

- `embedding/semantic_ids_rqvae.npy` exists
- it contains all `12,101` items
- every item has a unique final ID
- the first three levels follow the intended `4/16/256` ranges
- the 4th collision token stays within the configured capacity
- collision rate before adding the 4th token is recorded

Suggested quick checks:

- print total item count
- print unique final ID count
- print max collision group size before `c4`
- print per-level code usage

## What Not To Do Next

- do not spend more time trying to rescue `embedding/build_semantic_ids_nomic.py`
- do not update showcase docs for interview presentation yet
- do not assume the current uncommitted `rqvae.py` is correct just because it runs
- do not revert the downstream 4-token generative changes unless the new RQ-VAE design truly requires it

## Suggested Acceptance Criteria

The next agent should consider this phase complete only when:

1. `embedding/semantic_ids_rqvae.npy` is produced successfully
2. the file contains a unique final ID for every item
3. `train.py` runs end to end using that file
4. `evaluate.py` runs end to end on the trained checkpoint
5. the experiment is stable enough that rerunning does not depend on lucky seed selection

## Short Summary For The Next Agent

The repository already moved the downstream generative model to a TIGER-like 4-token ID pipeline.
The remaining unsolved problem is the **RQ-VAE training / code-generation recipe**.
Use the XiaoLongtaoo TIGER reproduction as the implementation reference for training flow and collision-based code generation, but keep the local project on the `4/16/256` semantic hierarchy rather than blindly copying its default hyperparameters.
