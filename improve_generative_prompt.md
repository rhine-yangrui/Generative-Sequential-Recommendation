# Legacy TODO

This file used to be the active improvement plan. The 4-token RQ-VAE pipeline
described here has been implemented (see `embedding/rqvae.py`,
`embedding/generate_rqvae_ids.py`, `model/tokenizer.py` `K_LEVELS=[4,16,256,512]`,
constrained beam search, Recall@10 early stopping).

Outstanding work — see [RQVAE_Analysis.md](./RQVAE_Analysis.md) for full context:

- Codebook capacity: `[4,16,256]` only yields ~41% unique_rate on Beauty
  (12,101 items vs 16,384 joint slots). Consider switching to `[256,256,256]`
  if downstream Recall@10 is unsatisfactory.
- After RQ-VAE quality is acceptable, shrink `c4` (currently 512) down to
  `max_collision_group + buffer` to reduce GPT vocab size.
