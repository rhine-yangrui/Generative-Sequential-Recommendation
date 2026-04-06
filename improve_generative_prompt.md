# Outstanding TODO

The 4-token RQ-VAE pipeline is implemented and downstream training is running.
Two known issues remain, in priority order:

## 1. RQ-VAE codebook capacity ceiling

Current `K_LEVELS = [4, 16, 256]` only yields **unique_rate ≈ 41%** on Beauty's
12,101 items (max collision group 24). All three codebooks are 100% used —
this is a capacity ceiling, not codebook collapse:

- joint slots = `4 × 16 × 256 = 16,384`
- residual quantization is non-uniform, so the first level's 4 cells
  cannot evenly distribute 12k items across the deeper `16 × 256` buckets
- the c4 collision token (vocab 512) is currently doing real work to make
  the final 4-tuple unique

**Fix if downstream Recall@10 underperforms SASRec (0.0573):**
switch to `K_LEVELS = [256, 256, 256]`, aligned with TIGER paper / the
XiaoLongtaoo reproduction. This requires updating:

- `embedding/rqvae.py` `K_LEVELS`
- `embedding/generate_rqvae_ids.py` `K_LEVELS`
- `model/tokenizer.py` `K_LEVELS = [256, 256, 256, COLLISION_K]`

## 2. Shrink c4 vocab once RQ-VAE is acceptable

Current `c4 = 512` is wasted vocab (only 0..23 are used after collision
resolution). Once RQ-VAE quality is locked in, shrink it to
`max_collision_group + buffer` (e.g. 32 or 64) to reduce GPT-2 vocab size.
Touch the same three files as above.
