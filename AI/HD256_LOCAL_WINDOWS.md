# SM100 head_dim 256: local (sliding) windows

Forward local windows in the dedicated hd256 2CTA kernel (`sm100_hd256_2cta_fmha_forward.py`), built for
DiffusionGemma's sliding-window layers (head_dim 256, window W = 1024, 16 q / 8 kv heads) served by
FluxServe from a paged KV cache. Backward still rejects local windows for hd256.

## What changed

- The kernel already threaded windows through `BlockInfo` (KV block ranges) and
  `Sm100FusedMask.apply_mask_via_causal_local` (masking); the `__call__` asserts that rejected
  `is_local` / `window_size_*` were the only blocker. Window sizes are converted to runtime `Int32`
  before launch (as in `FlashAttentionForwardSm100`), because the compile key only records whether
  each side is set: one compiled kernel serves every window value.
- Paged KV + an empty local range (a tile whose rows see no keys, e.g. far-left/negative windows):
  every role still runs one fully masked iteration, but the load warp's page-table lookup at
  `n_block_min` could land past the sequence's pages (an unused entry, or out of bounds). It now
  loads block `n_block_max - 1` (or 0) instead; the scores are masked either way, and V rows of that
  page are finite.
- `window_left_anchor="first_query"` (`flash_attn_func` / `flash_attn_varlen_func`, forward only,
  hd256 only): every row of a sequence uses the first query row's left edge,
  `col >= seqlen_k - seqlen_q - window_size[0]`, instead of `row + seqlen_k - seqlen_q - window_size[0]`.
  It is a constexpr `window_left_from_first_query` in `BlockInfo` (n_block_min and the left-mask
  boundary use row 0) and in the fused mask, and part of the compile key. With the default `"query"`
  the new branches are compile-time dead, and the paged clamp only exists in local variants, so the
  previously supported (non-local) hd256 variants trace the same code (not checked at SASS level).

## Why "first_query"

DiffusionGemma's denoise step attends from a 256-token canvas to the cached prefix plus the canvas,
bidirectionally. The Hugging Face reference decoder (transformers 5.12, eager/sdpa) concatenates the
encoder's sliding-window cache (the last W - 1 = 1023 prefix tokens) with the canvas, so every canvas
row sees the same prefix slice `[P - 1023, P)`: a per-row window. FluxServe used a per-query window
`|q - k| < W`, which differs once the prefix P >= W - 256 + 1 = 769.

Measured on google/diffusiongemma-26B-A4B-it, fp32, 24 pg19 excerpts, P in {1024, 1536, 2048, 3072}
(`agent_space/gemma_window_ab.py`): the HF default path matches per-row(1023) (argmax agreement
94-99%; a semantically identical control differs by KL <= 7e-4), while per-query differs from it by
KL 0.06-0.09 with 75-85% argmax agreement, and its canvas NLL is never significantly lower (higher in
7 of 8 cells, significantly at P = 1024). In bf16 the run-to-run noise of this MoE model (expert
routing flips) is as large as the per-query/per-row gap, so such comparisons need fp32.

## FluxServe mapping (bottom-right aligned, paged varlen, seqused_k)

| phase | causal | window_size | window_left_anchor |
|---|---|---|---|
| prefill / commit | True | (W - 1, None) | query |
| denoise, per_query | False | (W - 1, W - 1) | query |
| denoise, per_row | False | (L, W - 1), L = W - 1 by default | first_query |

`(W - 1, W - 1)` equals `(W - 1, None)` bit for bit on canvas shapes (256 <= W) and masks fewer blocks
(the right edge never falls inside the KV range).

## Tests

- `tests/cute/test_flash_attn.py`: the hd256 local cases are no longer skipped (forward; backward
  stays gated). With `-k "hd256 or -256-"` plus d=128 local slices: 3086 passed, two-pass clean.
- `tests/cute/test_flash_attn_hd256_local.py` (29, two-pass clean): FluxServe-shaped paged varlen
  calls with page 128, NaN in unreferenced pages and finite junk past seqused_k: ragged causal prefill,
  canvas commit and denoise with both anchors over prefixes 0..5000 around 768/769 and 1023/1024,
  num_splits 1 and 3, bitwise equivalences, runtime windows, empty ranges, CUDA graph replay with
  in-place seqused_k / page_table updates, and the guards.
- The varlen local slice of `test_flash_attn_varlen_output` passes (189) in a single pass; its
  two-pass run reports fake/real compile-key drift, which these varlen tests already have.
