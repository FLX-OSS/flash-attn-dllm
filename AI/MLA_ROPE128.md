# MLA with a 128-wide rope part (DiffusionGemma full-attention layers)

FA4's MLA forward (`qv`) computes `S = scale * (q_pe . k_pe + qv . V)`, `O = P V` with a latent V of 512.
It was written for DeepSeek's 64-wide rope part; this adds 128 on the SM100 2CTA kernel, for
DiffusionGemma's full-attention layers (16 q / 2 kv heads, head_dim 512).

## Why those layers are MLA

K and V come from one projection u: `V = u / rms(u)` (v_norm, no weight) and `K = RoPE(k_norm(u))`.
In this checkpoint every full layer's `k_norm` weight is a single scalar c (0.0601 .. 0.0635), so
`K = c * RoPE(V)`; RoPE rotates 128 of the 512 dims (proportional RoPE, pairs (j, j + 256) for j < 64)
and is the identity elsewhere. Hence, exactly:

    S = c * (q_R . RoPE(V)_R + q_notR . V_notR)

i.e. MLA with `q_pe = q_R`, `k_pe = RoPE(V)_R` (128), latent V (512), `qv = q` with the rope dims
zeroed, and `softmax_scale = c`. Checked on the real layer weights (`agent_space/gemma_mla_equiv.py`):
fp64 identity to ~1e-15; in bf16 the score error vs fp64 equals the original formulation's
(c folded into q instead would add one rounding: +20% rms). The cache holds k_pe + V = 640 values per
token per kv head instead of 1024.

## Kernel changes (`flash_fwd_mla_sm100.py`, `interface.py`)

- `FlashAttentionMLAForwardSm100(hdim=128)`: the interface passes the q/k head dim. With 4 V stages
  (32 KB each per CTA) a 128-wide rope part needs ~240 KB of smem, so it keeps 3.
- The O epilogue staged O in the V ring's "last Vt slots 2, 3", which holds only with 4 stages (each
  128-key block takes V0, V1, Vt0, Vt1). With fewer stages O now goes through the Qv buffer (the
  existing `overlap_sO_sV = False` path): `overlap_sO_sV = num_stages_V == 4`.
- Interface: `(128, 512)` passes `_validate_head_dims` only with `qv`; a 128-wide rope part is
  routed to the 2CTA kernel (the 1CTA kernels assume 64), dense or TMA-paged (page 128), no topk
  gather, no fp8 / descales, no split.
- Compile key: the MLA kernels bake the kv-head count into their packed-GQA layouts
  (`pack_gqa_layout(..., self.nheads_kv, ...)`) but the key had only heads per kv head, so 8/1 and
  16/2 heads shared one cached kernel and one of them got wrong output (garbage heads). The key now
  includes `num_head_kv` for MLA calls. Pre-existing upstream (MLA there always had one kv head).

## Debugging record (`agent_space/ledger_mla_rope128_iae.md`)

1. Illegal memory access at hdim 128. Predicted and confirmed that 3 V stages alone reproduce it at
   hdim 64 (4 stages pass); the `sO_idx = 2 + split` staging was the cause. Fixed as above.
2. Dense 16/2-head cases wrong only in the two-pass flow. LSE (T5) and hdim (T3) theories were
   falsified by cache-off probes; a cache-entry count showed 8/1 and 16/2 shared one compile key
   (the order-dependent direction I first predicted was wrong; why the 8/1-built kernel is right for
   16/2 is still open). Fixed by the compile key; both predicted checks (separate entries, two-pass
   pass) confirmed.
3. 60 fake-mode failures in `test_flash_attn_mla_paged` were an artifact of editing the kernel file
   while its compile pass ran (the DSL re-reads source at compile time); the same tests pass two-pass
   on the unchanged tree. The sparse-MLA backward dV tolerance failure
   (`test_flash_attn_mla_absorbed[...-True-True-64-...-True-mqa...]`) reproduces on clean a02ec08.

## Tests

- `tests/cute/test_flash_attn_mla_rope128.py` (50, two-pass clean): dense rope128 MLA for 16/2, 8/1,
  16/16 heads, causal and not, ragged shapes, determinism; DiffusionGemma's paged varlen calls
  (page 128, NaN in unreferenced pages, finite junk past seqused_k: ragged causal prefill, 256-query
  denoise and commit over prefixes 0..5000); and the full-layer rewrite itself against
  `q . RoPE(c * V)` in fp64 on GPU.
- Regression: dense MLA (absorbed / 1CTA dense / dispatch) 1071 passed + 1 pre-existing failure;
  `test_flash_attn_mla_paged` 94 two-pass; 1CTA dense MLA (`FLASH_ATTENTION_MLA_1CTA=1`) 425 two-pass.

## Speed (B200, per full layer, vs FlashInfer fa2 with 512-wide K and V)

| shape | FlashInfer | FA4 MLA | |
|---|---|---|---|
| denoise B=1, 2K prefix | 126 us | 32 us | 3.9x |
| denoise B=4, 8K prefix | 1548 us | 232 us | 6.7x |
| denoise B=16, 32K prefix | 24.2 ms | 4.5 ms | 5.4x |
| prefill causal, 32K prompt | 94.1 ms | 17.3 ms | 5.4x |
