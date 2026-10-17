# Fixed-shift softmax forward (SM100)

`softmax_mode="fixed_shift"` ports the dense-forward idea of Stashuk, Yu, Shah, *Efficient
FlashAttention on Blackwell via Fixed-Shift Softmax and Persistent Scheduling* (arXiv 2610.02229)
into `FlashAttentionForwardSm100`:

- **Fixed shift.** Every row uses the constant shift `softmax_shift` instead of the running row max,
  so the output accumulator is never rescaled and the row max is never computed.
- **Packed BF16 exp2.** Every score is exponentiated with a 4-instruction packed evaluator that
  writes BF16 weights directly (no `MUFU.EX2`, no fp32 exp, no separate BF16 convert).
- **FP32 row sums of the rounded weights.** The denominator is the fp32 sum of exactly the BF16
  weights that `P @ V` consumes (`add.rn.f32.bf16` = `FHADD.BF16`). The paper uses BF16 tree sums.

Forward only. Inference such as dLLM sampling is the target; training keeps the online softmax.

## Usage

```python
out = flash_attn_func(q, k, v, softmax_mode="fixed_shift")                  # shift 0
out, lse = flash_attn_func(q, k, v, softmax_mode="fixed_shift", softmax_shift=4.0, return_lse=True)
```

- **`FLASH_ATTENTION_SOFTMAX_MODE=fixed_shift`** (read per call) turns the mode on for eligible calls.
  Ineligible calls fall back to the online softmax silently. An explicit kwarg that cannot be
  honored raises `ValueError` instead.
- **Eligible calls:** SM100/SM110, bf16, `FlashAttentionForwardSm100` (not hd256 / MLA / gathered KV),
  non-causal and non-local, no score_mod / mask_mod / softcap / learnable_sink / block sparsity,
  and no input requiring grad. Varlen, paged KV (TMA and non-TMA pages) and split-KV are supported
  (each split's LSE is shift + ln of its row sum, so the combine weights stay consistent).
- **`FLASH_ATTENTION_FIXED_SHIFT_HARDEN=1`** (not public yet) adds one `max.bf16x2` per pair that
  removes the evaluator's NaN band (see below).
- **LSE** is `shift + ln(row_sum)`, i.e. the log of the denominator actually used.

## The evaluator

`utils.packed_exp2_bf16x2(s0, s1, A, B)` with `A = 128·scale_log2` and `B = 1.5·2^23 + 127·128 + OFF − 128·shift_log2`:

```
t  = fma.rn.f32x2(s, A, B)        # 1.5·2^23 => fp32 spacing 1: low 16 bits = round(128 y + 127·128 + OFF)
R  = prmt.b32 t0, t1, 0x5410      # Schraudolph BF16 bit patterns 2^n (1 + f), one per lane
R2 = add.s32 R, 0x003F003F        # K = 63: the mantissa carry into the exponent puts a kink at f = 1 − 63/128
P  = fma.rn.relu.bf16x2 R, κ, R2  # κ = 1.40625: κR + R2 is a 2-piece chord of 2^f; relu zeroes underflow
```

The paper names these instructions but not the constants. This reconstruction reproduces its §5.2
numbers (`benchmarks/tune_packed_exp.py`, 4M samples on the 21-bit lattice, seed 20260915):

| evaluator | ratio to 2^y | max rel | mean rel |
|---|---|---|---|
| ideal bf16 rounding (paper: 3.89e-3 / 1.41e-3) | [0.9961, 1.0039] | 3.89e-3 | 1.41e-3 |
| K=63, κ=0x3FB4, OFF=−188 (paper: [1.0440, 1.0687], 6.87e-2 / 5.79e-2) | [1.0430, 1.0684] | 6.84e-2 | 5.77e-2 |
| **deployed: OFF=−198 (gain centred)** | **[0.9880, 1.0120]** | **1.21e-2** | **4.54e-3** |

Centring the gain does not change the normalized output (it is a common factor), but it makes the
weights, and hence the LSE, nearly unbiased.

SASS on sm_100a, per pair: `FFMA2`, `PRMT`, `IADD3`, `HFMA2.BF16_V2.RELU` (κ is an immediate), plus
two `FHADD.BF16` for the fp32 row sum. With hardening there is one extra `HMNMX2.BF16_V2`. The GPU
matches the torch model `testing.packed_exp2_bf16_ref` bit for bit (`tests/cute/test_packed_exp.py`).

## Range contract (unchecked)

With `y = s·scale·log2e − shift·log2e`:

| y | weight |
|---|---|
| (−124.4, 128) | within ±1.2 % of 2^y |
| [−126.9, −125.5] | **NaN** without hardening; the whole row becomes NaN |
| (−382.9, −126.9) and (−125.5, −124.4) | 0 or tiny (correct underflow) |
| < −382.9 | **wraps around**: garbage |
| [128, 130.5] | inf / NaN |
| ≥ 130.6 (≥ 129.6 hardened) | **silently 0** |

The row sum and the fp32 `P @ V` accumulator must also stay finite. In natural-log units of
`z = softmax_scale·q·k − softmax_shift` this gives **−87 < z < 88.7 − ln N − ln max|V|**, about
±70 for N = 32K and |V| ≤ 1024.

Masked scores use the finite sentinel `utils.packed_exp2_mask_value` (y = −256 → exactly 0), not
−inf: −inf would give a 5.8e-39 weight and make fully masked rows divide by a tiny sum.

## Kernel changes (`flash_fwd_sm100.py`, all behind `const_expr(self.use_fixed_shift)`)

The default kernels compile to byte-identical SASS.

- **softmax_step:**
  - no row max, `acc_scale` or `sScale` publish, and no scale-subtract;
  - `SoftmaxSm100.apply_packed_exp2` writes the BF16 weights into the P registers;
  - `update_row_sum_packed` sums them in fp32 with 8 independent chains;
  - the per-block stats arrive moves right after the S wait.
- **correction_loop:** no `sScale` read, vote or O rescale; LSE uses the shift.
- **Barrier protocol unchanged.** The named barrier plus the `pipeline_sm_stats` credits are what
  stop a softmax warp from arriving twice on its 64-thread barrier before the correction warp syncs.
  Only the work around them is removed. Dropping the per-block handshake entirely (correction warps
  waiting on the S-full barrier instead) is a possible follow-up if profiling shows it matters.

## Accuracy

`tests/cute/test_flash_attn_fixed_shift.py`, Gaussian inputs, d ∈ {64, 96, 128, 192/128}:

- **vs the fixed-shift model** (`testing.attention_fixed_shift_ref`): the kernel is within bf16
  output rounding.
- **vs exact fp32 attention:** error ≈ the model's own, about 2× the online kernel's (the paper
  reports 5.2e-3 vs 2.35e-3 relative-ℓ2 for O).
- **LSE:** within 1e-3 of the model for ordinary inputs and ≤ 1.3e-2 of exact. On peaked rows a
  ~1e-6 tensor-core score difference can move one weight by one 1/128-binade step, giving LSE
  differences up to ~1e-2.

## Performance

Measured on an idle B200 at 1000 W, 2026-10-17. Grid: FA4 variable-batch, B·N = 32K, dense, bf16,
H = 16 (d=128) or 32 (d=64).

**Paper protocol** (`agent_space/bench_protocol.py`): FA4's original timer (triton `do_bench`,
warmup 5, rep 10, after 1 s idle, as in `benchmarks/benchmark_attn.py`), median of 6 repetitions
with alternating order. Online FA4 here is within 2.4 % of the paper's FA4 reference.

| d=128, TFLOP/s | 1K | 2K | 4K | 8K | 16K | 32K | geomean |
|---|---|---|---|---|---|---|---|
| online | 1325 | 1425 | 1483 | 1520 | 1540 | 1552 | 1472 |
| fixed-shift | 1466 | 1591 | 1665 | 1712 | 1741 | 1762 | **1653 (+12.3 %)** |
| fixed-shift + harden | 1434 | 1563 | 1642 | 1693 | 1722 | 1745 | 1629 (+10.7 %) |

For comparison, the paper reports its full TLX kernel at 1705 TF geomean, +13.0 % over FA4.

| d=64, TFLOP/s | 1K | 2K | 4K | 8K | 16K | 32K | geomean |
|---|---|---|---|---|---|---|---|
| online | 834 | 899 | 937 | 957 | 967 | 961 | 925 |
| fixed-shift | 1160 | 1248 | 1294 | 1311 | 1326 | 1312 | **1274 (+37.8 %)** |
| fixed-shift + harden | 1069 | 1144 | 1178 | 1195 | 1207 | 1101 | 1148 (+24.1 %) |

d=64 is bound by softmax issue rate, which is why it gains most, and why the hardening op costs
~10 % there versus ~1.4 % at d=128.

**Sustained load** (`agent_space/bench_fixed_shift.py`, CUDA-graph replays back to back): after
~1 s of continuous load both kernels lose ~15 % throughput (power/clock), and fixed-shift's lead
grows to 1.17-1.23x at d=128 (geomean 1.21x), presumably because it does less work per FLOP.

**Registers:** no kernel spills, and the register split doesn't matter. At d=128 (2CTA), the
softmax/correction splits 184/80 (default), 192/64, 192/72, 176/96 and 168/96 are within ±0.3 % of
each other; at d=64, 200/64 (default), 208/48, 192/64 and 192/80 are within run-to-run noise. The online
split is kept (`_FIXED_SHIFT_TUNING_CONFIG` is empty).

**Accuracy at benchmark scale** (`agent_space/accuracy_grid.py`; Gaussian inputs, d=128, 512 sampled
query rows per (batch, head) against fp64 attention; cf. the paper's Table 4):

| N | online rel-ℓ2 | fixed-shift rel-ℓ2 | online max abs | fixed-shift max abs |
|---|---|---|---|---|
| 1K | 2.32e-3 | 5.16e-3 | 1.89e-3 | 4.18e-3 |
| 4K | 2.34e-3 | 5.19e-3 | 7.73e-4 | 1.19e-3 |
| 32K | 2.35e-3 | 5.20e-3 | 1.46e-4 | 2.57e-4 |

The paper reports 2.35e-3 for FA4 and 5.21e-3 for its TLX kernel (dense O, worst over lengths).

## LLaDA 2.0 serving pattern (FluxServe FA4 backend)

FluxServe (`fluxserve/backend/layers/attention/{fa4,metadata}.py`) runs LLaDA 2.0's block-causal
attention without a mask. Every block of `block_length` (64) queries is a separate varlen task that
attends non-causally to its request's paged KV prefix up to the end of that block:

- **prefill:** a P-token prompt is P/64 tasks, task i sees `kv_len = (i + 1)·64`;
- **decode** (each denoising step): one task per request, `kv_len = prefix + 64`.

The call is `flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q, seqused_k, max_seqlen_q=64,
page_table, causal=False)`, with page_size = block_length = 64. Per GPU this is 16 Q / 4 KV heads for
LLaDA2.0-mini (TP 1) and 8 / 1 for LLaDA2.0-flash (TP 4), d = 128. FA4 picks pack_gqa, q_stage 2,
1CTA and no S ping-pong. With page size 64 the K/V pages are loaded by `cp.async` from two load warps
(non-TMA); TMA paging requires page_size == 128.

Fixed-shift supports this pattern (varlen + paged KV, TMA and non-TMA, and split-KV):
`tests/cute/test_flash_attn_fixed_shift.py::test_flash_attn_fixed_shift_paged_block_diffusion` and
`::test_flash_attn_fixed_shift_paged_decode_split_kv`. Relative-ℓ2 error vs exact is 5.0e-3 (online
2.2e-3), the same as for dense attention.

Benchmark: `agent_space/llada2_bench.py`. Idle B200, CUDA-graph replays (FluxServe captures decode
graphs), warm and interleaved, medians of 9. Times are µs per call; speedups are relative to online
FA4 without split-KV unless stated otherwise. `+split` means FA4's split-KV heuristic (`num_splits=0`).

**Decode.** Split-KV, not the softmax, dominates. At batch 1 one block per request gives only
`kv_heads` (or 2·`kv_heads`) CTAs. Online FA4 with `num_splits=0` is 1.3-10.8x faster for mini and
1.4-17x for flash-TP4. At batch 16 the heuristic over-splits mini (0.5-0.9x). Fixed-shift adds on top of
split only when K/V are loaded by TMA:

| mini decode, batch 1 | 1K | 4K | 16K | 64K |
|---|---|---|---|---|
| online, page 64 (production) | 18.8 | 53.4 | 191.6 | 849.1 |
| online + split, page 64 | 14.7 | 18.8 | 28.7 | 78.5 |
| fixed + split, page 64 | 14.4 | 18.5 | 28.5 | 78.9 |
| online + split, page 128 | 17.3 | 21.2 | 31.8 | 76.5 |
| fixed + split, page 128 | 14.9 | 18.4 | 26.5 | 70.0 |

**Prefill**, fixed-shift vs online (same page size):

| prompt | 1K | 4K | 16K | 64K |
|---|---|---|---|---|
| mini, page 64 | +8 % | +4 % | +3 % | −5 % |
| mini, page 128 | +16 % | +20 % | +23 % | +14 % |
| flash-TP4, page 64 | +11 % | +10 % | +12 % | +12 % |
| flash-TP4, page 128 | +16 % | +18 % | +27 % | +26 % |

Decode with split-KV at page 128, fixed vs online: mini +7..20 % and flash-TP4 +15..23 % at batch
1-4 (+8..15 % at batch 16).

**Against the current production configuration** (online, page 64, `num_splits=1`), attention time with
page 128 + split-KV + fixed-shift (picking no-split where the heuristic over-splits) is:

| | mini | flash-TP4 |
|---|---|---|
| decode, batch 1, 64K | 849 → 70.0 µs (12.1x) | 745 → 38.4 µs (19.4x) |
| decode, batch 16, 64K | 882 → 783 µs (1.13x, no split) | 858 → 294 µs (2.9x) |
| prefill, 64K prompt | 14.4 → 12.6 ms (1.14x) | 7.47 → 5.86 ms (1.27x) |

Why page 64 hurts fixed-shift on mini: online FA4 runs at the same speed with either page size, but
fixed-shift gets 17 % faster at page 128 (64K prefill: 15.2 → 12.6 ms). Fixed-shift's denser softmax (no
MUFU stalls) appears to take issue slots from the two `cp.async` load warps. Mini moves 4x more KV than
flash-TP4 per query head, so its loads are the bottleneck.

**Implications for FluxServe:**
1. Use split-KV for decode at small batch (`num_splits=0`, or a batch-aware choice since the
   heuristic over-splits mini at batch 16). This is independent of fixed-shift and is the largest win.
2. To get fixed-shift's gains on mini, use page_size 128 (TMA paging). FluxServe currently requires
   page_size == block_length for its FA4 decode graph.
