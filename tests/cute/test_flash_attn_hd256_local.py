"""SM100 head_dim=256 local (sliding) windows in the shapes of DiffusionGemma's sliding layers.

FluxServe serves those layers (head_dim 256, window W = 1024, canvas 256) from a paged KV cache whose
page is the hd256 kernel's TMA tile (128), with varlen queries (cu_seqlens_q) and seqused_k = KV length:
  prefill / commit: causal, window (W - 1, None): each query sees the W keys ending at itself;
  denoise: a non-causal canvas of 256 queries over prefix + canvas, window (W - 1, None), either per
    query (window_left_anchor="query") or with every row reading the first row's cache slice
    ("first_query", the HF reference decoder's per-row semantics).
The reference is attention_ref per sequence on keys gathered from the pages.
"""

import math
import os
import random

import pytest
import torch

from flash_attn.cute.interface import (
    _flash_attn_fwd,
    flash_attn_func,
    flash_attn_varlen_func,
)
from flash_attn.cute.testing import attention_ref, is_fake_mode, maybe_fake_tensor_mode

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0")) == 1
IS_SM100_FAMILY = torch.cuda.get_device_capability()[0] in (10, 11)

pytestmark = pytest.mark.skipif(not IS_SM100_FAMILY, reason="SM100 hd256 kernel")

D = 256
PAGE = 128
W = 1024
CANVAS = 256
# Prefix lengths around page boundaries and where per-query and per-row windows start to differ
# (P >= W - CANVAS + 1 = 769) or where the window stops covering the whole prefix (P >= W).
CANVAS_PREFIXES = [0, 1, 127, 128, 255, 700, 767, 768, 769, 1022, 1023, 1024, 1025, 1100, 2047, 5000]


def _paged_case(prefix_lens, q_lens, hq, hkv, seed=0, device="cuda"):
    """Sequence b holds prefix_lens[b] + q_lens[b] keys in scattered pages; its queries are the last
    q_lens[b] of them. Unreferenced pages hold NaN; rows past seqused_k in a last page are finite."""
    gen = torch.Generator().manual_seed(seed)
    totals = [p + n for p, n in zip(prefix_lens, q_lens)]
    pages_per_seq = [max(math.ceil(t / PAGE), 1) for t in totals]
    num_spare = 3
    num_pages = sum(pages_per_seq) + num_spare
    # Page layout in Python ints (no data-dependent tensor reads, so it also runs under FakeTensorMode).
    perm = list(range(num_pages))
    random.Random(seed).shuffle(perm)
    spare = perm[sum(pages_per_seq):]
    # Unused table entries point at NaN pages: the kernel must never load past n_block_max.
    rows, used = [], 0
    for npg in pages_per_seq:
        rows.append(perm[used:used + npg] + [spare[0]] * (max(pages_per_seq) - npg))
        used += npg
    page_table = torch.tensor(rows, dtype=torch.int32)
    k_cache = torch.randn(num_pages, PAGE, hkv, D, generator=gen).to(torch.bfloat16)
    v_cache = torch.randn(num_pages, PAGE, hkv, D, generator=gen).to(torch.bfloat16)
    for page in spare:
        k_cache[page] = float("nan")
        v_cache[page] = float("nan")
    for row, t in zip(rows, totals):
        if t % PAGE:
            k_cache[row[t // PAGE], t % PAGE:] = 300.0
            v_cache[row[t // PAGE], t % PAGE:] = 300.0
    # Scale 1.0 as in Gemma; q / 8 keeps the softmax moderately peaked so a key wrongly in or out
    # of the window moves the output.
    q = (torch.randn(sum(q_lens), hq, D, generator=gen) / 8).to(torch.bfloat16)
    cu_seqlens_q = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    seqused_k = torch.tensor(totals, dtype=torch.int32)
    case = {"q": q, "k_cache": k_cache, "v_cache": v_cache, "cu_seqlens_q": cu_seqlens_q,
            "seqused_k": seqused_k, "page_table": page_table}
    case = {k: v.to(device) for k, v in case.items()}
    case.update(totals=totals, q_lens=q_lens, max_seqlen_q=max(q_lens),
                max_seqlen_k=max(pages_per_seq) * PAGE, hq=hq, hkv=hkv)
    return case


def _run(case, causal, window_size, window_left_anchor="query", num_splits=1):
    out, _ = flash_attn_varlen_func(
        case["q"], case["k_cache"], case["v_cache"],
        cu_seqlens_q=case["cu_seqlens_q"], seqused_k=case["seqused_k"],
        max_seqlen_q=case["max_seqlen_q"], max_seqlen_k=case["max_seqlen_k"],
        page_table=case["page_table"], softmax_scale=1.0, causal=causal, window_size=window_size,
        window_left_anchor=window_left_anchor, num_splits=num_splits,
    )
    return out


def _check(out, case, causal, window_size, window_left_anchor="query"):
    """Per-sequence attention_ref (fp32 and bf16-ops) on keys gathered from the pages."""
    q0 = 0
    worst = []
    for b, (total, n) in enumerate(zip(case["totals"], case["q_lens"])):
        pages = case["page_table"][b, : math.ceil(total / PAGE)].long()
        k = case["k_cache"][pages].reshape(-1, case["hkv"], D)[:total][None]
        v = case["v_cache"][pages].reshape(-1, case["hkv"], D)[:total][None]
        # attention_ref scales by 1/sqrt(D); D = 256 makes q * 16 exact, i.e. softmax_scale 1.0.
        q = case["q"][q0:q0 + n][None] * math.sqrt(D)
        kw = {"causal": causal, "window_size": window_size, "window_left_anchor": window_left_anchor}
        out_ref = attention_ref(q, k, v, **kw)[0][0]
        out_pt = attention_ref(q, k, v, upcast=False, reorder_ops=True, **kw)[0][0]
        o = out[q0:q0 + n]
        err = (o - out_ref).abs().max().item()
        err_pt = (out_pt - out_ref).abs().max().item()
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        worst.append((err, err_pt, total - n))
        assert torch.isfinite(o).all(), f"non-finite output for prefix {total - n}"
        assert err <= 2 * err_pt + fwd_atol, f"prefix {total - n}: |out-ref| {err:.3e} vs |pt-ref| {err_pt:.3e}"
        q0 += n
    print("max |out-ref| (|pt-ref|) per prefix:", [(p, f"{e:.1e} ({ep:.1e})") for e, ep, p in worst])


@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize("hq, hkv", [(16, 8), (8, 4)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_hd256_local_paged_prefill_commit(hq, hkv, num_splits):
    """Causal sliding window: ragged prompt prefill and 256-token canvas commit after a prefix."""
    prefill = _paged_case([0] * 7, [1, 129, 300, 1023, 1024, 1500, 2100], hq, hkv)
    commit = _paged_case(CANVAS_PREFIXES, [CANVAS] * len(CANVAS_PREFIXES), hq, hkv, seed=1)
    window = (W - 1, None)
    out_prefill = _run(prefill, True, window, num_splits=num_splits)
    out_commit = _run(commit, True, window, num_splits=num_splits)
    if is_fake_mode():
        return
    _check(out_prefill, prefill, True, window)
    _check(out_commit, commit, True, window)


@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize(
    "window_size, anchor",
    [
        ((W - 1, None), "query"),
        ((W - 1, W - 1), "query"),
        ((W - 1, None), "first_query"),
        ((W - 1, W - 1), "first_query"),
        ((W, None), "first_query"),
    ],
)
@pytest.mark.parametrize("hq, hkv", [(16, 8), (8, 4)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_hd256_local_paged_denoise(hq, hkv, window_size, anchor, num_splits):
    """Non-causal canvas of 256 queries over a paged prefix, per-query or first-query anchored window."""
    case = _paged_case(CANVAS_PREFIXES, [CANVAS] * len(CANVAS_PREFIXES), hq, hkv, seed=2)
    out = _run(case, False, window_size, anchor, num_splits=num_splits)
    if is_fake_mode():
        return
    _check(out, case, False, window_size, anchor)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_hd256_local_canvas_equivalences():
    """On canvas shapes (256 <= W queries): (W-1, None) == (W-1, W-1), and the first-query anchor only
    changes rows whose own window starts inside the prefix (P >= W - CANVAS + 1)."""
    short = [p for p in CANVAS_PREFIXES if p <= W - CANVAS]
    case_all = _paged_case(CANVAS_PREFIXES, [CANVAS] * len(CANVAS_PREFIXES), 16, 8, seed=3)
    case_short = _paged_case(short, [CANVAS] * len(short), 16, 8, seed=4)
    left_only = _run(case_all, False, (W - 1, None))
    both = _run(case_all, False, (W - 1, W - 1))
    query = _run(case_short, False, (W - 1, None))
    first = _run(case_short, False, (W - 1, None), "first_query")
    if is_fake_mode():
        return
    assert torch.equal(left_only, both)
    assert torch.equal(query, first)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_hd256_local_window_is_runtime():
    """Window sizes are runtime values: one compiled kernel serves different windows correctly."""
    case = _paged_case([300, 1500, 4000], [CANVAS] * 3, 16, 8, seed=5)
    outs = {wl: _run(case, False, (wl, None), "first_query") for wl in (W - 1, 200)}
    outs_q = {wl: _run(case, True, (wl, None)) for wl in (W - 1, 37)}
    if is_fake_mode():
        return
    for wl, out in outs.items():
        _check(out, case, False, (wl, None), "first_query")
    for wl, out in outs_q.items():
        _check(out, case, True, (wl, None))


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_hd256_local_paged_empty_ranges():
    """Windows that leave rows (or whole tiles) without keys: those rows are 0 and the page table is
    never read past the sequence's pages (unused entries point at NaN pages)."""
    case = _paged_case([744, 0, 2000], [CANVAS, 300, CANVAS], 16, 8, seed=6)
    outs = {w: _run(case, False, w) for w in ((-50, None), (-300, None), (-4000, None), (None, -300))}
    if is_fake_mode():
        return
    for w, out in outs.items():
        _check(out, case, False, w)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_hd256_local_paged_cuda_graph():
    """Denoise under a CUDA graph (FluxServe replays it): capture once at capacity, then update
    seqused_k and the page table in place and replay."""
    a = _paged_case([700, 1024, 3000, 100], [CANVAS] * 4, 16, 8, seed=7)
    b = _paged_case([2047, 0, 1500, 5000], [CANVAS] * 4, 16, 8, seed=8)
    cap = max(a["max_seqlen_k"], b["max_seqlen_k"])
    for case in (a, b):
        case["max_seqlen_k"] = cap
    # Static metadata buffers at capacity; the KV cache tensors are b's (FluxServe's cache is static).
    static = dict(b)
    static["seqused_k"] = a["seqused_k"].clone()
    static["page_table"] = torch.zeros(4, cap // PAGE, dtype=torch.int32, device="cuda")
    static["page_table"][:, : a["page_table"].shape[1]].copy_(a["page_table"] % b["k_cache"].shape[0])
    eager = _run(b, False, (W - 1, None), "first_query")
    _run(static, False, (W - 1, None), "first_query")  # compile + warm up
    if is_fake_mode():
        return
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graphed = _run(static, False, (W - 1, None), "first_query")
    static["seqused_k"].copy_(b["seqused_k"])
    static["page_table"].zero_()
    static["page_table"][:, : b["page_table"].shape[1]].copy_(b["page_table"])
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(graphed, eager)
    _check(graphed, b, False, (W - 1, None), "first_query")


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_hd256_local_first_query_guards():
    """first_query is forward-only and hd256-only, and needs a left window."""
    q = torch.randn(1, 256, 16, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 512, 8, D, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="window_left_anchor must be one of"):
        flash_attn_func(q, k, k, window_size=(W - 1, None), window_left_anchor="row")
    with pytest.raises(ValueError, match="needs a local window"):
        flash_attn_func(q, k, k, window_left_anchor="first_query")
    with pytest.raises(ValueError, match="head_dim=256"):
        flash_attn_func(q[..., :128].contiguous(), k[..., :128].contiguous(), k[..., :128].contiguous(),
                        window_size=(W - 1, None), window_left_anchor="first_query")
    with pytest.raises(ValueError, match="forward only"):
        _flash_attn_fwd(q.requires_grad_(), k, k, window_size_left=W - 1, window_left_anchor="first_query")
