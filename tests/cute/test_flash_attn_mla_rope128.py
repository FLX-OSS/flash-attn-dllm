"""MLA (qv) forward with a 128-wide rope part on SM100, in the shapes of DiffusionGemma's full layers.

DiffusionGemma's full-attention layers (head_dim 512, 16 q / 2 kv heads, scale 1.0) derive K and V from
one projection u: V = u / rms(u), K = RoPE(c * V) with a per-layer scalar c (the k_norm weight) and
RoPE rotating 128 of the 512 dims. So S = c * (q_R . RoPE(V)_R + q_notR . V_notR): MLA with a 128-wide
rope part (q_pe, k_pe), the latent V, qv = q with the rope dims zeroed, and softmax_scale = c.
"""

import math
import os
import random

import pytest
import torch

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import attention_ref, is_fake_mode, maybe_fake_tensor_mode

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0")) == 1
IS_SM100_FAMILY = torch.cuda.get_device_capability()[0] in (10, 11)

pytestmark = pytest.mark.skipif(not IS_SM100_FAMILY, reason="SM100 MLA kernel")

D, DV, PAGE = 128, 512, 128


def _check(out, out_ref, out_pt, what=""):
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    err = (out.float() - out_ref.float()).abs().max().item()
    err_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    print(f"{what} |out-ref| {err:.3e} |pt-ref| {err_pt:.3e}")
    assert torch.isfinite(out).all(), what
    assert err <= 2 * err_pt + fwd_atol, f"{what}: |out-ref| {err:.3e} vs |pt-ref| {err_pt:.3e}"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads, nheads_kv", [(16, 2), (8, 1), (16, 16)])
@pytest.mark.parametrize(
    "seqlen_q, seqlen_k",
    [(1, 513), (113, 640), (256, 256), (256, 1280), (256, 4352), (300, 300), (2048, 2048)],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_mla_rope128_dense(seqlen_q, seqlen_k, nheads, nheads_kv, causal):
    torch.random.manual_seed(0)
    batch = 4 if seqlen_k <= 1280 else 2
    dt, dev = torch.bfloat16, "cuda"
    q = torch.randn(batch, seqlen_q, nheads, D, device=dev, dtype=dt)
    k = torch.randn(batch, seqlen_k, nheads_kv, D, device=dev, dtype=dt)
    v = torch.randn(batch, seqlen_k, nheads_kv, DV, device=dev, dtype=dt)
    qv = torch.randn(batch, seqlen_q, nheads, DV, device=dev, dtype=dt)
    out, lse = flash_attn_func(q, k, v, qv=qv, causal=causal, return_lse=True)
    out2, _ = flash_attn_func(q, k, v, qv=qv, causal=causal, return_lse=True)
    if is_fake_mode():
        return
    out_ref, _ = attention_ref(q, k, v, qv=qv, causal=causal)
    out_pt, _ = attention_ref(q, k, v, qv=qv, causal=causal, upcast=False, reorder_ops=True)
    _check(out, out_ref, out_pt, f"dense {seqlen_q}x{seqlen_k} h{nheads}/{nheads_kv} causal={causal}")
    assert not torch.isnan(lse).any()
    assert torch.equal(out, out2), "non-deterministic"


def _paged_case(prefix_lens, q_lens, nheads, nheads_kv, seed=0, dev="cuda"):
    """Sequence b holds prefix_lens[b] + q_lens[b] keys in scattered 128-token pages; its queries are
    the last q_lens[b]. Unreferenced pages hold NaN; rows past seqused_k in a last page are finite."""
    gen = torch.Generator().manual_seed(seed)
    totals = [p + n for p, n in zip(prefix_lens, q_lens)]
    pages_per_seq = [max(math.ceil(t / PAGE), 1) for t in totals]
    num_pages = sum(pages_per_seq) + 2
    perm = list(range(num_pages))
    random.Random(seed).shuffle(perm)
    spare = perm[sum(pages_per_seq):]
    rows, used = [], 0
    for npg in pages_per_seq:
        rows.append(perm[used:used + npg] + [spare[0]] * (max(pages_per_seq) - npg))
        used += npg
    k_cache = torch.randn(num_pages, PAGE, nheads_kv, D, generator=gen).to(torch.bfloat16)
    v_cache = torch.randn(num_pages, PAGE, nheads_kv, DV, generator=gen).to(torch.bfloat16)
    for page in spare:
        k_cache[page] = float("nan")
        v_cache[page] = float("nan")
    for row, t in zip(rows, totals):
        if t % PAGE:
            k_cache[row[t // PAGE], t % PAGE:] = 300.0
            v_cache[row[t // PAGE], t % PAGE:] = 300.0
    q = torch.randn(sum(q_lens), nheads, D, generator=gen).to(torch.bfloat16)
    qv = torch.randn(sum(q_lens), nheads, DV, generator=gen).to(torch.bfloat16)
    case = {
        "q": q, "qv": qv, "k_cache": k_cache, "v_cache": v_cache,
        "cu_seqlens_q": torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0).to(torch.int32),
        "seqused_k": torch.tensor(totals, dtype=torch.int32),
        "page_table": torch.tensor(rows, dtype=torch.int32),
    }
    case = {k: v.to(dev) for k, v in case.items()}
    case.update(totals=totals, q_lens=q_lens, max_seqlen_q=max(q_lens), max_seqlen_k=max(pages_per_seq) * PAGE,
                nheads_kv=nheads_kv)
    return case


def _run_paged(case, causal, softmax_scale=None):
    out, _ = flash_attn_varlen_func(
        case["q"], case["k_cache"], case["v_cache"], qv=case["qv"],
        cu_seqlens_q=case["cu_seqlens_q"], seqused_k=case["seqused_k"],
        max_seqlen_q=case["max_seqlen_q"], max_seqlen_k=case["max_seqlen_k"],
        page_table=case["page_table"], causal=causal, softmax_scale=softmax_scale,
    )
    return out


def _check_paged(out, case, causal):
    q0 = 0
    for b, (total, n) in enumerate(zip(case["totals"], case["q_lens"])):
        pages = case["page_table"][b, : math.ceil(total / PAGE)].long()
        k = case["k_cache"][pages].reshape(-1, case["nheads_kv"], D)[:total][None]
        v = case["v_cache"][pages].reshape(-1, case["nheads_kv"], DV)[:total][None]
        q, qv = case["q"][q0:q0 + n][None], case["qv"][q0:q0 + n][None]
        out_ref, _ = attention_ref(q, k, v, qv=qv, causal=causal)
        out_pt, _ = attention_ref(q, k, v, qv=qv, causal=causal, upcast=False, reorder_ops=True)
        _check(out[q0:q0 + n][None], out_ref, out_pt, f"seq {b} prefix {total - n} q {n}")
        q0 += n


CANVAS_PREFIXES = [0, 1, 127, 128, 255, 769, 1023, 1024, 1100, 2047, 5000]


@pytest.mark.parametrize("nheads, nheads_kv", [(16, 2), (8, 1)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_mla_rope128_paged_block_diffusion(nheads, nheads_kv):
    """DiffusionGemma's serving calls: ragged causal prefill, then 256-query canvases over a paged prefix,
    non-causal (denoise) and causal (commit)."""
    prefill = _paged_case([0] * 6, [1, 129, 300, 1024, 1500, 2100], nheads, nheads_kv, seed=1)
    canvas = _paged_case(CANVAS_PREFIXES, [256] * len(CANVAS_PREFIXES), nheads, nheads_kv, seed=2)
    out_prefill = _run_paged(prefill, causal=True)
    out_denoise = _run_paged(canvas, causal=False)
    out_commit = _run_paged(canvas, causal=True)
    if is_fake_mode():
        return
    _check_paged(out_prefill, prefill, True)
    _check_paged(out_denoise, canvas, False)
    _check_paged(out_commit, canvas, True)


def _gemma_rope(x, positions, theta=1e6, rotated_pairs=64):
    """NeoX 'proportional' RoPE on a 512-wide head: pairs (j, j + 256) for j < 64 rotate, the rest pass."""
    half = x.shape[-1] // 2
    inv_freq = torch.zeros(half, dtype=torch.float64, device=x.device)
    inv_freq[:rotated_pairs] = 1.0 / theta ** (
        torch.arange(0, 2 * rotated_pairs, 2, dtype=torch.float64, device=x.device) / x.shape[-1]
    )
    freqs = positions.double()[:, None] * inv_freq[None]
    emb = torch.cat((freqs, freqs), -1)[:, None, :]  # (seq, 1, 512)
    rot = torch.cat((-x[..., half:], x[..., :half]), -1)
    return (x.double() * emb.cos() + rot.double() * emb.sin()).to(x.dtype)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("prefix", [0, 700, 4096])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_mla_rope128_matches_gemma_full_layer(prefix, causal):
    """The DiffusionGemma rewrite: FA4 MLA (q_pe . k_pe + q_lat . V, softmax_scale = c) against the full
    layer's own math, q . RoPE(c * V) with scale 1.0, computed in fp64."""
    torch.random.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    nheads, nheads_kv, canvas, c = 16, 2, 256, 0.0623
    seqlen_k = prefix + canvas
    R = torch.cat([torch.arange(64), torch.arange(256, 320)]).to(dev)  # rotated dims of the 512-wide head
    u = torch.randn(seqlen_k, nheads_kv, DV, device=dev)
    V = (u * (u.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()).to(dt)  # v_norm(u)
    q = (torch.randn(canvas, nheads, DV, device=dev)).to(dt)  # q_norm(x Wq), before RoPE
    q_hat = _gemma_rope(q, torch.arange(prefix, seqlen_k, device=dev))
    k_rope_full = _gemma_rope(V, torch.arange(seqlen_k, device=dev))  # RoPE(V); K = c * RoPE(V)
    q_pe, k_pe = q_hat[..., R].contiguous(), k_rope_full[..., R].contiguous()
    q_lat = q_hat.clone()
    q_lat[..., R] = 0
    out = flash_attn_func(q_pe[None], k_pe[None], V[None], qv=q_lat[None], causal=causal, softmax_scale=c)[0][0]
    if is_fake_mode():
        return
    # Reference: the layer as written, S = q_hat . K with K = c * RoPE(V) (fp64), scale 1.0.
    K = c * k_rope_full.double()
    g = nheads // nheads_kv
    s = torch.einsum("qhd,khd->hqk", q_hat.double(), K.repeat_interleave(g, 1))
    if causal:
        qpos = torch.arange(prefix, seqlen_k, device=dev)[:, None]
        s = s.masked_fill(torch.arange(seqlen_k, device=dev)[None] > qpos, float("-inf"))
    ref = torch.einsum("hqk,khd->qhd", s.softmax(-1), V.double().repeat_interleave(g, 1))
    # bf16 pipeline of the original formulation (K rounded to bf16, fp32 attention) as the error budget.
    s_bf = torch.einsum("qhd,khd->hqk", q_hat.float(), (c * k_rope_full.float()).to(dt).float().repeat_interleave(g, 1))
    if causal:
        s_bf = s_bf.masked_fill(torch.arange(seqlen_k, device=dev)[None] > qpos, float("-inf"))
    ref_bf = torch.einsum("hqk,khd->qhd", s_bf.softmax(-1), V.float().repeat_interleave(g, 1))
    err = (out.double() - ref).abs().max().item()
    err_bf = (ref_bf.double() - ref).abs().max().item()
    print(f"prefix {prefix} causal {causal}: |fa4 mla - fp64| {err:.3e}, |bf16 original - fp64| {err_bf:.3e}")
    assert err <= 2 * err_bf + 1e-3
