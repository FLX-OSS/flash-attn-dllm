"""Forward tests for softmax_mode="fixed_shift" (constant row shift + packed bf16 exp2, SM100).

The kernel is checked against two references:
  * attention_fixed_shift_ref: the same arithmetic (bf16 weights from the packed evaluator, fp32 row
    sums), so the only differences are fp32 accumulation order and bf16 output rounding;
  * attention_ref in fp32: exact attention, with the model's own approximation error as the budget.
"""

import math
import os

import pytest
import torch

from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.testing import (
    attention_fixed_shift_ref,
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
    is_fake_mode,
    maybe_fake_tensor_mode,
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
IS_SM100_FAMILY = torch.cuda.get_device_capability()[0] in (10, 11)

pytestmark = pytest.mark.skipif(not IS_SM100_FAMILY, reason="fixed-shift softmax is SM100/SM110 only")


def _check_out(out, out_model, out_ref, lse=None, lse_model=None, lse_ref=None):
    """out vs the fixed-shift model and vs exact attention (with the model's own error as the budget).

    The model uses exact fp32 scores while the kernel's come from the tensor cores; the evaluator rounds
    128 * y to an integer, so a ~1e-6 score difference occasionally moves a weight by one 1/128-binade
    step (<= ~0.9%). Hence the max-error slack against the model; the mean must stay at rounding level.
    """
    out = out.float()
    rounding = (out_model.to(torch.bfloat16).float() - out_model).abs().max().item()
    err = (out - out_model).abs()
    err_model, mean_model = err.max().item(), err.mean().item()
    budget = (out_model - out_ref.float()).abs().max().item()
    err_ref = (out - out_ref.float()).abs().max().item()
    msg = (f"|out-model| max {err_model:.2e} mean {mean_model:.2e} (rounding {rounding:.2e}); "
           f"|out-exact| {err_ref:.2e} (model {budget:.2e})")
    if lse is not None:
        finite = torch.isfinite(lse_model)
        assert torch.equal(torch.isfinite(lse), finite)
        err_lse = (lse - lse_model)[finite].abs().max().item() if finite.any() else 0.0
        msg += f"; |lse-model| {err_lse:.2e}"
        if lse_ref is not None:
            err_lse_ref = (lse - lse_ref.float())[finite].abs().max().item() if finite.any() else 0.0
            msg += f" |lse-exact| {err_lse_ref:.2e}"
    print(msg)
    assert err_model <= 4 * rounding + 2e-3, msg
    assert mean_model <= 0.1 * rounding + 1e-5, msg
    assert err_ref <= 1.5 * budget + 2 * rounding, msg
    if lse is not None:
        assert err_lse <= 2e-2, msg
        if lse_ref is not None:
            assert err_lse_ref <= 2.5e-2, msg


# (seqlen_q, seqlen_k): q_stage 1, 1CTA q_stage 2 with partial blocks, 2CTA with partial blocks, 2CTA.
SEQLENS = [(64, 128), (200, 333), (1000, 777), (2048, 2048)]


@pytest.mark.parametrize("d, dv", [(64, 64), (96, 96), (128, 128), (192, 128)])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("seqlen_q, seqlen_k", SEQLENS)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_fixed_shift_output(seqlen_q, seqlen_k, mha_type, d, dv):
    device = "cuda"
    torch.random.manual_seed(0)
    batch = 4 if seqlen_k <= 1024 else 2
    nheads, nheads_kv = (6, 6) if mha_type == "mha" else (6, 2)
    q = torch.randn(batch, seqlen_q, nheads, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen_k, nheads_kv, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen_k, nheads_kv, dv, device=device, dtype=torch.bfloat16)
    out, lse = flash_attn_func(q, k, v, softmax_mode="fixed_shift", return_lse=True)
    out_nolse, _ = flash_attn_func(q, k, v, softmax_mode="fixed_shift")
    if is_fake_mode():
        return
    out_model, lse_model = attention_fixed_shift_ref(q, k, v)
    out_ref, _, lse_ref = attention_ref(q.float(), k.float(), v.float(), return_lse=True)
    _check_out(out, out_model, out_ref, lse, lse_model, lse_ref)
    assert torch.equal(out, out_nolse)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen_q, seqlen_k", [(113, 203), (1024, 1000), (2048, 1500)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_varlen_fixed_shift_output(seqlen_q, seqlen_k, d):
    device = "cuda"
    torch.random.manual_seed(0)
    batch, nheads, nheads_kv = 5, 6, 2
    q = torch.randn(batch, seqlen_q, nheads, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen_k, nheads_kv, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen_k, nheads_kv, d, device=device, dtype=torch.bfloat16)
    query_padding_mask = generate_random_padding_mask(seqlen_q, batch, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch, device, mode="random")
    (q_unpad, k_unpad, v_unpad, _, cu_seqlens_q, cu_seqlens_k, _, _, max_seqlen_q, max_seqlen_k,
     *_, output_pad_fn, _, _) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)
    out_unpad, _lse = flash_attn_varlen_func(
        q_unpad.detach(), k_unpad.detach(), v_unpad.detach(),
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        softmax_mode="fixed_shift", return_lse=True,
    )
    if is_fake_mode():
        return
    out = output_pad_fn(out_unpad)
    out_model, _ = attention_fixed_shift_ref(
        q, k, v, key_padding_mask=key_padding_mask, query_padding_mask=query_padding_mask
    )
    out_ref, _ = attention_ref(
        q.float(), k.float(), v.float(), query_padding_mask, key_padding_mask
    )
    _check_out(out, out_model, out_ref)


@pytest.mark.parametrize("softmax_shift", [-3.0, 5.0])
@pytest.mark.parametrize("logit_scale", [1.0, 6.0])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_fixed_shift_shift_and_large_logits(softmax_shift, logit_scale):
    """Non-zero shifts, and inputs scaled so that scaled logits reach about +-50 (inside the contract)."""
    device = "cuda"
    torch.random.manual_seed(1)
    b, s, h, d = 2, 1024, 4, 128
    q = (torch.randn(b, s, h, d, device=device) * math.sqrt(logit_scale)).to(torch.bfloat16)
    k = (torch.randn(b, s, h, d, device=device) * math.sqrt(logit_scale)).to(torch.bfloat16)
    v = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16)
    out, lse = flash_attn_func(
        q, k, v, softmax_mode="fixed_shift", softmax_shift=softmax_shift, return_lse=True
    )
    if is_fake_mode():
        return
    out_model, lse_model = attention_fixed_shift_ref(q, k, v, softmax_shift=softmax_shift)
    out_ref, _, lse_ref = attention_ref(q.float(), k.float(), v.float(), return_lse=True)
    _check_out(out, out_model, out_ref, lse, lse_model, lse_ref)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_fixed_shift_harden(monkeypatch):
    """FLASH_ATTENTION_FIXED_SHIFT_HARDEN=1: same output in range, no NaN for scores in the NaN band."""
    device = "cuda"
    torch.random.manual_seed(2)
    b, s, h, d = 2, 512, 4, 128
    q = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16)
    # Row 0 of head 0: one key with scaled logit -87.5 (inside the unhardened NaN band).
    q_band, k_band = q.clone(), k.clone()
    q_band[0, 0, 0] = 0
    k_band[0, 0, 0] = 0
    q_band[0, 0, 0, 0] = 16.0
    k_band[0, 0, 0, 0] = -87.5 * math.sqrt(d) / 16.0
    monkeypatch.setenv("FLASH_ATTENTION_FIXED_SHIFT_HARDEN", "1")
    out_h = flash_attn_func(q, k, v, softmax_mode="fixed_shift")[0]
    out_h_band = flash_attn_func(q_band, k_band, v, softmax_mode="fixed_shift")[0]
    monkeypatch.setenv("FLASH_ATTENTION_FIXED_SHIFT_HARDEN", "0")
    out = flash_attn_func(q, k, v, softmax_mode="fixed_shift")[0]
    out_band = flash_attn_func(q_band, k_band, v, softmax_mode="fixed_shift")[0]
    if is_fake_mode():
        return
    assert torch.equal(out_h, out)  # no score of these inputs is in the NaN band or below
    assert torch.isnan(out_band[0, 0, 0]).any(), "expected the documented NaN band without hardening"
    assert not torch.isnan(out_h_band).any()
    out_model, _ = attention_fixed_shift_ref(q_band, k_band, v, harden=True)
    out_ref, _ = attention_ref(q_band.float(), k_band.float(), v.float())
    _check_out(out_h_band, out_model, out_ref)


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_fixed_shift_gating(monkeypatch):
    """Explicit requests that cannot be honored raise; env-driven ones fall back to online softmax."""
    device = "cuda"
    torch.random.manual_seed(3)
    q, k, v = [torch.randn(2, 512, 4, 128, device=device, dtype=torch.bfloat16) for _ in range(3)]
    for kwargs in [{"causal": True}, {"softcap": 15.0}, {"num_splits": 2}, {"window_size": (64, 64)}]:
        with pytest.raises(ValueError, match="fixed_shift"):
            flash_attn_func(q, k, v, softmax_mode="fixed_shift", **kwargs)
    with pytest.raises(ValueError, match="fixed_shift"):
        flash_attn_func(q.half(), k.half(), v.half(), softmax_mode="fixed_shift")
    with pytest.raises(ValueError, match="softmax_mode must be one of"):
        flash_attn_func(q, k, v, softmax_mode="fixed")
    qg, kg, vg = [t.clone().requires_grad_() for t in (q, k, v)]
    with pytest.raises(ValueError, match="forward only"):
        flash_attn_func(qg, kg, vg, softmax_mode="fixed_shift")

    monkeypatch.setenv("FLASH_ATTENTION_SOFTMAX_MODE", "fixed_shift")
    out_env = flash_attn_func(q, k, v)[0]  # eligible: fixed-shift kernel
    out_env_grad = flash_attn_func(qg, kg, vg)[0]  # requires grad: falls back to online
    out_env_causal = flash_attn_func(q, k, v, causal=True)[0]  # causal: falls back to online
    monkeypatch.setenv("FLASH_ATTENTION_SOFTMAX_MODE", "online")
    out_online = flash_attn_func(q, k, v)[0]
    out_online_grad = flash_attn_func(qg, kg, vg)[0]
    out_online_causal = flash_attn_func(q, k, v, causal=True)[0]
    out_fixed = flash_attn_func(q, k, v, softmax_mode="fixed_shift")[0]
    if is_fake_mode():
        return
    assert torch.equal(out_env, out_fixed)
    assert torch.equal(out_env_grad, out_online_grad)
    assert torch.equal(out_env_causal, out_online_causal)
    assert not torch.equal(out_fixed, out_online)
