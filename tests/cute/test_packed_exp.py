"""Tests for the packed BF16 exp2 behind softmax_mode="fixed_shift" (utils.packed_exp2_bf16x2).

The GPU results must match flash_attn.cute.testing.packed_exp2_bf16_ref bit for bit.
"""

import math
import os

import pytest
import torch

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Float32, Int32
from quack.compile_utils import make_fake_tensor as fake_tensor

from flash_attn.cute import utils
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.testing import is_fake_mode, maybe_fake_tensor_mode, packed_exp2_bf16_ref

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1


class PackedExp2Kernel:
    """out[i] = packed_exp2_bf16x2(s[2i], s[2i+1]); acc[2i:2i+2] += the two bf16 weights in fp32."""

    def __init__(self, harden: bool):
        self.harden = harden

    @cute.jit
    def __call__(
        self,
        mS: cute.Tensor,
        mP: cute.Tensor,
        mAcc: cute.Tensor,
        A: Float32,
        B: Float32,
        stream: cuda.CUstream = None,
    ):
        n = mP.shape[0]
        self.kernel(mS, mP, mAcc, A, B).launch(
            grid=[cute.ceil_div(n, 128), 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mS: cute.Tensor, mP: cute.Tensor, mAcc: cute.Tensor, A: Float32, B: Float32):
        i = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if i < mP.shape[0]:
            p = utils.packed_exp2_bf16x2(mS[2 * i], mS[2 * i + 1], A, B, harden=self.harden)
            mP[i] = p
            mAcc[2 * i], mAcc[2 * i + 1] = utils.add_bf16x2_to_f32x2(p, mAcc[2 * i], mAcc[2 * i + 1])


def _compiled(harden: bool):
    key = (harden,)
    if key not in _compiled.cache:
        n, n2 = cute.sym_int(), cute.sym_int()
        _compiled.cache[key] = cute.compile(
            PackedExp2Kernel(harden),
            fake_tensor(Float32, (n2,), divisibility=2, leading_dim=0),
            fake_tensor(Int32, (n,), divisibility=1, leading_dim=0),
            fake_tensor(Float32, (n2,), divisibility=2, leading_dim=0),
            Float32(128.0),
            Float32(utils.PACKED_EXP2_MAGIC),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _compiled.cache[key]


_compiled.cache = get_jit_cache("test_packed_exp")


def _run(s: torch.Tensor, A: float, B: float, harden: bool, acc: torch.Tensor):
    """Returns (bf16 weights shaped like s, fp32 acc + weights)."""
    words = torch.empty(s.numel() // 2, dtype=torch.int32, device=s.device)
    acc = acc.clone()
    _compiled(harden)(s, words, acc, A, B)
    return words.view(torch.bfloat16), acc


def _assert_bitexact(gpu: torch.Tensor, ref: torch.Tensor):
    gpu_nan, ref_nan = torch.isnan(gpu.float()), torch.isnan(ref.float())
    assert torch.equal(gpu_nan, ref_nan), f"NaN mismatch at {torch.nonzero(gpu_nan != ref_nan)[:8].tolist()}"
    g, r = gpu.view(torch.int16), ref.view(torch.int16)
    bad = (g != r) & ~gpu_nan
    assert not bad.any(), (
        f"{int(bad.sum())} mismatches, first at {torch.nonzero(bad)[:4].flatten().tolist()}: "
        f"gpu={g[bad][:4].tolist()} ref={r[bad][:4].tolist()}"
    )


def _inputs(device):
    gen = torch.Generator().manual_seed(20260915)
    lattice = torch.randint(0, 2**21, (1 << 20,), generator=gen).float() / 2**21
    sweep = torch.arange(-420.0, 140.0, 1 / 256, dtype=torch.float64).float()
    special = torch.tensor(
        [0.0, -0.0, -math.inf, math.inf, math.nan, 1e30, -1e30, -126.0, -256.0, 127.95, 131.0, -384.5],
        dtype=torch.float32,
    )
    s = torch.cat([lattice, sweep, special])
    return _pad_even(s).to(device)


def _pad_even(s):
    return torch.cat([s, s.new_zeros(s.numel() % 2)])


@pytest.mark.parametrize("harden", [False, True])
@pytest.mark.parametrize("scale_shift", [(1.0, 0.0), (1.0 / math.sqrt(128) * utils.LOG2_E, 0.7)])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_packed_exp2_bitexact(harden, scale_shift):
    """GPU == bit-exact torch model, for unit scale and for a realistic scale with a fractional shift."""
    _compiled(harden)
    if is_fake_mode():
        return
    scale_log2, shift_log2 = scale_shift
    A, B = utils.packed_exp2_coeffs(scale_log2, shift_log2)
    A, B = float(torch.tensor(A, dtype=torch.float32)), float(torch.tensor(B, dtype=torch.float32))
    s = _inputs("cuda")
    if scale_log2 != 1.0:  # scores whose y spans the same range as the unit-scale sweep
        s = s / scale_log2
    acc = torch.randn(s.shape, device="cuda") * 4
    gpu, gpu_acc = _run(s, A, B, harden, acc)
    ref = packed_exp2_bf16_ref(s.cpu(), A, B, harden=harden)
    _assert_bitexact(gpu.cpu(), ref)
    ref_acc = acc.cpu() + ref.float()
    ok = (gpu_acc.cpu() == ref_acc) | (torch.isnan(gpu_acc.cpu()) & torch.isnan(ref_acc))
    assert ok.all(), f"fp32 accumulate mismatches: {int((~ok).sum())}"


@pytest.mark.parametrize("harden", [False, True])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_packed_exp2_accuracy(harden):
    """The paper's protocol (21-bit lattice in [0, 1), seed 20260915) with the centred constants."""
    _compiled(harden)
    if is_fake_mode():
        return
    gen = torch.Generator().manual_seed(20260915)
    y = (torch.randint(0, 2**21, (4_000_000,), generator=gen).double() / 2**21).float()
    gpu, _ = _run(y.cuda(), 128.0, utils.PACKED_EXP2_MAGIC, harden, torch.zeros_like(y, device="cuda"))
    ratio = gpu.cpu().double() / torch.exp2(y.double())
    assert 0.987 < ratio.min().item() and ratio.max().item() < 1.013, (ratio.min().item(), ratio.max().item())
    assert (ratio - 1).abs().mean().item() < 5e-3


@pytest.mark.parametrize("harden", [False, True])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_packed_exp2_ranges(harden):
    """Documented behaviour: mask value -> exactly 0; NaN band only without hardening; y >= 130.6 -> 0."""
    _compiled(harden)
    if is_fake_mode():
        return
    for scale_log2, shift_log2 in [(1.0, 0.0), (0.12752, 0.7), (0.0441, -3.25), (2.3, 10.0)]:
        A, B = utils.packed_exp2_coeffs(scale_log2, shift_log2)
        A, B = float(torch.tensor(A, dtype=torch.float32)), float(torch.tensor(B, dtype=torch.float32))
        mask_value = utils.packed_exp2_mask_value(scale_log2, shift_log2)
        s = torch.full((64,), mask_value, dtype=torch.float32, device="cuda")
        gpu, _ = _run(s, A, B, harden, torch.zeros_like(s))
        assert (gpu.view(torch.int16) == 0).all(), gpu
    y = torch.arange(-126.9, -125.5, 1 / 64, dtype=torch.float32)
    gpu, _ = _run(_pad_even(y).cuda(), 128.0, utils.PACKED_EXP2_MAGIC, harden, torch.zeros(y.numel() + y.numel() % 2, device="cuda"))
    has_nan = torch.isnan(gpu.float()).any().item()
    assert has_nan != harden
    if harden:
        assert (gpu.float() < 1e-37).all()
    y = torch.arange(130.6, 200.0, 0.25, dtype=torch.float32)
    gpu, _ = _run(_pad_even(y).cuda(), 128.0, utils.PACKED_EXP2_MAGIC, harden, torch.zeros(y.numel() + y.numel() % 2, device="cuda"))
    assert (gpu.float() == 0).all()
