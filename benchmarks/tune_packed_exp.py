"""Constant search and error report for the packed BF16 exp2 used by softmax_mode="fixed_shift".

CPU only. Reproduces the paper's Table 3 protocol (arXiv 2610.02229 §5.2: 4M exponents on the 21-bit
lattice in [0, 1), seed 20260915) and searches the lane add K, the bf16 multiplier kappa and the
encoding offset OFF of flash_attn.cute.utils.packed_exp2_bf16x2.

    python benchmarks/tune_packed_exp.py            # report for the deployed constants
    python benchmarks/tune_packed_exp.py --search   # grid search (a few minutes)
"""

import argparse
import math

import torch

from flash_attn.cute import utils
from flash_attn.cute.testing import _bf16_bits_to_f32


def evaluate(y: torch.Tensor, K: int, kappa_bits: int, off: int) -> torch.Tensor:
    """Elementwise model of the evaluator at unit scale (s = y): fp32 result of the bf16 weights."""
    t = (y.double() * 128.0 + (1.5 * 2**23 + 127 * 128 + off)).float()  # exact: y on a 2^-21 lattice
    R = t.view(torch.int32).to(torch.int64) & 0xFFFF
    R2 = (R + K) & 0xFFFF
    kappa = _bf16_bits_to_f32(torch.tensor(kappa_bits)).item()
    z = _bf16_bits_to_f32(R) * kappa + _bf16_bits_to_f32(R2)
    return torch.where(z < 0, torch.zeros_like(z), z).to(torch.bfloat16).float()


def stats(p: torch.Tensor, y: torch.Tensor) -> dict:
    r = p.double() / torch.exp2(y.double())
    return {
        "lo": r.min().item(),
        "hi": r.max().item(),
        "band": (r.max() / r.min()).item(),
        "max_rel": (r - 1).abs().max().item(),
        "mean_rel": (r - 1).abs().mean().item(),
    }


def fmt(name, s):
    return (f"{name:44s} ratio [{s['lo']:.4f}, {s['hi']:.4f}] band {s['band']:.4f} "
            f"max rel {s['max_rel']:.3e} mean rel {s['mean_rel']:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--samples", type=int, default=4_000_000)
    args = ap.parse_args()
    gen = torch.Generator().manual_seed(20260915)
    y = (torch.randint(0, 2**21, (args.samples,), generator=gen).double() / 2**21).float()
    K, kap, off = utils.PACKED_EXP2_K, utils.PACKED_EXP2_KAPPA_BF16, utils.PACKED_EXP2_OFF
    print(fmt("ideal bf16(exp2(y))  [paper: 3.89e-3 / 1.41e-3]",
              stats(torch.exp2(y.double()).float().to(torch.bfloat16).float(), y)))
    print(fmt("plain Schraudolph (R alone, no kink)", stats(evaluate(y, 0, 0x0000, 0), y)))
    print(fmt(f"K={K} kappa=0x{kap:04X} OFF=-188  [paper: 6.87e-2 / 5.79e-2]", stats(evaluate(y, K, kap, -188), y)))
    print(fmt(f"deployed K={K} kappa=0x{kap:04X} OFF={off}", stats(evaluate(y, K, kap, off), y)))
    if not args.search:
        return
    sub = y[:500_000]
    best = []
    for K_ in range(48, 81):
        for kap_ in range(0x3FA8, 0x3FC0):  # bf16 1.3125 .. 1.4921875
            # The kink shape fixes the band; OFF only rescales, so centre it analytically.
            s = stats(evaluate(sub, K_, kap_, -188), sub)
            off_c = -188 - round(128 * math.log2(math.sqrt(s["lo"] * s["hi"])))
            best.append((s["band"], K_, kap_, off_c))
    best.sort()
    print("\nbest (band, K, kappa, centred OFF):")
    for band, K_, kap_, off_c in best[:8]:
        print(fmt(f"  K={K_} kappa=0x{kap_:04X} OFF={off_c}", stats(evaluate(y, K_, kap_, off_c), y)))


if __name__ == "__main__":
    main()
