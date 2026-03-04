# freqmuon.py
#
# "No compromise" per-step full frequency-bin orthogonalization on an MxM circular grid,
# but optimized to reduce type casting and allocations:
# - uses rfft2/irfft2 (exact for real inputs) to avoid full FFT + symmetry fills
# - avoids realification via cat() (2m x 2n). Instead does NS5 on split (Re, Im) directly
# - keeps momentum buffers in float32 for conv filters (only one grad->float32 cast)
# - batches conv params with the same (Cout,Cin,kH,kW) shape
#
# Usage:
#   python freqmuon.py --runs 1 --seed 0 --fft_size 8 --ns_steps 2

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import airbench94_muon as ab


@dataclass
class FreqMuonCfg:
    fft_size: int = 8
    ns_steps: int = 2
    eps: float = 1e-7


# Cache: which rfft2 bins must be real (v=0 and v=M/2 when M even), for each M.
_REAL_BINS_CACHE: Dict[int, torch.Tensor] = {}


def _rfft2_real_bins(M: int, device: torch.device) -> torch.Tensor:
    """
    For rfft2 output shaped [M, M//2+1], bins at v=0 (and v=M/2 if M even) are self-conjugate
    along the last axis and must be real for a real spatial signal.
    Return linear indices into flattened freq dimension F = M*(M//2+1) for those bins.
    """
    if M in _REAL_BINS_CACHE and _REAL_BINS_CACHE[M].device == device:
        return _REAL_BINS_CACHE[M]

    W = M // 2 + 1
    u = torch.arange(M, device=device)
    v_list = [0]
    if M % 2 == 0:
        v_list.append(M // 2)
    v = torch.tensor(v_list, device=device, dtype=torch.long)

    # All (u, v in {0, M/2})
    U = u[:, None].expand(M, v.numel())
    V = v[None, :].expand(M, v.numel())
    idx = (U * W + V).reshape(-1).contiguous()  # [M * (#v)]
    _REAL_BINS_CACHE[M] = idx
    return idx


def _zeropower_ns5_split_complex(
    Xre: torch.Tensor, Xim: torch.Tensor, steps: int, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Muon-style NS5 polynomial, but operating on complex matrices represented as (Re, Im) float32.

    Inputs:
      Xre, Xim: [B, m, n] float32

    Returns:
      Qre, Qim: [B, m, n] float32  (approx polar(X) = U V^*)
    """
    assert Xre.ndim == 3 and Xim.ndim == 3
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Frobenius normalize: denom = ||X||_F
    denom = torch.sqrt(
        (Xre * Xre).sum(dim=(-2, -1)) + (Xim * Xim).sum(dim=(-2, -1))
    ).clamp_min(eps)
    Xre = Xre / denom[:, None, None]
    Xim = Xim / denom[:, None, None]

    m, n = Xre.shape[-2], Xre.shape[-1]
    transposed = False
    if m > n:
        Xre = Xre.transpose(-2, -1)
        Xim = Xim.transpose(-2, -1)
        transposed = True

    for _ in range(steps):
        Xt_re_T = Xre.transpose(-2, -1)
        Xt_im_T = Xim.transpose(-2, -1)

        # A = X X^H = (R+iI)(R^T - i I^T)
        # Ar = R R^T + I I^T
        # Ai = I R^T - R I^T
        Ar = Xre @ Xt_re_T
        Ar = Ar + (Xim @ Xt_im_T)

        Ai = Xim @ Xt_re_T
        Ai = Ai - (Xre @ Xt_im_T)

        # A2 = A A
        # (Ar+iAi)^2 = (Ar Ar - Ai Ai) + i(Ar Ai + Ai Ar)
        ArAr = Ar @ Ar
        AiAi = Ai @ Ai
        Ar2 = ArAr - AiAi

        ArAi = Ar @ Ai
        AiAr = Ai @ Ar
        Ai2 = ArAi + AiAr

        # B = bA + cA2
        Br = b * Ar + c * Ar2
        Bi = b * Ai + c * Ai2

        # BX = (Br+iBi)(R+iI) = (BrR - BiI) + i(BrI + BiR)
        BrR = Br @ Xre
        BiI = Bi @ Xim
        Yre = BrR - BiI

        BrI = Br @ Xim
        BiR = Bi @ Xre
        Yim = BrI + BiR

        # X <- aX + BX
        Xre = a * Xre + Yre
        Xim = a * Xim + Yim

    if transposed:
        Xre = Xre.transpose(-2, -1)
        Xim = Xim.transpose(-2, -1)

    return Xre, Xim


def _freq_muon_conv_update_batched(g32: torch.Tensor, cfg: FreqMuonCfg) -> torch.Tensor:
    """
    g32: [P, Cout, Cin, kH, kW] float32
    returns: [P, Cout, Cin, kH, kW] float32
    Full-bin (no subsampling) frequency-domain Muon polar update on an MxM circular grid.
    """
    assert g32.ndim == 5 and g32.dtype == torch.float32
    P, Cout, Cin, kH, kW = g32.shape
    M = cfg.fft_size
    if kH > M or kW > M:
        raise ValueError(f"Kernel {kH}x{kW} > fft_size {M}. Increase --fft_size.")

    device = g32.device
    W = M // 2 + 1
    F = M * W

    # Pad to MxM (float32)
    pad = torch.zeros((P, Cout, Cin, M, M), device=device, dtype=torch.float32)
    pad[:, :, :, :kH, :kW] = g32

    # rfft2: [P, Cout, Cin, M, W] complex64 (real/imag float32)
    Khat = torch.fft.rfft2(pad, dim=(-2, -1), norm="ortho")

    # Flatten freqs: [P*F, Cout, Cin] complex64
    Kflat = Khat.permute(0, 3, 4, 1, 2).reshape(P * F, Cout, Cin)

    # Split into real/imag views (float32)
    Xre = Kflat.real
    Xim = Kflat.imag

    # Polar per frequency bin (full bins, full steps)
    Qre, Qim = _zeropower_ns5_split_complex(Xre, Xim, steps=cfg.ns_steps, eps=cfg.eps)

    # Enforce "must-be-real" bins for rfft2 (v=0 and v=M/2 if even)
    real_f = _rfft2_real_bins(M, device=device)  # indices into [F]
    base = (torch.arange(P, device=device) * F)[:, None]  # [P,1]
    real_idx = (base + real_f[None, :]).reshape(-1)       # [P * (#real_f)]
    Qim[real_idx] = 0.0

    # Write back into Kflat in-place (no clone)
    Kflat.real.copy_(Qre)
    Kflat.imag.copy_(Qim)

    # Unflatten and irfft2 back to spatial (real float32)
    Khat2 = Kflat.reshape(P, M, W, Cout, Cin).permute(0, 3, 4, 1, 2)  # [P,Cout,Cin,M,W]
    upd_pad = torch.fft.irfft2(Khat2, s=(M, M), dim=(-2, -1), norm="ortho")  # [P,Cout,Cin,M,M]

    return upd_pad[:, :, :, :kH, :kW]


class MuonFreqUltraFast(torch.optim.Optimizer):
    """
    Muon-like optimizer for conv filters, but using full-bin frequency-domain polar updates.
    Optimized to avoid unnecessary casting and allocations.
    """
    def __init__(self, params, lr=1e-3, momentum=0.0, nesterov=False, cfg: FreqMuonCfg = FreqMuonCfg()):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if nesterov and momentum <= 0.0:
            raise ValueError("Nesterov requires momentum > 0")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.cfg = cfg

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]

            # Bucket conv params by shape to batch FFT + polar
            buckets: Dict[Tuple[int, int, int, int, torch.dtype, torch.device], List[torch.Tensor]] = {}
            ge32_map: Dict[torch.Tensor, torch.Tensor] = {}

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                st = self.state[p]
                if "momentum_buffer32" not in st:
                    # Float32 momentum buffer to eliminate repeated dtype churn
                    st["momentum_buffer32"] = torch.zeros_like(g, dtype=torch.float32)
                    st["wn_scale"] = math.sqrt(p.data.numel())

                buf32 = st["momentum_buffer32"]
                buf32.mul_(momentum).add_(g.to(torch.float32))
                ge32 = buf32 if not nesterov else (buf32 + momentum * g.to(torch.float32))
                ge32_map[p] = ge32

                if p.data.ndim != 4:
                    # Fallback: simple SGD for non-conv params in this optimizer
                    p.data.add_(ge32.to(p.data.dtype), alpha=-lr)
                    continue

                Cout, Cin, kH, kW = p.data.shape
                key = (Cout, Cin, kH, kW, p.data.dtype, p.data.device)
                buckets.setdefault(key, []).append(p)

            for (Cout, Cin, kH, kW, dt, dev), plist in buckets.items():
                # Keep airbench-style weight normalization (in param dtype, as baseline does)
                for p in plist:
                    st = self.state[p]
                    scale = st["wn_scale"]
                    p.data.mul_(scale / (p.data.norm() + 1e-12))

                ge_stack = torch.stack([ge32_map[p] for p in plist], dim=0)  # float32 [P,Cout,Cin,kH,kW]

                upd32 = _freq_muon_conv_update_batched(ge_stack, self.cfg)    # float32

                # Apply update (single cast here)
                upd = upd32.to(dt)
                for i, p in enumerate(plist):
                    p.data.add_(upd[i], alpha=-lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--fft_size", type=int, default=8)
    parser.add_argument("--ns_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = FreqMuonCfg(fft_size=args.fft_size, ns_steps=args.ns_steps)

    # Patch airbench Muon factory
    def _make_muon(params, lr=1e-3, momentum=0.0, nesterov=False):
        return MuonFreqUltraFast(params, lr=lr, momentum=momentum, nesterov=nesterov, cfg=cfg)

    ab.Muon = _make_muon

    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    if not args.no_compile:
        model.compile(mode="default")

    ab.print_columns(ab.logging_columns_list, is_head=True)
    
    def run_fn(run, model):
        torch.manual_seed(args.seed + run if run != "warmup" else args.seed - 1)
        return ab.main(run, model)

    # run_fn("warmup", model)

    accs = torch.tensor([run_fn(run, model) for run in range(args.runs)])
    print("Mean: %.4f Std: %.4f" % (accs.mean(), accs.std()))


if __name__ == "__main__":
    main()