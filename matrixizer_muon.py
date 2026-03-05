# freqmuon.py
#
# Muon optimizer using matrixizer()+real NS5 for all parameters.
# Minimal-change version of the previous freqmuon.py with the FFT path removed.
#
# Usage:
#   python freqmuon.py --runs 1 --seed 0 --ns_steps 2

import argparse
import math
from dataclasses import dataclass

import torch
import airbench94_muon as ab


@dataclass
class FreqMuonCfg:
    ns_steps: int = 2
    eps: float = 1e-7


def matrixizer(t):
    """
    It returns triple (f, invf, matrix_shape) for tensor <=> matrix convertion such that
        1) invf(f(t)) = t
        2) matrix_shape = f(t).shape
        3) Preconditioner for matrix f(t) has the minimum size.

    A few examples,
        1), f(t)=t.reshape([1, 1]) for t = torch.randn([])
        2), f(t)=t.reshape([1, 10]) for t = torch.randn(10)
        3), f(t)=t for t = torch.randn(2, 5)
        4), f(t)=t.reshape(6, 5) for t = torch.randn(2,3,5)
        5), f(t)=t.permute(0,1,3,2,4).reshape(42,55) for t = torch.randn(2,3,5,7,11)
    """
    def prod(arr):
        result = 1
        for a in arr:
            result *= a
        return result

    def permutations(p0):
        if len(p0) == 1:
            yield p0
        else:
            for i in range(len(p0)):
                for q in permutations(p0[:i] + p0[i+1:]):
                    yield (p0[i], *q)

    if t.dim() == 2:
        return (lambda u: u, lambda v: v, t.shape)
    elif t.dim() < 2:
        mtx_shape = (1, t.numel())
        return (
            lambda u, shape=mtx_shape: u.reshape(shape),
            lambda v, shape=t.shape: v.reshape(shape),
            mtx_shape,
        )
    else:
        p0, s0 = tuple(range(t.dim())), t.shape
        min_precond_size, opt_p, opt_s, opt_i = float("inf"), None, None, None
        for p in permutations(p0):
            s = tuple(s0[j] for j in p)
            for i in range(1, len(p)):
                new_size = prod(s[:i]) ** 2 + prod(s[i:]) ** 2
                if new_size < min_precond_size:
                    min_precond_size = new_size
                    opt_p, opt_s, opt_i = p, s, i

        if opt_p == p0:
            mtx_shape = (prod(s0[:opt_i]), prod(s0[opt_i:]))
            return (
                lambda u, shape=mtx_shape: u.reshape(shape),
                lambda v, shape=s0: v.reshape(shape),
                mtx_shape,
            )
        else:
            mtx_shape = (prod(opt_s[:opt_i]), prod(opt_s[opt_i:]))
            q = tuple(pair[1] for pair in sorted([(k, i) for (i, k) in enumerate(opt_p)]))
            return (
                lambda u, permute=opt_p, shape=mtx_shape: u.permute(permute).reshape(shape),
                lambda v, permute=q, shape=opt_s: v.reshape(shape).permute(permute),
                mtx_shape,
            )


def _zeropower_ns5_real(X: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """
    Muon-style NS5 polynomial for real matrices.

    Input:
      X: [m, n] float32

    Returns:
      Q: [m, n] float32  (approx polar(X) = U V^T)
    """
    assert X.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    denom = X.norm().clamp_min(eps)
    X = X / denom

    m, n = X.shape
    transposed = False
    if m > n:
        X = X.transpose(-2, -1)
        transposed = True

    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        A2 = A @ A
        B = b * A + c * A2
        X = a * X + B @ X

    if transposed:
        X = X.transpose(-2, -1)

    return X


def _matrix_muon_update(g32: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    """
    Generic tensor Muon update using matrixizer() followed by real NS5.
    """
    f, invf, _ = matrixizer(g32)
    G = f(g32)
    U = _zeropower_ns5_real(G, steps=steps, eps=eps)
    return invf(U)


class MuonFreqUltraFast(torch.optim.Optimizer):
    """
    Muon-like optimizer using matrixizer()+real NS5 for all parameters.
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

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                st = self.state[p]
                if "momentum_buffer32" not in st:
                    st["momentum_buffer32"] = torch.zeros_like(g, dtype=torch.float32)
                    st["wn_scale"] = math.sqrt(p.data.numel())

                g32 = g.to(torch.float32)
                buf32 = st["momentum_buffer32"]
                buf32.mul_(momentum).add_(g32)
                ge32 = buf32 if not nesterov else (buf32 + momentum * g32)

                scale = st["wn_scale"]
                p.data.mul_(scale / (p.data.norm() + 1e-26))

                upd32 = _matrix_muon_update(ge32, steps=self.cfg.ns_steps, eps=self.cfg.eps)
                p.data.add_(upd32.to(p.data.dtype), alpha=-lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--ns_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = FreqMuonCfg(ns_steps=args.ns_steps)

    def _make_muon(params, lr=1e-3, momentum=0.0, nesterov=False):
        return MuonFreqUltraFast(params, lr=1.8, momentum=momentum, nesterov=nesterov, cfg=cfg)

    ab.Muon = _make_muon

    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    # lazy dont wanan install on colab
    # if not args.no_compile:
    #     model.compile(mode="default")

    ab.print_columns(ab.logging_columns_list, is_head=True)

    def run_fn(run, model):
        torch.manual_seed(args.seed + run if run != "warmup" else args.seed - 1)
        return ab.main(run, model)

    accs = torch.tensor([run_fn(run, model) for run in range(args.runs)])
    print("Mean: %.4f Std: %.4f" % (accs.mean(), accs.std()))


if __name__ == "__main__":
    main()
