# freqmuon.py
#
# Replace Muon with PSGD Kron Whiten using dQ = Q0.5EQ1.5
#
# Usage:
#   python freqmuon.py --runs 1 --seed 0
#
# Notes:
#   pip install opt_einsum
#
# This is a torch.optim-style wrapper around PSGD Kron whitening so it can
# plug into code that expects:
#   loss.backward()
#   optimizer.step()

import argparse
from dataclasses import dataclass

import opt_einsum
import torch
import airbench94_muon as ab


@dataclass
class PSGDKronCfg:
    lr_params: float = 1.8
    lr_preconditioner: float = 0.1
    betaL: float = 0.9
    damping: float = 1e-9
    momentum: float = 0.0
    grad_clip_max_amps: tuple = (2.0, 10.0)
    preconditioner_update_probability: float = 1.0
    update_preconditioner_first: bool = True
    whiten_grad: bool = True
    preconditioner_max_size: float = float("inf")
    preconditioner_max_skew: float = 1.0
    preconditioner_init_scale: float | None = None
    dQ: str = "Q0.5EQ1.5"


def norm_lower_bound_spd(A, k=32, half_iters=2):
    """
    Cheap lower bound for ||A||_2 when A is SPD / Hermitian PSD-ish.
    """
    smallest_normal = torch.finfo(A.dtype).smallest_normal
    normalizing_factor = A.diagonal().real.amax() + smallest_normal
    A = A / normalizing_factor
    j = torch.argmax(torch.linalg.vector_norm(A, dim=1))
    V = torch.randn(k, A.shape[1], dtype=A.dtype, device=A.device)
    V = A[j] + torch.sgn(torch.sum(A[j] * V.conj(), dim=1, keepdim=True)) * V
    for _ in range(half_iters):
        V = V @ A
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + smallest_normal
        V = V @ A
    return normalizing_factor * torch.amax(torch.linalg.vector_norm(V, dim=1))


def norm_lower_bound_skh(A, k=32, half_iters=2):
    """
    Cheap lower bound for ||A||_2 when A is skew-Hermitian-ish.
    """
    smallest_normal = torch.finfo(A.dtype).smallest_normal
    normalizing_factor = A.abs().amax() + smallest_normal
    A = A / normalizing_factor
    j = torch.argmax(torch.linalg.vector_norm(A, dim=1))
    V = torch.randn(k, A.shape[1], dtype=A.dtype, device=A.device)
    V = A[j] + torch.sgn(torch.sum(A[j] * V.conj(), dim=1, keepdim=True)) * V
    for _ in range(half_iters):
        V = V @ A
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + smallest_normal
        V = V @ A
    return normalizing_factor * torch.amax(torch.linalg.vector_norm(V, dim=1))


def lift2single(x):
    return x.to(torch.float32) if torch.finfo(x.dtype).eps > 1e-6 else x


def procrustes_step2(Q, max_step_size=1 / 8):
    """
    In-place online orthogonal Procrustes step to keep Q approximately SPD.
    """
    R = Q.H - Q
    R /= norm_lower_bound_skh(R) + torch.finfo(R.dtype).smallest_normal
    RQ = R @ Q
    RRQ = R @ RQ
    tr_RQ = RQ.diagonal().real.sum()
    tr_RRQ = RRQ.diagonal().real.sum()
    a = torch.where(
        tr_RRQ < 0,
        torch.clamp(-tr_RQ / tr_RRQ, max=max_step_size),
        max_step_size,
    )
    Q.add_(a * (RQ + 0.5 * a * RRQ))


def init_kron(t, Scale=1.0, max_size=float("inf"), max_skew=1.0, dQ="Q0.5EQ1.5"):
    """
    Initialize Kronecker-product whitening preconditioner states for tensor t.
    """
    if dQ in {"QUAD4P", "PRO4P"}:
        Scale = Scale ** 2

    shape = t.shape

    if len(shape) == 0:
        Q = [Scale * torch.ones_like(t)]
        L = [lift2single(torch.zeros_like(t.real))]
        exprA = opt_einsum.contract_expression(",->", Q[0].shape, t.shape)
        exprP = opt_einsum.contract_expression(",,->", Q[0].shape, Q[0].shape, t.shape)
        exprGs = [opt_einsum.contract_expression(",->", t.shape, t.shape)]
        exprQs = [opt_einsum.contract_expression(",->", Q[0].shape, t.shape)]
    else:
        if len(shape) > 26:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; einsum runs out of letters."
            )

        scale = Scale ** (1 / len(shape))
        Q, L = [], []
        exprGs, exprQs = [], []
        piece1A, piece2A, piece3A = [], "", ""
        piece1P, piece2P, piece3P, piece4P = [], [], "", ""

        for i, size in enumerate(shape):
            L.append(lift2single(torch.zeros([], dtype=t.real.dtype, device=t.device)))

            if size <= 1 or size > max_size or size**2 > max_skew * t.numel():
                Q.append(scale * torch.ones(size, dtype=t.dtype, device=t.device))

                piece1A.append(opt_einsum.get_symbol(i))
                piece2A += opt_einsum.get_symbol(i)
                piece3A += opt_einsum.get_symbol(i)

                piece1P.append(opt_einsum.get_symbol(i + 26))
                piece2P.append(opt_einsum.get_symbol(i + 26))
                piece3P += opt_einsum.get_symbol(i + 26)
                piece4P += opt_einsum.get_symbol(i + 26)

                piece1 = "".join(
                    [
                        opt_einsum.get_symbol(i + 26)
                        if j == i
                        else opt_einsum.get_symbol(j)
                        for j in range(len(shape))
                    ]
                )
                subscripts = piece1 + "," + piece1 + "->" + opt_einsum.get_symbol(i + 26)
                exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))

                subscripts = opt_einsum.get_symbol(i + 26) + "," + piece1 + "->" + piece1
                exprQs.append(opt_einsum.contract_expression(subscripts, Q[-1].shape, t.shape))
            else:
                Q.append(scale * torch.eye(size, dtype=t.dtype, device=t.device))

                piece1A.append(opt_einsum.get_symbol(i) + opt_einsum.get_symbol(i + 26))
                piece2A += opt_einsum.get_symbol(i + 26)
                piece3A += opt_einsum.get_symbol(i)

                a, b, c = (
                    opt_einsum.get_symbol(i),
                    opt_einsum.get_symbol(i + 26),
                    opt_einsum.get_symbol(i + 805),
                )
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P += c
                piece4P += b

                piece1 = "".join(
                    [
                        opt_einsum.get_symbol(i + 26)
                        if j == i
                        else opt_einsum.get_symbol(j)
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        opt_einsum.get_symbol(i + 805)
                        if j == i
                        else opt_einsum.get_symbol(j)
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1
                    + ","
                    + piece2
                    + "->"
                    + opt_einsum.get_symbol(i + 26)
                    + opt_einsum.get_symbol(i + 805)
                )
                exprGs.append(opt_einsum.contract_expression(subscripts, t.shape, t.shape))

                subscripts = (
                    opt_einsum.get_symbol(i + 26)
                    + opt_einsum.get_symbol(i + 805)
                    + ","
                    + piece2
                    + "->"
                    + piece1
                )
                exprQs.append(opt_einsum.contract_expression(subscripts, Q[-1].shape, t.shape))

        subscripts = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprA = opt_einsum.contract_expression(subscripts, *[q.shape for q in Q], t.shape)

        subscripts = (
            ",".join(piece1P)
            + ","
            + ",".join(piece2P)
            + ","
            + piece3P
            + "->"
            + piece4P
        )
        exprP = opt_einsum.contract_expression(
            subscripts, *[q.shape for q in Q], *[q.shape for q in Q], t.shape
        )

    exprGs, exprQs = tuple(exprGs), tuple(exprQs)

    if dQ == "QEP":
        return [[Q, L], (exprP, exprGs, exprQs)]
    elif dQ == "EQ":
        return [[Q, L], (exprP, exprGs, exprA)]
    elif dQ in {"QEQ", "QUAD", "Q0p5EQ1p5", "Q0.5EQ1.5"}:
        return [[Q, L], (exprP, exprGs)]
    else:
        raise ValueError(f"Unsupported dQ={dQ} for this script")


def balance_kron_precond(Q):
    """
    In-place balancing of factor dynamic ranges.
    """
    order = len(Q)
    if order > 1:
        norms = [torch.max(torch.abs(q)) for q in Q]
        gmean = torch.prod(torch.stack(norms)) ** (1 / order)
        for i, q in enumerate(Q):
            q.mul_(gmean / norms[i])


def precond_grad_kron(QL, exprs, G):
    """
    Apply whitening preconditioner to gradient G.
    """
    Q, exprP = QL[0], exprs[0]
    return exprP(*[q.conj() for q in Q], *Q, G)


def update_precond_kron_whiten_q0p5eq1p5(
    QL, exprs, G, lr=0.1, betaL=0.9, damping=1e-9
):
    """
    Update Kron whitening preconditioner with dQ = Q^0.5 E Q^1.5.
    This is the recommended choice.
    """
    Q, L = QL
    exprP, exprGs = exprs

    total_numel = G.numel()
    damping = damping + torch.finfo(G.dtype).eps * G.abs()
    Pg = exprP(*[q.conj() for q in Q], *Q, G + damping * torch.randn_like(G))

    for i, q in enumerate(Q):
        term1 = exprGs[i](Pg, Pg.conj())

        if q.dim() < 2:
            term2 = total_numel / q.numel()
            ell = torch.max(torch.real(term1)) + term2
            L[i].copy_(torch.max(betaL * L[i] + (1 - betaL) * ell, ell))
            q.mul_(1 - lr / L[i] * (term1 - term2))
        else:
            term2 = total_numel / q.shape[0]
            ell = norm_lower_bound_spd(term1) + term2
            L[i].copy_(torch.max(betaL * L[i] + (1 - betaL) * ell, ell))
            q.sub_(lr / L[i] * (term1 @ q - term2 * q))
            procrustes_step2(q)

    if torch.rand([]) < 0.01:
        balance_kron_precond(Q)


class PSGDKronWhiten(torch.optim.Optimizer):
    """
    Torch-optimizer style PSGD Kron whitening optimizer.

    This wrapper uses p.grad directly, so it can plug into ordinary training code.
    """

    def __init__(
        self,
        params,
        lr=1.8,
        momentum=0.0,
        nesterov=False,  # accepted for compatibility, not used
        cfg: PSGDKronCfg = PSGDKronCfg(),
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

        self.cfg = cfg
        self._update_precond = update_precond_kron_whiten_q0p5eq1p5
        self._precond_grad = precond_grad_kron

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr_params = group["lr"]
            momentum = group["momentum"]

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            grads = [p.grad.detach().to(torch.float32).squeeze() for p in params]

            for p, g in zip(params, grads):
                st = self.state[p]

                if "QL_exprs" not in st:
                    if self.cfg.preconditioner_init_scale is None:
                        scale = (torch.mean(torch.abs(g) ** 4) + self.cfg.damping**4) ** (-1 / 8)
                    else:
                        scale = self.cfg.preconditioner_init_scale

                    st["QL_exprs"] = init_kron(
                        g,
                        Scale=scale,
                        max_size=self.cfg.preconditioner_max_size,
                        max_skew=self.cfg.preconditioner_max_skew,
                        dQ=self.cfg.dQ,
                    )

                if momentum > 0.0:
                    if "momentum_buffer32" not in st:
                        st["momentum_buffer32"] = torch.zeros_like(g, dtype=torch.float32)
                        st["momentum_warmup_counter"] = 0

                    buf = st["momentum_buffer32"]
                    counter = st["momentum_warmup_counter"]
                    beta = min(counter / (1 + counter), momentum)
                    st["momentum_warmup_counter"] = counter + 1
                    buf.mul_(beta).add_(g, alpha=1 - beta)
                    whitening_target = g if self.cfg.whiten_grad else buf
                    search_dir_input = buf
                else:
                    whitening_target = g
                    search_dir_input = g

                do_precond_update = (
                    torch.rand([]).item() < self.cfg.preconditioner_update_probability
                )
                update_first = do_precond_update and self.cfg.update_preconditioner_first
                update_last = do_precond_update and (not self.cfg.update_preconditioner_first)

                if update_first:
                    self._update_precond(
                        *st["QL_exprs"],
                        whitening_target,
                        lr=self.cfg.lr_preconditioner,
                        betaL=self.cfg.betaL,
                        damping=self.cfg.damping,
                    )

                pre_g = self._precond_grad(*st["QL_exprs"], search_dir_input)

                if update_last:
                    self._update_precond(
                        *st["QL_exprs"],
                        whitening_target,
                        lr=self.cfg.lr_preconditioner,
                        betaL=self.cfg.betaL,
                        damping=self.cfg.damping,
                    )

                max_avg_amp, max_element_amp = self.cfg.grad_clip_max_amps
                avg_amp = torch.sqrt(torch.real(torch.mean(pre_g * pre_g.conj())))

                if avg_amp > max_avg_amp:
                    pre_g = pre_g * (max_avg_amp / avg_amp)

                if torch.is_complex(pre_g):
                    pre_g = pre_g / torch.clamp(torch.abs(pre_g) / max_element_amp, min=1.0)
                else:
                    pre_g = pre_g.clamp(min=-max_element_amp, max=max_element_amp)

                p.data.add_(pre_g.view_as(p).to(p.data.dtype), alpha=-lr_params)

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--no_compile", action="store_true")

    parser.add_argument("--lr", type=float, default=1.8)
    parser.add_argument("--momentum", type=float, default=0.1)
    parser.add_argument("--lr_preconditioner", type=float, default=0.3)
    parser.add_argument("--betaL", type=float, default=0.8)
    parser.add_argument("--damping", type=float, default=1e-9)
    parser.add_argument("--preconditioner_update_probability", type=float, default=1.0)
    parser.add_argument("--preconditioner_max_size", type=float, default=float("inf"))
    parser.add_argument("--preconditioner_max_skew", type=float, default=1.0)
    parser.add_argument("--max_avg_amp", type=float, default=2.0)
    parser.add_argument("--max_element_amp", type=float, default=10.0)
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    # torch.set_float32_matmul_precision("high")

    cfg = PSGDKronCfg(
        lr_params=args.lr,
        lr_preconditioner=args.lr_preconditioner,
        betaL=args.betaL,
        damping=args.damping,
        momentum=args.momentum,
        grad_clip_max_amps=(args.max_avg_amp, args.max_element_amp),
        preconditioner_update_probability=args.preconditioner_update_probability,
        update_preconditioner_first=True,
        whiten_grad=True,
        # preconditioner_max_size=args.preconditioner_max_size,
        # preconditioner_max_skew=args.preconditioner_max_skew,
        # preconditioner_init_scale=None,
        # dQ="Q0.5EQ1.5",
        dQ="QUAD"
    )

    def _make_psgd(params, lr=1e-3, momentum=0.0, nesterov=False):
        return PSGDKronWhiten(
            params,
            lr=4e-3,
            momentum=momentum if args.momentum == 0.0 else args.momentum,
            nesterov=True,
            cfg=cfg,
        )

    ab.Muon = _make_psgd

    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
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
