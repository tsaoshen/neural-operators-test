import argparse
import os
import math
import torch
import matplotlib.pyplot as plt


def total_variation(u: torch.Tensor) -> torch.Tensor:
    # u: [..., N]
    return torch.sum(torch.abs(u[..., 1:] - u[..., :-1]), dim=-1)


def shock_location(u: torch.Tensor, dx: float) -> torch.Tensor:
    # u: [N] or [B,N]
    # indicator: |du/dx|
    du = torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)
    s = torch.abs(du) / (2.0 * dx)
    return torch.argmax(s, dim=-1)  # index


def overshoot_near_shock(u_pred: torch.Tensor, u_true: torch.Tensor, dx: float, radius: int = 8) -> torch.Tensor:
    """
    u_pred, u_true: [T,N] (single sample)
    Overshoot score: max_{|i-ishock|<=r} |u_pred - u_true|
    """
    T, N = u_true.shape
    out = []
    for j in range(T):
        ish = int(shock_location(u_true[j], dx).item())
        lo = (ish - radius) % N
        hi = (ish + radius) % N
        if lo <= hi:
            idx = torch.arange(lo, hi + 1)
        else:
            idx = torch.cat([torch.arange(lo, N), torch.arange(0, hi + 1)])
        out.append(torch.max(torch.abs(u_pred[j, idx] - u_true[j, idx])))
    return torch.stack(out, dim=0)  # [T]


def kruzkov_entropy(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # u: [...], k: [K]
    # returns eta_k(u)=|u-k| with broadcasting
    return torch.abs(u[..., None] - k[None, ...])


def kruzkov_entropy_flux(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # f(u)=u^2/2
    # q_k(u)=sign(u-k)(f(u)-f(k))
    f = 0.5 * u * u
    fk = 0.5 * k * k
    sgn = torch.sign(u[..., None] - k[None, ...])
    return sgn * (f[..., None] - fk[None, ...])


def entropy_violation_proxy(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, n_k: int = 16) -> float:
    """
    Proxy: for random k's and nonnegative test function phi(x,t),
    compute E = ∫∫ (eta_k(u) phi_t + q_k(u) phi_x) dx dt
    and report mean relu(E) across k.

    u: [T,N], x:[N], t:[T]
    """
    device = u.device
    dtype = u.dtype
    Tn, N = u.shape
    dx = float((x[1] - x[0]).item())
    dt = float((t[1] - t[0]).item()) if Tn > 1 else 1.0

    # choose k's from data range
    umin = float(u.min().item())
    umax = float(u.max().item())
    k = torch.linspace(umin, umax, steps=n_k, device=device, dtype=dtype)

    # simple nonnegative separable phi(x,t) = bump_x(x) * bump_t(t)
    # bump_x: Gaussian centered mid-domain; bump_t: Gaussian centered mid-time
    xc = 0.5 * float(x[-1].item() + dx)  # about L/2
    tc = 0.5 * float(t[-1].item())
    sigx = 0.15 * float(x[-1].item() + dx)
    sigt = 0.25 * float(t[-1].item() + 1e-12)

    bump_x = torch.exp(-0.5 * ((x - xc) / sigx) ** 2)  # [N], >=0
    bump_t = torch.exp(-0.5 * ((t - tc) / sigt) ** 2)  # [T], >=0
    phi = bump_t[:, None] * bump_x[None, :]            # [T,N]

    # derivatives (finite diff, periodic in x)
    phi_t = torch.zeros_like(phi)
    if Tn > 1:
        phi_t[1:-1] = (phi[2:] - phi[:-2]) / (2.0 * dt)
        phi_t[0] = (phi[1] - phi[0]) / dt
        phi_t[-1] = (phi[-1] - phi[-2]) / dt

    phi_x = (torch.roll(phi, -1, dims=1) - torch.roll(phi, 1, dims=1)) / (2.0 * dx)

    eta = kruzkov_entropy(u, k)        # [T,N,K]
    q = kruzkov_entropy_flux(u, k)     # [T,N,K]

    integrand = eta * phi_t[..., None] + q * phi_x[..., None]  # [T,N,K]
    E = torch.sum(integrand, dim=(0, 1)) * (dx * dt)           # [K]
    score = torch.mean(torch.relu(E)).item()
    return score


def load_preds(path: str):
    d = torch.load(path, map_location="cpu")
    # expected keys: pred, true, u0, x, t
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_a", type=str, required=True, help="e.g. runs/preds_fno.pt")
    parser.add_argument("--pred_b", type=str, required=True, help="e.g. runs/preds_fcfno.pt")
    parser.add_argument("--label_a", type=str, default="FNO")
    parser.add_argument("--label_b", type=str, default="FC-FNO")
    parser.add_argument("--n_show", type=int, default=3)
    parser.add_argument("--times", type=str, default="0,5,10,15", help="comma-separated time indices")
    parser.add_argument("--out", type=str, default="", help="optional path to save figure (png)")
    args = parser.parse_args()

    A = load_preds(args.pred_a)
    B = load_preds(args.pred_b)

    pred_a = A["pred"]  # [M,T,N]
    pred_b = B["pred"]  # [M,T,N]
    true = A["true"]
    x = A["x"]
    t = A["t"]

    M, Tn, N = true.shape
    dx = float((x[1] - x[0]).item())

    time_ids = [int(s) for s in args.times.split(",") if s.strip() != ""]
    time_ids = [tid for tid in time_ids if 0 <= tid < Tn]
    if len(time_ids) == 0:
        time_ids = [0, Tn // 2, Tn - 1]

    n_show = min(args.n_show, M)

    # ---- Summary metrics over shown batch subset ----
    tv_true = total_variation(true)                 # [M,T]
    tv_a = total_variation(pred_a)
    tv_b = total_variation(pred_b)

    ent_a = [entropy_violation_proxy(pred_a[i], x, t) for i in range(n_show)]
    ent_b = [entropy_violation_proxy(pred_b[i], x, t) for i in range(n_show)]
    ent_t = [entropy_violation_proxy(true[i], x, t) for i in range(n_show)]

    print(f"[Entropy proxy] true(mean)={sum(ent_t)/n_show:.3e} "
          f"{args.label_a}(mean)={sum(ent_a)/n_show:.3e} "
          f"{args.label_b}(mean)={sum(ent_b)/n_show:.3e}")

    # ---- Plot: curves at selected times ----
    fig = plt.figure(figsize=(14, 4 * n_show))
    for i in range(n_show):
        for j, tid in enumerate(time_ids):
            ax = plt.subplot(n_show, len(time_ids), i * len(time_ids) + j + 1)
            ax.plot(x.numpy(), true[i, tid].numpy(), label="true")
            ax.plot(x.numpy(), pred_a[i, tid].numpy(), label=args.label_a)
            ax.plot(x.numpy(), pred_b[i, tid].numpy(), label=args.label_b)

            # overshoot metric printed in title
            os_a = overshoot_near_shock(pred_a[i], true[i], dx=dx)[tid].item()
            os_b = overshoot_near_shock(pred_b[i], true[i], dx=dx)[tid].item()
            ax.set_title(f"sample {i}, t_idx={tid} | overshoot: {args.label_a}={os_a:.2e}, {args.label_b}={os_b:.2e}")
            ax.grid(True)
            if i == 0 and j == 0:
                ax.legend()

    plt.tight_layout()

    # ---- Plot: TV over time for sample 0 ----
    plt.figure(figsize=(10, 4))
    plt.plot(t.numpy(), tv_true[0].numpy(), label="true")
    plt.plot(t.numpy(), tv_a[0].numpy(), label=args.label_a)
    plt.plot(t.numpy(), tv_b[0].numpy(), label=args.label_b)
    plt.title("Total Variation vs time (sample 0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.savefig(args.out, dpi=150)
        print(f"[OK] saved figure to {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
