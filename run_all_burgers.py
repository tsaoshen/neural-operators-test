# run_all_burgers.py
# Extended version:
#  (1) Viscosity sweep: train+eval for multiple nu values and print an aggregate table
#  (2) Full-test metrics: shock-speed error + entropy-violation proxy over the entire test set
#
# Assumes Part-1 datasets contain: u0 [n,N], u [n,T,N], x [N], t [T], nu [n,1]
# Assumes Part-2/3 models exist: models/fno1d.py, models/flux_corrector.py, models/fc_fno.py
# Optional losses: models/losses_burgers.py
#
# Example (single dataset):
#   python run_all_burgers.py --train_pt ... --val_pt ... --test_pt ... --out_dir runs/all --name burgers
#
# Example (sweep):
#   python run_all_burgers.py --sweep_nu 1e-2,1e-3 \
#     --train_tmpl data/cached/burgers_nu{nu}_train.pt \
#     --val_tmpl   data/cached/burgers_nu{nu}_val.pt \
#     --test_tmpl  data/cached/burgers_nu{nu}_test.pt \
#     --out_dir runs/sweep --name burgers

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.burgers_dataset import BurgersOperatorDataset, burgers_collate
from models.fno1d import FNO1d
from models.flux_corrector import FluxCorrector1d
from models.fc_fno import FCFNO1d
from models.losses_burgers import BurgersWeakResidualLoss, BurgersEntropyInequalityLoss


# -------------------------
# Input construction
# -------------------------
def make_input(u0, x, nu):
    """
    u0: [B,N], x:[N], nu:[B,1]
    -> x_in: [B,3,N] channels [u0, x, nu]
    """
    B, N = u0.shape
    x_chan = x[None, :].expand(B, N)
    nu_chan = nu[:, 0][:, None].expand(B, N)
    return torch.stack([u0, x_chan, nu_chan], dim=1)


def total_variation(u: torch.Tensor) -> torch.Tensor:
    # u: [..., N]
    return torch.sum(torch.abs(u[..., 1:] - u[..., :-1]), dim=-1)


# -------------------------
# Shock metrics
# -------------------------
def shock_indicator(u: torch.Tensor, dx: float) -> torch.Tensor:
    """
    u: [B,T,N]
    returns s: [B,T,N] ~ |du/dx|
    """
    du = torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)
    return torch.abs(du) / (2.0 * dx)


def shock_location_idx(u: torch.Tensor, dx: float) -> torch.Tensor:
    """
    u: [B,T,N]
    returns idx: [B,T] argmax_x |du/dx|
    """
    s = shock_indicator(u, dx)
    return torch.argmax(s, dim=-1)


def unwrap_circular_idx(idx: torch.Tensor, N: int) -> torch.Tensor:
    """
    idx: [B,T] integer indices in [0,N-1]
    returns unwrapped idx_u: [B,T] float, continuous-ish by minimal wrap step per time.
    """
    idx = idx.to(torch.float32)
    B, T = idx.shape
    out = idx.clone()
    for t in range(1, T):
        prev = out[:, t - 1]
        cur = out[:, t]
        # choose delta among {cur-prev, cur+N-prev, cur-N-prev} with minimal abs
        d0 = cur - prev
        d1 = (cur + N) - prev
        d2 = (cur - N) - prev
        # pick min abs
        abs_stack = torch.stack([d0.abs(), d1.abs(), d2.abs()], dim=0)  # [3,B]
        choice = torch.argmin(abs_stack, dim=0)  # [B]
        delta = torch.where(choice == 0, d0, torch.where(choice == 1, d1, d2))
        out[:, t] = prev + delta
    return out


def shock_speed(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, min_strength: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    u: [B,T,N], x:[N], t:[T]
    returns:
      speed: [B,T-1] in physical units (dx/dt)
      valid: [B] bool indicating whether a strong shock exists (based on indicator max)
    """
    B, Tn, N = u.shape
    dx = float((x[1] - x[0]).item())
    dt = float((t[1] - t[0]).item()) if Tn > 1 else 1.0

    s = shock_indicator(u, dx)                    # [B,T,N]
    smax = torch.amax(s, dim=(1, 2))              # [B]
    valid = smax > float(min_strength)

    idx = shock_location_idx(u, dx)               # [B,T]
    idx_u = unwrap_circular_idx(idx, N)           # [B,T]
    x_u = idx_u * dx                              # [B,T]
    v = (x_u[:, 1:] - x_u[:, :-1]) / dt           # [B,T-1]
    return v, valid


# -------------------------
# Entropy proxy over full test set (batchable)
# -------------------------
def entropy_proxy_batch(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, n_k: int = 16, n_bumps: int = 8) -> torch.Tensor:
    """
    Vectorized proxy for Kružkov entropy inequality violations.
    Returns per-sample score: [B]
    score = mean_{m,k} relu(E_{m,k})  where
      E_{m,k} = ∫∫ (eta_k(u) phi_t + q_k(u) phi_x) dx dt
    using nonnegative Gaussian bump tests phi_m.

    u: [B,T,N], x:[N], t:[T]
    """
    device, dtype = u.device, u.dtype
    B, Tn, N = u.shape
    dx = float((x[1] - x[0]).item())
    dt = float((t[1] - t[0]).item()) if Tn > 1 else 1.0
    Lx = float(x[-1].item() + dx)

    # k grid per batch based on global min/max in u (detached for stability)
    umin = u.detach().amin()
    umax = u.detach().amax()
    k = torch.linspace(umin, umax, steps=n_k, device=device, dtype=dtype)  # [K]
    K = k.numel()

    # build bumps phi_m(x,t) >= 0
    # (same for all samples)
    phis = []
    for i in range(n_bumps):
        xc = (i + 0.5) * (Lx / n_bumps)
        tc = (i + 0.5) * (float(t[-1].item()) / max(n_bumps, 1)) if Tn > 1 else 0.0
        sigx = 0.12 * Lx
        sigt = 0.25 * float(t[-1].item() + 1e-12)

        bx = torch.exp(-0.5 * ((x - xc) / sigx) ** 2)  # [N]
        bt = torch.exp(-0.5 * ((t - tc) / max(sigt, 1e-6)) ** 2)  # [T]
        phis.append(bt[:, None] * bx[None, :])

    phi = torch.stack(phis, dim=0).to(device=device, dtype=dtype)  # [M,T,N]
    M = phi.shape[0]

    # derivatives (finite diff)
    phi_x = (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1)) / (2.0 * dx)  # [M,T,N]
    phi_t = torch.zeros_like(phi)
    if Tn > 1:
        phi_t[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dt)
        phi_t[:, 0] = (phi[:, 1] - phi[:, 0]) / dt
        phi_t[:, -1] = (phi[:, -1] - phi[:, -2]) / dt

    # eta_k, q_k
    # u: [B,T,N]
    # broadcast to [B,T,N,K]
    eta = torch.abs(u[..., None] - k[None, None, None, :])  # [B,T,N,K]
    f_u = 0.5 * u * u                                        # [B,T,N]
    f_k = 0.5 * k * k                                        # [K]
    sgn = torch.sign(u[..., None] - k[None, None, None, :])  # [B,T,N,K]
    q = sgn * (f_u[..., None] - f_k[None, None, None, :])    # [B,T,N,K]

    # apply tests: [M,T,N] -> [1,M,T,N,1]
    phi_t_b = phi_t[None, :, :, :, None]  # [1,M,T,N,1]
    phi_x_b = phi_x[None, :, :, :, None]  # [1,M,T,N,1]
    eta_b = eta[:, None, :, :, :]         # [B,M,T,N,K]
    q_b = q[:, None, :, :, :]             # [B,M,T,N,K]

    integrand = eta_b * phi_t_b + q_b * phi_x_b               # [B,M,T,N,K]
    E = torch.sum(integrand, dim=(2, 3)) * (dx * dt)          # [B,M,K]
    score = torch.mean(F.relu(E), dim=(1, 2))                 # [B]
    return score


# -------------------------
# Training / eval
# -------------------------
@torch.no_grad()
def evaluate_mse(model, loader, device):
    model.eval()
    mse = 0.0
    n = 0
    for batch in loader:
        u0 = batch["u0"].to(device)
        u_true = batch["u"].to(device)
        x = batch["x"].to(device)
        nu = batch["nu"].to(device)
        dx = float((x[1] - x[0]).item())

        x_in = make_input(u0, x, nu)
        y = model(x_in, dx) if isinstance(model, FCFNO1d) else model(x_in)

        loss = torch.mean((y - u_true) ** 2)
        mse += loss.item() * u0.shape[0]
        n += u0.shape[0]
    return mse / max(n, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    save_path,
    lambda_weak=0.0,
    lambda_ent=0.0,
    weak_loss=None,
    ent_loss=None,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        running_sup = 0.0
        running_w = 0.0
        running_e = 0.0
        n = 0

        for batch in train_loader:
            u0 = batch["u0"].to(device)
            u_true = batch["u"].to(device)
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            nu = batch["nu"].to(device)

            dx = float((x[1] - x[0]).item())
            x_in = make_input(u0, x, nu)

            y = model(x_in, dx) if isinstance(model, FCFNO1d) else model(x_in)

            sup = torch.mean((y - u_true) ** 2)
            loss = sup

            wv = torch.tensor(0.0, device=device)
            ev = torch.tensor(0.0, device=device)

            if lambda_weak > 0.0:
                wv = weak_loss(y, x, t, nu)
                loss = loss + lambda_weak * wv
            if lambda_ent > 0.0:
                ev = ent_loss(y, x, t)
                loss = loss + lambda_ent * ev

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            B = u0.shape[0]
            running += loss.item() * B
            running_sup += sup.item() * B
            running_w += wv.item() * B
            running_e += ev.item() * B
            n += B

        scheduler.step()
        train_loss = running / max(n, 1)
        train_sup = running_sup / max(n, 1)
        train_w = running_w / max(n, 1)
        train_e = running_e / max(n, 1)

        val_mse = evaluate_mse(model, val_loader, device)

        if val_mse < best_val:
            best_val = val_mse
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_mse": best_val,
                    "lambda_weak": lambda_weak,
                    "lambda_ent": lambda_ent,
                },
                save_path,
            )

        print(
            f"Epoch {ep:03d} | train_loss={train_loss:.4e} "
            f"(sup={train_sup:.4e}, weak={train_w:.4e}, ent={train_e:.4e}) "
            f"| val_mse={val_mse:.4e} | best={best_val:.4e}"
        )

    return best_val


@torch.no_grad()
def save_predictions(model, loader, device, save_path, max_store=16):
    """
    Saves compact preds file:
      { pred, true, u0, x, t, mse }
    """
    model.eval()
    preds, trues, u0s = [], [], []
    mse = 0.0
    n = 0
    x_ref = None
    t_ref = None

    stored = 0
    for batch in loader:
        u0 = batch["u0"].to(device)
        u_true = batch["u"].to(device)
        x = batch["x"].to(device)
        nu = batch["nu"].to(device)
        dx = float((x[1] - x[0]).item())

        x_in = make_input(u0, x, nu)
        y = model(x_in, dx) if isinstance(model, FCFNO1d) else model(x_in)

        loss = torch.mean((y - u_true) ** 2)
        mse += loss.item() * u0.shape[0]
        n += u0.shape[0]

        if x_ref is None:
            x_ref = batch["x"].detach().cpu()
            t_ref = batch["t"].detach().cpu()

        if stored < max_store:
            take = min(max_store - stored, u0.shape[0])
            preds.append(y[:take].detach().cpu())
            trues.append(u_true[:take].detach().cpu())
            u0s.append(u0[:take].detach().cpu())
            stored += take

    mse = mse / max(n, 1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "pred": torch.cat(preds, dim=0) if preds else torch.empty(0),
            "true": torch.cat(trues, dim=0) if trues else torch.empty(0),
            "u0": torch.cat(u0s, dim=0) if u0s else torch.empty(0),
            "x": x_ref,
            "t": t_ref,
            "mse": mse,
        },
        save_path,
    )
    print(f"[OK] saved preds: {save_path} | mse={mse:.4e} | stored={stored}")
    return mse


@torch.no_grad()
def full_test_metrics(model, loader, device, entropy_k=16, entropy_bumps=8, shock_min_strength=0.0) -> Dict[str, float]:
    """
    Computes metrics over the full test set:
      - test_mse
      - shock_speed_mae (pred vs true) averaged over samples/time (valid shocks only)
      - entropy_proxy_mean (mean over samples)
    """
    model.eval()
    mse_sum = 0.0
    n_sum = 0

    speed_err_sum = 0.0
    speed_cnt = 0

    ent_sum = 0.0
    ent_cnt = 0

    for batch in loader:
        u0 = batch["u0"].to(device)
        u_true = batch["u"].to(device)     # [B,T,N]
        x = batch["x"].to(device)
        t = batch["t"].to(device)
        nu = batch["nu"].to(device)
        dx = float((x[1] - x[0]).item())

        x_in = make_input(u0, x, nu)
        u_pred = model(x_in, dx) if isinstance(model, FCFNO1d) else model(x_in)

        # MSE
        loss = torch.mean((u_pred - u_true) ** 2)
        B = u0.shape[0]
        mse_sum += loss.item() * B
        n_sum += B

        # Shock-speed error (true vs pred), valid if true has strong enough front
        v_true, valid = shock_speed(u_true, x, t, min_strength=shock_min_strength)  # [B,T-1], [B]
        v_pred, _ = shock_speed(u_pred, x, t, min_strength=shock_min_strength)

        if valid.any():
            vte = torch.abs(v_pred[valid] - v_true[valid])  # [Bv,T-1]
            speed_err_sum += vte.mean().item() * int(valid.sum().item())
            speed_cnt += int(valid.sum().item())

        # Entropy proxy mean
        ent = entropy_proxy_batch(u_pred, x, t, n_k=entropy_k, n_bumps=entropy_bumps)  # [B]
        ent_sum += ent.sum().item()
        ent_cnt += B

    out = {
        "test_mse": mse_sum / max(n_sum, 1),
        "shock_speed_mae": (speed_err_sum / max(speed_cnt, 1)) if speed_cnt > 0 else float("nan"),
        "entropy_proxy_mean": ent_sum / max(ent_cnt, 1),
        "shock_valid_frac": (speed_cnt / max(n_sum, 1)) if n_sum > 0 else 0.0,
    }
    return out


def make_compare_plot(pred_a_path, pred_b_path, out_dir, label_a="FNO", label_b="FC-FNO", times="0,5,10,15", n_show=3):
    A = torch.load(pred_a_path, map_location="cpu")
    B = torch.load(pred_b_path, map_location="cpu")

    pred_a = A["pred"]
    pred_b = B["pred"]
    true = A["true"]
    x = A["x"]
    t = A["t"]

    if pred_a.numel() == 0 or pred_b.numel() == 0:
        print("[WARN] No stored predictions to plot.")
        return

    M, Tn, N = true.shape
    n_show = min(n_show, M)
    time_ids = [int(s) for s in times.split(",") if s.strip() != ""]
    time_ids = [tid for tid in time_ids if 0 <= tid < Tn]
    if len(time_ids) == 0:
        time_ids = [0, Tn // 2, Tn - 1]

    dx = float((x[1] - x[0]).item())

    # Curves
    fig1 = plt.figure(figsize=(14, 4 * n_show))
    for i in range(n_show):
        for j, tid in enumerate(time_ids):
            ax = plt.subplot(n_show, len(time_ids), i * len(time_ids) + j + 1)
            ax.plot(x.numpy(), true[i, tid].numpy(), label="true")
            ax.plot(x.numpy(), pred_a[i, tid].numpy(), label=label_a)
            ax.plot(x.numpy(), pred_b[i, tid].numpy(), label=label_b)

            # overshoot around shock (using true shock location)
            # quick per-time overshoot:
            ish = int(torch.argmax(torch.abs(torch.roll(true[i, tid], -1) - torch.roll(true[i, tid], 1))).item())
            r = 8
            lo = (ish - r) % N
            hi = (ish + r) % N
            if lo <= hi:
                idx = torch.arange(lo, hi + 1)
            else:
                idx = torch.cat([torch.arange(lo, N), torch.arange(0, hi + 1)])
            os_a = torch.max(torch.abs(pred_a[i, tid, idx] - true[i, tid, idx])).item()
            os_b = torch.max(torch.abs(pred_b[i, tid, idx] - true[i, tid, idx])).item()

            ax.set_title(f"sample {i}, t_idx={tid} | overshoot: {label_a}={os_a:.2e}, {label_b}={os_b:.2e}")
            ax.grid(True)
            if i == 0 and j == 0:
                ax.legend()
    plt.tight_layout()

    # TV over time (sample 0)
    tv_true = total_variation(true[0])
    tv_a = total_variation(pred_a[0])
    tv_b = total_variation(pred_b[0])

    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(t.numpy(), tv_true.numpy(), label="true")
    plt.plot(t.numpy(), tv_a.numpy(), label=label_a)
    plt.plot(t.numpy(), tv_b.numpy(), label=label_b)
    plt.title("Total Variation vs time (sample 0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fig1_path = os.path.join(out_dir, "compare_curves.png")
    fig2_path = os.path.join(out_dir, "compare_tv.png")
    fig1.savefig(fig1_path, dpi=150)
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig1)
    plt.close(fig2)
    print(f"[OK] saved plots:\n  {fig1_path}\n  {fig2_path}")


# -------------------------
# Sweep wiring
# -------------------------
def parse_list_csv(s: str) -> List[str]:
    return [z.strip() for z in s.split(",") if z.strip()]


def format_template(tmpl: str, nu_str: str) -> str:
    # nu_str is exactly what user provides in sweep list (e.g. "1e-3")
    return tmpl.format(nu=nu_str)


@dataclass
class RunResult:
    nu: str
    fno: Dict[str, float]
    fcfno: Dict[str, float]
    out_dir: str


def run_once(
    train_pt: str,
    val_pt: str,
    test_pt: str,
    out_dir: str,
    name: str,
    device: torch.device,
    batch: int,
    epochs: int,
    lr: float,
    width: int,
    modes: int,
    layers: int,
    corr_hidden: int,
    corr_layers: int,
    corr_kernel: int,
    lambda_weak: float,
    lambda_ent: float,
    weak_pmax: int,
    weak_qmax: int,
    ent_k: int,
    ent_bumps: int,
    plot_times: str,
    plot_nshow: int,
    shock_min_strength: float,
) -> RunResult:
    # Datasets/loaders
    train_ds = BurgersOperatorDataset(train_pt, device=None)
    val_ds = BurgersOperatorDataset(val_pt, device=None)
    test_ds = BurgersOperatorDataset(test_pt, device=None)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, collate_fn=burgers_collate)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=burgers_collate)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=burgers_collate)

    T = train_ds.u.shape[1]
    in_channels = 3

    # Optional losses
    weak_loss = None
    ent_loss = None
    if lambda_weak > 0.0:
        weak_loss = BurgersWeakResidualLoss(p_max=weak_pmax, q_max=weak_qmax, use_cos=True).to(device)
        print(f"[Loss] weak enabled: lambda_weak={lambda_weak}")
    if lambda_ent > 0.0:
        ent_loss = BurgersEntropyInequalityLoss(n_k=ent_k, n_bumps=ent_bumps).to(device)
        print(f"[Loss] entropy enabled: lambda_ent={lambda_ent}")

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # 1) Train baseline FNO
    # -------------------------
    fno = FNO1d(in_channels=in_channels, out_channels=T, width=width, modes=modes, n_layers=layers, activation="gelu").to(device)
    fno_ckpt = os.path.join(out_dir, f"{name}_fno.pt")

    print("\n==============================")
    print(f"[1/4] Training baseline FNO | out={out_dir}")
    print("==============================")
    train_model(
        model=fno,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        save_path=fno_ckpt,
        lambda_weak=lambda_weak,
        lambda_ent=lambda_ent,
        weak_loss=weak_loss,
        ent_loss=ent_loss,
    )

    # Load best for eval
    fno_best = FNO1d(in_channels=in_channels, out_channels=T, width=width, modes=modes, n_layers=layers, activation="gelu").to(device)
    fno_best.load_state_dict(torch.load(fno_ckpt, map_location="cpu")["model"])
    fno_best.eval()

    # -------------------------
    # 2) Train FC-FNO
    # -------------------------
    fno2 = FNO1d(in_channels=in_channels, out_channels=T, width=width, modes=modes, n_layers=layers, activation="gelu").to(device)
    corr = FluxCorrector1d(hidden=corr_hidden, n_layers=corr_layers, kernel_size=corr_kernel).to(device)
    fcfno = FCFNO1d(fno=fno2, corrector=corr).to(device)
    fcfno_ckpt = os.path.join(out_dir, f"{name}_fcfno.pt")

    print("\n==============================")
    print("[2/4] Training FC-FNO (with flux corrector)")
    print("==============================")
    train_model(
        model=fcfno,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        save_path=fcfno_ckpt,
        lambda_weak=lambda_weak,
        lambda_ent=lambda_ent,
        weak_loss=weak_loss,
        ent_loss=ent_loss,
    )

    # Load best for eval
    fno3 = FNO1d(in_channels=in_channels, out_channels=T, width=width, modes=modes, n_layers=layers, activation="gelu").to(device)
    corr2 = FluxCorrector1d(hidden=corr_hidden, n_layers=corr_layers, kernel_size=corr_kernel).to(device)
    fcfno_best = FCFNO1d(fno=fno3, corrector=corr2).to(device)
    fcfno_best.load_state_dict(torch.load(fcfno_ckpt, map_location="cpu")["model"])
    fcfno_best.eval()

    # -------------------------
    # 3) Evaluate + save preds
    # -------------------------
    print("\n==============================")
    print("[3/4] Evaluating and saving predictions")
    print("==============================")
    preds_fno_path = os.path.join(out_dir, "preds_fno.pt")
    preds_fcfno_path = os.path.join(out_dir, "preds_fcfno.pt")

    save_predictions(fno_best, test_loader, device, preds_fno_path, max_store=16)
    save_predictions(fcfno_best, test_loader, device, preds_fcfno_path, max_store=16)

    # Full-test metrics (this is what you requested)
    print("\n==============================")
    print("[4/4] Full-test structured metrics")
    print("==============================")
    fno_metrics = full_test_metrics(
        fno_best, test_loader, device,
        entropy_k=ent_k, entropy_bumps=ent_bumps,
        shock_min_strength=shock_min_strength,
    )
    fcfno_metrics = full_test_metrics(
        fcfno_best, test_loader, device,
        entropy_k=ent_k, entropy_bumps=ent_bumps,
        shock_min_strength=shock_min_strength,
    )

    print(f"[FNO]   {fno_metrics}")
    print(f"[FCFNO] {fcfno_metrics}")

    # Plots (bundle)
    make_compare_plot(
        preds_fno_path,
        preds_fcfno_path,
        out_dir=out_dir,
        label_a="FNO",
        label_b="FC-FNO",
        times=plot_times,
        n_show=plot_nshow,
    )

    # Report viscosity from file if present
    nu_val = "unknown"
    try:
        data_meta = torch.load(test_pt, map_location="cpu")
        if "nu" in data_meta:
            nu_val = str(float(torch.mean(data_meta["nu"]).item()))
    except Exception:
        pass

    return RunResult(nu=nu_val, fno=fno_metrics, fcfno=fcfno_metrics, out_dir=out_dir)


def print_results_table(results: List[RunResult]):
    # Simple fixed-width table
    headers = [
        "nu",
        "FNO MSE", "FNO shockMAE", "FNO ent", "FNO shockFrac",
        "FC MSE", "FC shockMAE", "FC ent", "FC shockFrac",
        "out_dir",
    ]
    rows = []
    for r in results:
        rows.append([
            r.nu,
            f"{r.fno['test_mse']:.3e}",
            f"{r.fno['shock_speed_mae']:.3e}" if not math.isnan(r.fno["shock_speed_mae"]) else "nan",
            f"{r.fno['entropy_proxy_mean']:.3e}",
            f"{r.fno['shock_valid_frac']:.3f}",
            f"{r.fcfno['test_mse']:.3e}",
            f"{r.fcfno['shock_speed_mae']:.3e}" if not math.isnan(r.fcfno["shock_speed_mae"]) else "nan",
            f"{r.fcfno['entropy_proxy_mean']:.3e}",
            f"{r.fcfno['shock_valid_frac']:.3f}",
            r.out_dir,
        ])

    # compute column widths
    colw = [len(h) for h in headers]
    for row in rows:
        for j, v in enumerate(row):
            colw[j] = max(colw[j], len(str(v)))

    def fmt_row(vals):
        return " | ".join(str(v).ljust(colw[j]) for j, v in enumerate(vals))

    print("\n=== Sweep Results ===")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in colw))
    for row in rows:
        print(fmt_row(row))


def main():
    parser = argparse.ArgumentParser()

    # Single-run inputs
    parser.add_argument("--train_pt", type=str, default="")
    parser.add_argument("--val_pt", type=str, default="")
    parser.add_argument("--test_pt", type=str, default="")

    # Sweep inputs
    parser.add_argument("--sweep_nu", type=str, default="", help="comma-separated, e.g. 1e-2,1e-3")
    parser.add_argument("--train_tmpl", type=str, default="", help="e.g. data/cached/burgers_nu{nu}_train.pt")
    parser.add_argument("--val_tmpl", type=str, default="", help="e.g. data/cached/burgers_nu{nu}_val.pt")
    parser.add_argument("--test_tmpl", type=str, default="", help="e.g. data/cached/burgers_nu{nu}_test.pt")

    # Compute
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)

    # FNO params
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)

    # Corrector params
    parser.add_argument("--corr_hidden", type=int, default=32)
    parser.add_argument("--corr_layers", type=int, default=3)
    parser.add_argument("--corr_kernel", type=int, default=5)

    # Weak/entropy loss params (applied to both models if enabled)
    parser.add_argument("--lambda_weak", type=float, default=0.0)
    parser.add_argument("--lambda_ent", type=float, default=0.0)
    parser.add_argument("--weak_pmax", type=int, default=6)
    parser.add_argument("--weak_qmax", type=int, default=6)
    parser.add_argument("--ent_k", type=int, default=16)
    parser.add_argument("--ent_bumps", type=int, default=8)

    # Metrics knobs
    parser.add_argument(
        "--shock_min_strength",
        type=float,
        default=0.0,
        help="minimum max(|du/dx|) in true solution to consider shock-speed metric valid (increase to ignore rarefactions)",
    )

    # Outputs
    parser.add_argument("--out_dir", type=str, default="runs/all")
    parser.add_argument("--name", type=str, default="burgers")
    parser.add_argument("--plot_times", type=str, default="0,5,10,15")
    parser.add_argument("--plot_nshow", type=int, default=3)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Decide sweep or single
    sweep_list = parse_list_csv(args.sweep_nu) if args.sweep_nu else []
    results: List[RunResult] = []

    if sweep_list:
        assert args.train_tmpl and args.val_tmpl and args.test_tmpl, (
            "For sweep, provide --train_tmpl --val_tmpl --test_tmpl (with {nu})."
        )

        for nu_str in sweep_list:
            train_pt = format_template(args.train_tmpl, nu_str)
            val_pt = format_template(args.val_tmpl, nu_str)
            test_pt = format_template(args.test_tmpl, nu_str)

            run_dir = os.path.join(args.out_dir, f"nu_{nu_str}")
            print("\n\n============================================================")
            print(f"SWEEP nu={nu_str} | train={train_pt}")
            print("============================================================")

            res = run_once(
                train_pt=train_pt,
                val_pt=val_pt,
                test_pt=test_pt,
                out_dir=run_dir,
                name=args.name,
                device=device,
                batch=args.batch,
                epochs=args.epochs,
                lr=args.lr,
                width=args.width,
                modes=args.modes,
                layers=args.layers,
                corr_hidden=args.corr_hidden,
                corr_layers=args.corr_layers,
                corr_kernel=args.corr_kernel,
                lambda_weak=args.lambda_weak,
                lambda_ent=args.lambda_ent,
                weak_pmax=args.weak_pmax,
                weak_qmax=args.weak_qmax,
                ent_k=args.ent_k,
                ent_bumps=args.ent_bumps,
                plot_times=args.plot_times,
                plot_nshow=args.plot_nshow,
                shock_min_strength=args.shock_min_strength,
            )
            # override nu label with sweep string for table clarity
            res.nu = nu_str
            results.append(res)

        print_results_table(results)

    else:
        assert args.train_pt and args.val_pt and args.test_pt, (
            "Provide --train_pt --val_pt --test_pt for a single run, "
            "or use --sweep_nu with templates."
        )
        res = run_once(
            train_pt=args.train_pt,
            val_pt=args.val_pt,
            test_pt=args.test_pt,
            out_dir=args.out_dir,
            name=args.name,
            device=device,
            batch=args.batch,
            epochs=args.epochs,
            lr=args.lr,
            width=args.width,
            modes=args.modes,
            layers=args.layers,
            corr_hidden=args.corr_hidden,
            corr_layers=args.corr_layers,
            corr_kernel=args.corr_kernel,
            lambda_weak=args.lambda_weak,
            lambda_ent=args.lambda_ent,
            weak_pmax=args.weak_pmax,
            weak_qmax=args.weak_qmax,
            ent_k=args.ent_k,
            ent_bumps=args.ent_bumps,
            plot_times=args.plot_times,
            plot_nshow=args.plot_nshow,
            shock_min_strength=args.shock_min_strength,
        )
        results = [res]
        print_results_table(results)


if __name__ == "__main__":
    main()
