# data/inspect_ic.py
"""
Inspect / validate initial-condition sampling for
  u0 ~ N(0, 625(-Δ + 25 I)^(-2))  on 1D periodic domain.

What it does:
- Samples many u0 on the reference grid (N_ref from config).
- Computes rFFT coefficients U_k (torch.fft.rfft with norm=None).
- Estimates empirical Var(U_k) across samples.
- Compares against the target discrete variance:
      Var(U_k) = N^2 * 625 / (( (2π k/L)^2 + 25 )^2)
  where k are the rFFT frequencies (angular wavenumber 2πk/L).

Outputs:
- A log-log plot of empirical vs target variance over modes.
- A plot of a few sampled u0 curves.
"""

import os
import math
import yaml
import argparse
import torch
import matplotlib.pyplot as plt

from data.utils.grids import make_grid_1d
from data.ic.sample_ic import sample_u0_batch

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/burgers.yaml")
    parser.add_argument("--n", type=int, default=4096, help="number of IC samples")
    parser.add_argument("--family", type=str, default="smooth", help="family to sample: smooth|near_front|riemann")
    parser.add_argument("--plot_samples", type=int, default=6, help="how many u0 curves to plot")
    parser.add_argument("--max_mode", type=int, default=256, help="max Fourier mode to display in variance plot")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cpu"))
    dtype = DTYPE_MAP[cfg.get("dtype", "float32")]

    # Use reference grid for analysis
    N = int(cfg["solver"]["N_ref"])
    L = float(cfg["domain"]["L"])

    x = make_grid_1d(N, L=L, device=device, dtype=dtype)  # [N]

    # Build a minimal dataset config to sample ONLY the requested family
    family = args.family
    if family not in ["smooth", "near_front", "riemann"]:
        raise ValueError(f"--family must be one of smooth|near_front|riemann, got {family}")

    families = [family]
    mix_probs = [1.0]

    # family_cfg is expected to be cfg["dataset"] by our sampler
    family_cfg = cfg["dataset"]

    # Sample
    n_samples = int(args.n)
    u0, meta = sample_u0_batch(
        n_samples=n_samples,
        x=x,
        families=families,
        mix_probs=mix_probs,
        family_cfg=family_cfg,
        device=device,
        dtype=dtype,
    )  # [n, N]

    # rFFT coefficients (norm=None / backward, default)
    # U: [n, N//2+1] complex
    U = torch.fft.rfft(u0, dim=-1)

    # Empirical variance across samples for each mode
    # For complex coefficients: Var(U_k) = E[|U_k|^2] - |E[U_k]|^2
    mean_U = U.mean(dim=0)
    emp_var = (U.abs() ** 2).mean(dim=0) - (mean_U.abs() ** 2)
    emp_var = emp_var.real  # [nfreq]

    # Target variance for discrete FFT coefficients:
    # Var(U_k) = N^2 * 625 / (( (2π k/L)^2 + alpha )^2)
    # with alpha=25, scale=625 by default; read from config if present
    # For "near_front", the base uses same alpha/scale but with additional tanh ramp (so variance won't match exactly).
    alpha = float(family_cfg.get(family, {}).get("alpha", family_cfg.get("smooth", {}).get("alpha", 25.0)))
    scale = float(family_cfg.get(family, {}).get("scale", family_cfg.get("smooth", {}).get("scale", 625.0)))

    # rfftfreq returns cycles per unit; multiply by 2π to get angular wavenumber
    k_cyc = torch.fft.rfftfreq(N, d=L / N).to(device=device, dtype=dtype)  # cycles per unit length
    k_ang = 2.0 * math.pi * k_cyc  # angular wavenumber: 2π k / L
    lam = k_ang * k_ang
    target_var = (N * N) * (scale / ((lam + alpha) ** 2))  # [nfreq]

    # Move to CPU for plotting
    emp_var_cpu = emp_var.detach().cpu()
    target_var_cpu = target_var.detach().cpu()
    k_idx = torch.arange(emp_var_cpu.numel())

    # Limit plotted modes
    max_mode = min(int(args.max_mode), emp_var_cpu.numel() - 1)
    k_plot = k_idx[: max_mode + 1]
    emp_plot = emp_var_cpu[: max_mode + 1]
    tgt_plot = target_var_cpu[: max_mode + 1]

    # ---- Plot 1: variance vs mode ----
    plt.figure()
    plt.loglog(k_plot.numpy() + 1e-12, emp_plot.numpy() + 1e-30)  # avoid log(0)
    plt.loglog(k_plot.numpy() + 1e-12, tgt_plot.numpy() + 1e-30)
    plt.xlabel("Fourier mode index k (rFFT)")
    plt.ylabel("Var(U_k)  (discrete FFT coeff variance)")
    plt.title(f"Empirical vs target Var(U_k) for family='{family}' (n={n_samples}, N={N})")
    plt.legend(["empirical", "target"])
    plt.grid(True, which="both")

    # ---- Plot 2: a few samples ----
    plt.figure()
    ns = min(int(args.plot_samples), n_samples)
    x_cpu = x.detach().cpu()
    for i in range(ns):
        plt.plot(x_cpu.numpy(), u0[i].detach().cpu().numpy())
    plt.xlabel("x")
    plt.ylabel("u0(x)")
    plt.title(f"Sampled initial conditions (family='{family}')")
    plt.grid(True)

    plt.show()

    # Optional numeric check: relative error over a mode band
    # (ignore k=0, where empirical mean might deviate slightly at finite samples)
    band = slice(1, max_mode + 1)
    rel_err = (emp_plot[band] - tgt_plot[band]).abs() / (tgt_plot[band].abs() + 1e-30)
    print(f"[Stats] median rel error (modes 1..{max_mode}): {rel_err.median().item():.3e}")
    print(f"[Stats] mean   rel error (modes 1..{max_mode}): {rel_err.mean().item():.3e}")


if __name__ == "__main__":
    main()
