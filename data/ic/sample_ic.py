import math
import torch
from typing import Dict, List, Tuple, Any


def _sample_gaussian_measure(
    u_shape,
    x: torch.Tensor,
    cfg: Dict[str, Any],
    device,
    dtype,
) -> torch.Tensor:
    """
    Sample u0 ~ N(0, 625 (-Δ + 25 I)^(-2)) on 1D periodic domain [0, L).

    Discrete implementation via rFFT:
      For mode k (integer), eigenvalue of -Δ is (2πk)^2.
      Continuous variance for Fourier series coeff: 625 / (( (2πk)^2 + 25 )^2)

    For torch FFT with norm=None ("backward"), irfft does 1/N scaling, so
    we sample discrete Fourier coefficients with variance:
        Var(U_k) = N^2 * 625 / (( (2πk)^2 + 25 )^2)

    Output: real field u0 of shape [n, N].
    """
    n, N = u_shape
    dx = (x[1] - x[0])
    L = float(dx.item() * N)

    # Parameters from the measure; keep defaults matching the user request.
    # mu = N(0, s^2 (-Δ + alpha I)^(-2)), with s^2 = 625 and alpha = 25.
    alpha = float(cfg.get("alpha", 25.0))
    s2 = float(cfg.get("scale", 625.0))  # this is 625 in the statement

    # rFFT frequency indices: k = 0..N/2
    # torch.fft.rfftfreq(N, d=1/N) gives integer k; multiply by 2π/L => 2πk/L.
    k_int = torch.fft.rfftfreq(N, d=L / N).to(device=device, dtype=dtype)  # cycles per unit length
    k_real = 2.0 * math.pi * k_int  # angular wavenumber: 2πk/L

    # Eigenvalues of -Δ: (k_real)^2
    lam = k_real * k_real

    # Discrete FFT coefficient variance (see docstring)
    var = (N * N) * (s2 / ((lam + alpha) ** 2))  # [N//2+1]

    # Sample complex coefficients with conjugate symmetry implicit in rFFT.
    # For k=0 and (if N even) k=N/2, coefficients must be real.
    nfreq = var.numel()
    U = torch.zeros(n, nfreq, device=device, dtype=torch.complex64 if dtype == torch.float32 else torch.complex128)

    # k=0 (real)
    U[:, 0] = torch.randn(n, device=device, dtype=dtype) * torch.sqrt(var[0])

    # k=Nyquist (real) if N even
    if N % 2 == 0:
        nyq = nfreq - 1
        U[:, nyq] = torch.randn(n, device=device, dtype=dtype) * torch.sqrt(var[nyq])

        k_start = 1
        k_end = nyq  # exclusive
    else:
        k_start = 1
        k_end = nfreq  # exclusive

    if k_end > k_start:
        v = var[k_start:k_end]  # [K]
        # For complex normal: Re,Im ~ N(0, v/2)
        std = torch.sqrt(v * 0.5)  # [K]
        re = torch.randn(n, k_end - k_start, device=device, dtype=dtype) * std
        im = torch.randn(n, k_end - k_start, device=device, dtype=dtype) * std
        U[:, k_start:k_end] = torch.complex(re, im)

    # Inverse rFFT to get real field
    u = torch.fft.irfft(U, n=N, dim=-1)  # [n, N]
    return u


def _sample_smooth(u_shape, x: torch.Tensor, cfg: Dict[str, Any], device, dtype):
    """
    UPDATED: default smooth sampler is now the Gaussian measure
      u0 ~ N(0, 625(-Δ + 25 I)^(-2)).
    """
    return _sample_gaussian_measure(u_shape, x, cfg, device, dtype)


def _sample_riemann(u_shape, x: torch.Tensor, cfg: Dict[str, Any], device, dtype):
    n, N = u_shape
    umin = float(cfg.get("umin", -1.0))
    umax = float(cfg.get("umax", 1.0))

    uL = (umin + (umax - umin) * torch.rand(n, 1, device=device, dtype=dtype))
    uR = (umin + (umax - umin) * torch.rand(n, 1, device=device, dtype=dtype))
    x0 = torch.rand(n, 1, device=device, dtype=dtype)

    dx = (x[1] - x[0])
    L = float(dx.item() * N)
    x0 = x0 * L

    u = torch.where(x[None, :] < x0, uL, uR)
    meta = {
        "uL": uL.detach().cpu(),
        "uR": uR.detach().cpu(),
        "x0": x0.detach().cpu(),
        "label": torch.where(uL > uR, torch.ones_like(uL), -torch.ones_like(uL)).cpu(),  # 1=shock, -1=rarefaction
    }
    return u, meta


def _sample_near_front(u_shape, x: torch.Tensor, cfg: Dict[str, Any], device, dtype):
    """
    Smooth base (from Gaussian measure) + steep tanh ramp.
    """
    n, N = u_shape
    base_cfg = {
        "alpha": float(cfg.get("alpha", 25.0)),
        "scale": float(cfg.get("scale", 625.0)),
    }
    u_base = _sample_gaussian_measure(u_shape, x, base_cfg, device, dtype)

    ramp_amp = float(cfg.get("ramp_amp", 1.0))
    eps = float(cfg.get("ramp_eps", 0.01))
    dx = (x[1] - x[0])
    L = float(dx.item() * N)
    x0 = torch.rand(n, 1, device=device, dtype=dtype) * L

    u = u_base + ramp_amp * torch.tanh((x[None, :] - x0) / eps)
    return u


def sample_u0_batch(
    n_samples: int,
    x: torch.Tensor,
    families: List[str],
    mix_probs: List[float],
    family_cfg: Dict[str, Any],
    device,
    dtype,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    N = x.numel()
    u0 = torch.empty(n_samples, N, device=device, dtype=dtype)

    probs = torch.tensor(mix_probs, device=device, dtype=dtype)
    probs = probs / probs.sum()
    fam_ids = torch.multinomial(probs, num_samples=n_samples, replacement=True)

    meta = {"family_id": fam_ids.detach().cpu(), "families": families, "riemann": None}

    riemann_meta = None
    for idx, fam in enumerate(families):
        mask = (fam_ids == idx)
        count = int(mask.sum().item())
        if count == 0:
            continue

        if fam == "smooth":
            # now Gaussian measure sampler
            u0[mask] = _sample_smooth((count, N), x, family_cfg.get("smooth", {}), device, dtype)
        elif fam == "riemann":
            u_part, rmeta = _sample_riemann((count, N), x, family_cfg["riemann"], device, dtype)
            u0[mask] = u_part
            if riemann_meta is None:
                riemann_meta = {k: [] for k in rmeta.keys()}
            for k in rmeta.keys():
                riemann_meta[k].append(rmeta[k])
        elif fam == "near_front":
            u0[mask] = _sample_near_front((count, N), x, family_cfg["near_front"], device, dtype)
        else:
            raise ValueError(f"Unknown family: {fam}")

    if riemann_meta is not None:
        meta["riemann"] = {k: torch.cat(v, dim=0) for k, v in riemann_meta.items()}

    return u0, meta
