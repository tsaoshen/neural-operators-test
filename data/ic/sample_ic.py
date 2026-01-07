import math
import torch
from typing import Dict, List, Tuple, Any


def _sample_smooth(u_shape, x: torch.Tensor, cfg: Dict[str, Any], device, dtype):
    """
    Random smooth Fourier series:
      u0(x) = mean + sum_{k=1..K} a_k cos(2πkx/L) + b_k sin(2πkx/L),
      with decaying amplitude ~ 1/k^p.
    """
    n, N = u_shape
    K = int(cfg["n_modes"])
    amp = float(cfg.get("amp", 1.0))
    p = float(cfg.get("decay_power", 2.0))
    mean = float(cfg.get("mean", 0.0))

    L = float(x[-1].item() + (x[1] - x[0]).item())  # approx L
    # safer: derive L from grid spacing * N
    dx = (x[1] - x[0])
    L = float(dx.item() * N)

    k = torch.arange(1, K + 1, device=device, dtype=dtype)  # [K]
    decay = 1.0 / (k ** p)  # [K]
    a = torch.randn(n, K, device=device, dtype=dtype) * decay * amp
    b = torch.randn(n, K, device=device, dtype=dtype) * decay * amp

    # shape [n, N]
    phase = 2.0 * math.pi * (k[None, :, None] * x[None, None, :] / L)  # [1,K,N]
    u = mean + torch.sum(a[:, :, None] * torch.cos(phase) + b[:, :, None] * torch.sin(phase), dim=1)
    return u


def _sample_riemann(u_shape, x: torch.Tensor, cfg: Dict[str, Any], device, dtype):
    """
    Piecewise-constant Riemann IC with random jump location:
      u0(x) = uL for x < x0, uR otherwise (periodic).
    """
    n, N = u_shape
    umin = float(cfg.get("umin", -1.0))
    umax = float(cfg.get("umax", 1.0))

    uL = (umin + (umax - umin) * torch.rand(n, 1, device=device, dtype=dtype))
    uR = (umin + (umax - umin) * torch.rand(n, 1, device=device, dtype=dtype))
    x0 = torch.rand(n, 1, device=device, dtype=dtype)  # in [0,1), will scale by L
    # infer L from grid
    dx = (x[1] - x[0])
    L = float(dx.item() * N)
    x0 = x0 * L

    # u[i, j] = uL if x[j] < x0[i] else uR
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
    Smooth base + steep tanh ramp:
      u0 = smooth(base_modes) + ramp_amp * tanh((x-x0)/eps)
    """
    n, N = u_shape
    base_cfg = {
        "n_modes": int(cfg.get("base_modes", 10)),
        "amp": float(cfg.get("amp", 0.5)),
        "decay_power": 2.0,
        "mean": float(cfg.get("mean", 0.0)),
    }
    u_base = _sample_smooth(u_shape, x, base_cfg, device, dtype)

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
    """
    Returns:
      u0: [n_samples, N]
      meta: dict with per-sample family ids and optional Riemann parameters
    """
    N = x.numel()
    u0 = torch.empty(n_samples, N, device=device, dtype=dtype)

    probs = torch.tensor(mix_probs, device=device, dtype=dtype)
    probs = probs / probs.sum()
    fam_ids = torch.multinomial(probs, num_samples=n_samples, replacement=True)  # [n_samples]

    # meta storage
    meta = {"family": [], "riemann": None}

    # fill per-family
    riemann_meta = None
    for idx, fam in enumerate(families):
        mask = (fam_ids == idx)
        count = int(mask.sum().item())
        if count == 0:
            continue

        if fam == "smooth":
            u0[mask] = _sample_smooth((count, N), x, family_cfg["smooth"], device, dtype)
        elif fam == "riemann":
            u_part, rmeta = _sample_riemann((count, N), x, family_cfg["riemann"], device, dtype)
            u0[mask] = u_part
            # collect riemann metadata (store arrays aligned to subset indices)
            if riemann_meta is None:
                riemann_meta = {k: [] for k in rmeta.keys()}
            for k in rmeta.keys():
                riemann_meta[k].append(rmeta[k])
        elif fam == "near_front":
            u0[mask] = _sample_near_front((count, N), x, family_cfg["near_front"], device, dtype)
        else:
            raise ValueError(f"Unknown family: {fam}")

        meta["family"] += [fam] * count

    # family list is out of original order because we appended by family; rebuild stable meta by storing ids
    # For simplicity, also store integer fam_ids; user can reconstruct if desired
    meta["family_id"] = fam_ids.detach().cpu()
    meta["families"] = families

    if riemann_meta is not None:
        # concatenate collected blocks
        meta["riemann"] = {k: torch.cat(v, dim=0) for k, v in riemann_meta.items()}

    return u0, meta
