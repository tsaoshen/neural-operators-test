# models/fno1d.py
"""
Baseline 1D Fourier Neural Operator (FNO) + Conservative Flux Corrector (FC-FNO).

This file provides:
- SpectralConv1d: Fourier layer with truncated modes and learned complex weights
- FNO1d: baseline FNO mapping u0 (and optional conditioning) -> u(x,t) snapshots
- FluxCorrector1d: conservative local correction via learned numerical flux and Dx divergence
- FCFNO1d: wrapper that runs FNO then applies corrector

Assumptions:
- Periodic domain in x (matches Burgers periodic setup).
- Outputs are "one-shot" nt_out time snapshots (like FNO 2021 style datasets).

Typical usage:
    model = FCFNO1d(width=64, modes=32, n_layers=4, nt_out=16, use_nu=True).to(device)
    pred = model(u0, x=x_grid, t=t_grid, nu=nu)  # pred: [B, nt_out, N]

Input conventions:
- u0: [B, N] initial condition
- x:  [N] or [B, N] (optional coordinate)
- t:  [nt_out] (optional time grid)
- nu: [B, 1] or scalar tensor (optional viscosity conditioning)

If you do not pass x/t/nu, the model still runs (it just uses u0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------

def _ensure_batch_grid(x: torch.Tensor, B: int, N: int) -> torch.Tensor:
    """Return x as [B, N]. Accepts [N] or [B, N]."""
    if x.dim() == 1:
        assert x.numel() == N
        return x[None, :].expand(B, N)
    assert x.dim() == 2 and x.shape == (B, N)
    return x


def _periodic_dx(flux: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Conservative divergence: (F_{i+1/2} - F_{i-1/2}) / dx
    flux: [B, T, N] values aligned with i+1/2 (periodic)
    """
    flux_imhalf = torch.roll(flux, shifts=1, dims=-1)
    return (flux - flux_imhalf) / dx


# ----------------------------
# Spectral Convolution Layer
# ----------------------------

class SpectralConv1d(nn.Module):
    """
    1D spectral convolution using rFFT:
      y(x) = irfft( W(k) * rfft(x) )
    where only modes 0..modes-1 are used (truncation).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = int(modes)

        # Learned complex weights for low modes: [in, out, modes]
        # Store as real+imag separately for stability/portability.
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes))

    def compl_mul1d(self, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Complex multiply in Fourier space.
        x_hat: [B, in, K]
        weights: [in, out, modes]
        return: [B, out, K] but only first modes filled
        """
        B, in_ch, K = x_hat.shape
        m = min(self.modes, K)
        # Build complex weight
        w = torch.complex(self.weight_real[..., :m], self.weight_imag[..., :m])  # [in, out, m]
        # Multiply: (B,in,m) x (in,out,m) -> (B,out,m)
        out_hat_low = torch.einsum("bim,iom->bom", x_hat[:, :, :m], w)
        out_hat = torch.zeros(B, self.out_channels, K, device=x_hat.device, dtype=x_hat.dtype)
        out_hat[:, :, :m] = out_hat_low
        return out_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in, N]
        returns: [B, out, N]
        """
        B, in_ch, N = x.shape
        # rFFT over last dim
        x_hat = torch.fft.rfft(x, dim=-1)  # [B, in, K], complex
        out_hat = self.compl_mul1d(x_hat)  # [B, out, K]
        y = torch.fft.irfft(out_hat, n=N, dim=-1)  # [B, out, N], real
        return y


# ----------------------------
# Baseline FNO1d
# ----------------------------

class FNOBlock1d(nn.Module):
    """
    One FNO block: spectral conv + pointwise conv + activation.
    """

    def __init__(self, width: int, modes: int, act: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU() if act.lower() == "gelu" else nn.ReLU()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: [B, width, N]
        return self.act(self.spectral(v) + self.pointwise(v))


class FNO1d(nn.Module):
    """
    Baseline 1D FNO mapping u0 -> u(x,t) snapshots (one-shot).

    Input feature construction:
      features per x: [u0(x), x, nu] and then expanded/repeated across t,
      plus time embedding appended (t) if provided.

    Output:
      u_pred: [B, nt_out, N]
    """

    def __init__(
        self,
        width: int = 64,
        modes: int = 32,
        n_layers: int = 4,
        nt_out: int = 16,
        use_x: bool = True,
        use_t: bool = True,
        use_nu: bool = True,
        act: str = "gelu",
    ):
        super().__init__()
        self.width = int(width)
        self.modes = int(modes)
        self.n_layers = int(n_layers)
        self.nt_out = int(nt_out)

        self.use_x = bool(use_x)
        self.use_t = bool(use_t)
        self.use_nu = bool(use_nu)

        in_ch = 1
        if self.use_x:
            in_ch += 1
        if self.use_nu:
            in_ch += 1
        if self.use_t:
            in_ch += 1  # time appended per snapshot

        # Lift to width channels
        self.lift = nn.Linear(in_ch, self.width)

        self.blocks = nn.ModuleList([FNOBlock1d(self.width, self.modes, act=act) for _ in range(self.n_layers)])

        # Project back to scalar
        self.proj1 = nn.Conv1d(self.width, self.width, kernel_size=1)
        self.proj2 = nn.Conv1d(self.width, 1, kernel_size=1)
        self.act = nn.GELU() if act.lower() == "gelu" else nn.ReLU()

    def _build_features(
        self,
        u0: torch.Tensor,               # [B, N]
        x: Optional[torch.Tensor],      # [N] or [B, N]
        t: Optional[torch.Tensor],      # [T]
        nu: Optional[torch.Tensor],     # [B,1] or scalar
    ) -> torch.Tensor:
        """
        Returns features shaped [B, T, N, in_ch]
        """
        B, N = u0.shape
        T = self.nt_out

        feats = [u0[:, None, :, None].expand(B, T, N, 1)]  # [B,T,N,1]

        if self.use_x:
            if x is None:
                # Default x in [0,1)
                x = torch.linspace(0, 1, steps=N, device=u0.device, dtype=u0.dtype)
            xB = _ensure_batch_grid(x.to(device=u0.device, dtype=u0.dtype), B, N)
            feats.append(xB[:, None, :, None].expand(B, T, N, 1))

        if self.use_nu:
            if nu is None:
                nu = torch.zeros(B, 1, device=u0.device, dtype=u0.dtype)
            if nu.dim() == 0:
                nu = nu.view(1, 1).expand(B, 1)
            if nu.dim() == 1:
                nu = nu.view(B, 1)
            feats.append(nu[:, None, None, :].expand(B, T, N, 1))

        if self.use_t:
            if t is None:
                t = torch.linspace(0, 1, steps=T, device=u0.device, dtype=u0.dtype)
            t = t.to(device=u0.device, dtype=u0.dtype)
            assert t.numel() == T, f"t should have length nt_out={T}"
            feats.append(t[None, :, None, None].expand(B, T, N, 1))

        return torch.cat(feats, dim=-1)  # [B,T,N,in_ch]

    def forward(
        self,
        u0: torch.Tensor,               # [B, N]
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        nu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N = u0.shape
        feats = self._build_features(u0, x=x, t=t, nu=nu)      # [B,T,N,in_ch]
        B, T, N, C = feats.shape

        # Lift pointwise: [B,T,N,C] -> [B,T,N,width]
        v = self.lift(feats)  # linear on last dim
        # Reshape to conv format per time: merge B and T
        v = v.reshape(B * T, N, self.width).permute(0, 2, 1).contiguous()  # [B*T,width,N]

        for blk in self.blocks:
            v = blk(v)  # [B*T,width,N]

        v = self.act(self.proj1(v))
        v = self.proj2(v)  # [B*T,1,N]
        v = v.squeeze(1)   # [B*T,N]
        u = v.view(B, T, N).contiguous()
        return u


# ----------------------------
# Conservative Flux Corrector
# ----------------------------

class FluxNet1d(nn.Module):
    """
    Small 1D CNN to produce a numerical flux field F_{i+1/2} from local stencils of u.

    Input:  u: [B, T, N] (cell centered)
    Output: flux: [B, T, N] (interpreted as flux at i+1/2 aligned with i via periodic conv)
    """

    def __init__(self, hidden: int = 32, n_layers: int = 3, kernel_size: int = 5, act: str = "gelu"):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.kernel_size = int(kernel_size)
        self.act = nn.GELU() if act.lower() == "gelu" else nn.ReLU()

        layers = []
        in_ch = 1
        for i in range(n_layers - 1):
            layers.append(nn.Conv1d(in_ch, hidden, kernel_size=self.kernel_size, padding=0, bias=True))
            layers.append(self.act)
            in_ch = hidden
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=self.kernel_size, padding=0, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: [B, T, N]
        return flux: [B, T, N]
        Periodic padding is applied manually to make the conv periodic.
        """
        B, T, N = u.shape
        # Merge B and T for 1D conv
        z = u.reshape(B * T, 1, N)

        pad = self.kernel_size // 2
        # periodic pad
        zpad = torch.cat([z[..., -pad:], z, z[..., :pad]], dim=-1)  # [B*T,1,N+2pad]

        flux = self.net(zpad)  # [B*T,1,N]
        flux = flux.view(B, T, N)
        return flux


class FluxCorrector1d(nn.Module):
    """
    Conservative correction:
        u_fc = u_fno - alpha * Dx( flux_net(u_fno) )

    where Dx uses periodic conservative divergence.

    This is NOT a full PDE solver; it's a post-correction designed to reduce ringing and
    improve shock structure while preserving conservation form.
    """

    def __init__(
        self,
        L: float = 1.0,
        alpha_init: float = 0.1,
        learn_alpha: bool = True,
        flux_hidden: int = 32,
        flux_layers: int = 3,
        flux_kernel: int = 5,
        act: str = "gelu",
    ):
        super().__init__()
        self.L = float(L)
        self.flux_net = FluxNet1d(hidden=flux_hidden, n_layers=flux_layers, kernel_size=flux_kernel, act=act)

        alpha = torch.tensor(float(alpha_init))
        self.alpha_param = nn.Parameter(alpha) if learn_alpha else alpha
        self.learn_alpha = bool(learn_alpha)

    def forward(self, u: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        u: [B, T, N]
        x: [N] or [B,N] optional; if provided, used to compute dx; else dx=L/N
        """
        B, T, N = u.shape

        if x is not None:
            if x.dim() == 1:
                dx = float((x[1] - x[0]).detach().cpu().item())
            else:
                dx = float((x[0, 1] - x[0, 0]).detach().cpu().item())
        else:
            dx = self.L / N

        flux = self.flux_net(u)  # [B,T,N]
        dudx = _periodic_dx(flux, dx=dx)  # [B,T,N]

        alpha = F.softplus(self.alpha_param) if self.learn_alpha else float(self.alpha_param)
        u_corr = u - alpha * dudx
        return u_corr


# ----------------------------
# FC-FNO wrapper
# ----------------------------

class FCFNO1d(nn.Module):
    """
    FNO baseline + conservative flux corrector.
    """

    def __init__(
        self,
        width: int = 64,
        modes: int = 32,
        n_layers: int = 4,
        nt_out: int = 16,
        use_x: bool = True,
        use_t: bool = True,
        use_nu: bool = True,
        # corrector params
        L: float = 1.0,
        alpha_init: float = 0.1,
        learn_alpha: bool = True,
        flux_hidden: int = 32,
        flux_layers: int = 3,
        flux_kernel: int = 5,
        act: str = "gelu",
    ):
        super().__init__()
        self.fno = FNO1d(
            width=width,
            modes=modes,
            n_layers=n_layers,
            nt_out=nt_out,
            use_x=use_x,
            use_t=use_t,
            use_nu=use_nu,
            act=act,
        )
        self.corrector = FluxCorrector1d(
            L=L,
            alpha_init=alpha_init,
            learn_alpha=learn_alpha,
            flux_hidden=flux_hidden,
            flux_layers=flux_layers,
            flux_kernel=flux_kernel,
            act=act,
        )

    def forward(
        self,
        u0: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        nu: Optional[torch.Tensor] = None,
        apply_corrector: bool = True,
    ) -> torch.Tensor:
        u_pred = self.fno(u0, x=x, t=t, nu=nu)  # [B,T,N]
        if apply_corrector:
            u_pred = self.corrector(u_pred, x=x)
        return u_pred
