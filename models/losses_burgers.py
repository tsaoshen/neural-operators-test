import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def fd_central_periodic_x(u: torch.Tensor, dx: float) -> torch.Tensor:
    # u: [B,T,N]
    return (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2.0 * dx)


def fd_forward_t(u: torch.Tensor, dt: float) -> torch.Tensor:
    # u: [B,T,N]
    # forward difference in t (first order), last time uses backward
    B, T, N = u.shape
    ut = torch.zeros_like(u)
    if T == 1:
        return ut
    ut[:, :-1] = (u[:, 1:] - u[:, :-1]) / dt
    ut[:, -1] = (u[:, -1] - u[:, -2]) / dt
    return ut


class TestFunctionBank(nn.Module):
    """
    Simple fixed bank of separable test functions:
      phi_{p,q,kind}(x,t) = trig_p(x) * trig_q(t)   (sin/cos combinations)
    plus optional nonnegative bumps for entropy.
    """

    def __init__(self, p_max: int = 8, q_max: int = 8, use_cos: bool = True):
        super().__init__()
        self.p_max = p_max
        self.q_max = q_max
        self.use_cos = use_cos

    def make_trig(self, k: int, grid: torch.Tensor, kind: str, L: float) -> torch.Tensor:
        # grid: [N] or [T]
        ang = 2.0 * math.pi * k * grid / L
        if kind == "sin":
            return torch.sin(ang)
        elif kind == "cos":
            return torch.cos(ang)
        else:
            raise ValueError(kind)

    def build_phi(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          phi:   [M, T, N]
          phi_t: [M, T, N] (analytic)
          phi_x: [M, T, N] (analytic)
        """
        device, dtype = x.device, x.dtype
        N = x.numel()
        Tn = t.numel()

        dx = float((x[1] - x[0]).item())
        Lx = float(x[-1].item() + dx)
        Lt = float(t[-1].item() - t[0].item() + (t[1] - t[0]).item()) if Tn > 1 else 1.0

        kinds = ["sin"] + (["cos"] if self.use_cos else [])

        phis = []
        phit = []
        phix = []

        for p in range(1, self.p_max + 1):
            for q in range(1, self.q_max + 1):
                for kx in kinds:
                    for kt in kinds:
                        tx = self.make_trig(p, x, kx, Lx)  # [N]
                        tt = self.make_trig(q, t, kt, Lt)  # [T]

                        phi = tt[:, None] * tx[None, :]  # [T,N]

                        # derivatives analytically
                        # d/dx trig = (2πp/Lx) * (cos or -sin)
                        ax = 2.0 * math.pi * p / Lx
                        at = 2.0 * math.pi * q / Lt

                        if kx == "sin":
                            dtx = ax * torch.cos(2.0 * math.pi * p * x / Lx)
                        else:
                            dtx = -ax * torch.sin(2.0 * math.pi * p * x / Lx)

                        if kt == "sin":
                            dtt = at * torch.cos(2.0 * math.pi * q * t / Lt)
                        else:
                            dtt = -at * torch.sin(2.0 * math.pi * q * t / Lt)

                        phi_x = dtt[:, None] * tx[None, :] * 0.0 + tt[:, None] * dtx[None, :]
                        phi_t = dtt[:, None] * tx[None, :]

                        phis.append(phi)
                        phix.append(phi_x)
                        phit.append(phi_t)

        phi = torch.stack(phis, dim=0).to(device=device, dtype=dtype)    # [M,T,N]
        phi_x = torch.stack(phix, dim=0).to(device=device, dtype=dtype)
        phi_t = torch.stack(phit, dim=0).to(device=device, dtype=dtype)
        return phi, phi_t, phi_x

    def build_nonnegative_bumps(self, x: torch.Tensor, t: torch.Tensor, n_bumps: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For entropy: phi >= 0. We'll use simple Gaussian bumps.
        Returns:
          phi: [M,T,N], phi_x: [M,T,N], phi_t: [M,T,N] (finite-diff for simplicity)
        """
        device, dtype = x.device, x.dtype
        N = x.numel()
        Tn = t.numel()
        dx = float((x[1] - x[0]).item())
        dt = float((t[1] - t[0]).item()) if Tn > 1 else 1.0
        Lx = float(x[-1].item() + dx)

        phis = []
        for i in range(n_bumps):
            xc = (i + 0.5) * (Lx / n_bumps)
            tc = (i + 0.5) * (float(t[-1].item()) / n_bumps) if Tn > 1 else 0.0
            sigx = 0.12 * Lx
            sigt = 0.25 * float(t[-1].item() + 1e-12)

            bx = torch.exp(-0.5 * ((x - xc) / sigx) ** 2)  # [N]
            bt = torch.exp(-0.5 * ((t - tc) / max(sigt, 1e-6)) ** 2)  # [T]
            phis.append(bt[:, None] * bx[None, :])

        phi = torch.stack(phis, dim=0).to(device=device, dtype=dtype)  # [M,T,N]

        # derivatives via finite differences (good enough for training penalty)
        phi_x = (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1)) / (2.0 * dx)
        phi_t = torch.zeros_like(phi)
        if Tn > 1:
            phi_t[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dt)
            phi_t[:, 0] = (phi[:, 1] - phi[:, 0]) / dt
            phi_t[:, -1] = (phi[:, -1] - phi[:, -2]) / dt
        return phi, phi_t, phi_x


class BurgersWeakResidualLoss(nn.Module):
    def __init__(self, p_max: int = 6, q_max: int = 6, use_cos: bool = True):
        super().__init__()
        self.bank = TestFunctionBank(p_max=p_max, q_max=q_max, use_cos=use_cos)

    def forward(self, u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """
        u:  [B,T,N]
        x:  [N]
        t:  [T]
        nu: [B,1] or [B]
        returns scalar loss
        """
        B, Tn, N = u.shape
        dx = float((x[1] - x[0]).item())
        dt = float((t[1] - t[0]).item()) if Tn > 1 else 1.0

        # compute derivatives
        ux = fd_central_periodic_x(u, dx)  # [B,T,N]

        # test functions
        phi, phi_t, phi_x = self.bank.build_phi(x, t)  # [M,T,N]
        M = phi.shape[0]

        # flux f(u)=u^2/2
        f = 0.5 * u * u

        # Broadcast: [B,M,T,N]
        u_b = u[:, None, :, :]
        f_b = f[:, None, :, :]
        ux_b = ux[:, None, :, :]

        # nu broadcast to [B,1,1,1]
        if nu.dim() == 2:
            nu_b = nu[:, 0].view(B, 1, 1, 1)
        else:
            nu_b = nu.view(B, 1, 1, 1)

        integrand = u_b * phi_t[None, :, :, :] + f_b * phi_x[None, :, :, :] + nu_b * ux_b * phi_x[None, :, :, :]
        R = torch.sum(integrand, dim=(2, 3)) * (dx * dt)  # [B,M]
        loss = torch.mean(R ** 2)
        return loss


class BurgersEntropyInequalityLoss(nn.Module):
    def __init__(self, n_k: int = 16, n_bumps: int = 8):
        super().__init__()
        self.n_k = n_k
        self.n_bumps = n_bumps
        self.bank = TestFunctionBank(p_max=1, q_max=1, use_cos=False)  # not used for entropy bumps

    def forward(self, u: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Entropy inequality proxy for inviscid Burgers:
          ∫∫ (eta_k(u) phi_t + q_k(u) phi_x) dx dt <= 0 for phi >= 0.
        Penalize relu(E).

        u: [B,T,N]
        """
        B, Tn, N = u.shape
        dx = float((x[1] - x[0]).item())
        dt = float((t[1] - t[0]).item()) if Tn > 1 else 1.0

        # choose k per-batch from global range
        umin = u.detach().min()
        umax = u.detach().max()
        k = torch.linspace(umin, umax, steps=self.n_k, device=u.device, dtype=u.dtype)  # [K]

        # nonnegative bumps
        phi, phi_t, phi_x = self.bank.build_nonnegative_bumps(x, t, n_bumps=self.n_bumps)  # [M,T,N]
        M = phi.shape[0]
        K = k.numel()

        # eta_k = |u-k|
        eta = torch.abs(u[..., None] - k[None, None, None, :])  # [B,T,N,K]

        # q_k = sign(u-k)(f(u)-f(k))
        f = 0.5 * u * u
        fk = 0.5 * k * k
        sgn = torch.sign(u[..., None] - k[None, None, None, :])
        q = sgn * (f[..., None] - fk[None, None, None, :])      # [B,T,N,K]

        # apply bumps: [M,T,N] -> [1,M,T,N,1]
        phi_t_b = phi_t[None, :, :, :, None]
        phi_x_b = phi_x[None, :, :, :, None]

        # expand u dims with M: [B,1,T,N,K]
        eta = eta[:, None, :, :, :]
        q = q[:, None, :, :, :]

        integrand = eta * phi_t_b + q * phi_x_b   # [B,M,T,N,K]
        E = torch.sum(integrand, dim=(2, 3)) * (dx * dt)  # [B,M,K]

        loss = torch.mean(F.relu(E) ** 2)
        return loss
