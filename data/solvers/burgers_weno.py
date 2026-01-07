import torch
import math
from typing import Tuple


def _roll_periodic(u: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(u, shifts=shift, dims=-1)


def _weno5_reconstruct_flux(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    WENO5 reconstruction for positive-going flux at i+1/2.
    Input f: [B, N] values at cell centers.
    Output fhat: [B, N] numerical flux at interfaces i+1/2 aligned with i.
    Periodic assumed by using torch.roll.
    """
    f_im2 = _roll_periodic(f, 2)
    f_im1 = _roll_periodic(f, 1)
    f_i   = f
    f_ip1 = _roll_periodic(f, -1)
    f_ip2 = _roll_periodic(f, -2)

    # Candidate polynomials (left-biased)
    p0 = (2.0*f_im2 - 7.0*f_im1 + 11.0*f_i) / 6.0
    p1 = (-1.0*f_im1 + 5.0*f_i + 2.0*f_ip1) / 6.0
    p2 = (2.0*f_i + 5.0*f_ip1 - 1.0*f_ip2) / 6.0

    # Smoothness indicators beta
    beta0 = (13.0/12.0)*(f_im2 - 2.0*f_im1 + f_i)**2 + 0.25*(f_im2 - 4.0*f_im1 + 3.0*f_i)**2
    beta1 = (13.0/12.0)*(f_im1 - 2.0*f_i + f_ip1)**2 + 0.25*(f_im1 - f_ip1)**2
    beta2 = (13.0/12.0)*(f_i - 2.0*f_ip1 + f_ip2)**2 + 0.25*(3.0*f_i - 4.0*f_ip1 + f_ip2)**2

    # Linear weights
    d0, d1, d2 = 0.1, 0.6, 0.3

    alpha0 = d0 / (eps + beta0)**2
    alpha1 = d1 / (eps + beta1)**2
    alpha2 = d2 / (eps + beta2)**2
    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum

    fhat = w0*p0 + w1*p1 + w2*p2  # at i+1/2
    return fhat


def _weno5_flux_derivative(u: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Compute -(f(u))_x with WENO5 + Lax-Friedrichs flux splitting.
    u: [B, N]
    returns rhs_conv: [B, N]
    """
    # Physical flux
    f = 0.5 * u * u

    # Lax-Friedrichs splitting
    a = torch.max(torch.abs(u), dim=-1, keepdim=True).values  # [B,1]
    fp = 0.5 * (f + a * u)
    fm = 0.5 * (f - a * u)

    # Reconstruct fp at i+1/2 from left; reconstruct fm at i-1/2 from right.
    fp_hat_iphalf = _weno5_reconstruct_flux(fp)              # aligned with i -> i+1/2
    # For negative-going flux, reconstruct on reversed stencil by symmetry:
    # fm_hat_{i+1/2} = reconstruct_flux( roll(fm, -1) ) shifted back
    fm_hat_iphalf = _roll_periodic(_weno5_reconstruct_flux(_roll_periodic(fm, -1)), 1)

    # Total numerical flux at i+1/2
    flux_iphalf = fp_hat_iphalf + fm_hat_iphalf  # [B,N] at i+1/2

    # Divergence: (F_{i+1/2} - F_{i-1/2}) / dx
    flux_imhalf = _roll_periodic(flux_iphalf, 1)
    rhs_conv = -(flux_iphalf - flux_imhalf) / dx
    return rhs_conv


def _laplacian_periodic(u: torch.Tensor, dx: float) -> torch.Tensor:
    return (_roll_periodic(u, -1) - 2.0*u + _roll_periodic(u, 1)) / (dx*dx)


@torch.no_grad()
def solve_burgers_weno(
    u0: torch.Tensor,
    L: float,
    T: float,
    nu: float,
    cfl: float,
    nt_out: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve u_t + (u^2/2)_x = nu u_xx on periodic domain [0,L],
    using finite volume WENO5 for convection + central diff for diffusion,
    advanced by SSPRK3 with CFL-based dt.

    Inputs:
      u0: [B, N]
    Returns:
      u_snap: [B, nt_out, N] snapshots including t=0 and t=T
      t_out:  [nt_out]
    """
    B, N = u0.shape
    dx = L / N
    u = u0.clone()

    # Estimate dt from CFL on max |u|
    umax = torch.max(torch.abs(u)).item()
    # For viscous term, include diffusion stability heuristic (explicit): dt <= 0.5 dx^2/nu
    dt_conv = cfl * dx / max(umax, 1e-6)
    dt_diff = 0.5 * dx * dx / nu if nu > 0.0 else float("inf")
    dt = min(dt_conv, dt_diff)

    n_steps = int(math.ceil(T / dt))
    dt = T / n_steps  # hit T exactly

    # Snapshot indices in time
    # include t=0 and t=T, total nt_out
    snap_ids = torch.linspace(0, n_steps, steps=nt_out, device=u.device).round().long()
    snap_ids[0] = 0
    snap_ids[-1] = n_steps

    u_snap = torch.empty(B, nt_out, N, device=u.device, dtype=u.dtype)
    t_out = torch.linspace(0.0, T, steps=nt_out, device=u.device, dtype=u.dtype)

    snap_ptr = 0
    u_snap[:, snap_ptr, :] = u
    snap_ptr += 1

    def rhs(uu: torch.Tensor) -> torch.Tensor:
        r = _weno5_flux_derivative(uu, dx)
        if nu > 0.0:
            r = r + nu * _laplacian_periodic(uu, dx)
        return r

    # SSPRK3 time stepping
    for n in range(1, n_steps + 1):
        k1 = rhs(u)
        u1 = u + dt * k1

        k2 = rhs(u1)
        u2 = 0.75 * u + 0.25 * (u1 + dt * k2)

        k3 = rhs(u2)
        u = (1.0/3.0) * u + (2.0/3.0) * (u2 + dt * k3)

        if snap_ptr < nt_out and n == snap_ids[snap_ptr].item():
            u_snap[:, snap_ptr, :] = u
            snap_ptr += 1

    return u_snap, t_out
