import torch


def make_grid_1d(N: int, L: float, device, dtype):
    # cell centers in [0, L)
    dx = L / N
    x = torch.arange(N, device=device, dtype=dtype) * dx
    return x


def downsample_1d(u: torch.Tensor, N_out: int) -> torch.Tensor:
    """
    u: [B, T, N] or [B, N] or [B, 1, N]
    Uses simple strided sampling if divisible; else uses linear interpolation.
    """
    if u.dim() == 2:
        u = u[:, None, :]
        squeeze_T = True
    else:
        squeeze_T = False

    B, T, N = u.shape
    if N == N_out:
        out = u
    elif N % N_out == 0:
        stride = N // N_out
        out = u[:, :, ::stride]
    else:
        # linear interpolation along last axis
        x_in = torch.linspace(0, 1, steps=N, device=u.device, dtype=u.dtype)
        x_out = torch.linspace(0, 1, steps=N_out, device=u.device, dtype=u.dtype)
        out = torch.empty(B, T, N_out, device=u.device, dtype=u.dtype)
        for b in range(B):
            for t in range(T):
                out[b, t] = torch.interp(x_out, x_in, u[b, t])

    return out[:, 0, :] if squeeze_T else out


def pick_time_indices(n_steps: int, nt_out: int, device):
    ids = torch.linspace(0, n_steps, steps=nt_out, device=device).round().long()
    ids[0] = 0
    ids[-1] = n_steps
    return ids
