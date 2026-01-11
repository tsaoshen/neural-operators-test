import torch
import torch.nn as nn
import torch.nn.functional as F


class FluxCorrector1d(nn.Module):
    """
    Conservative correction applied to each time slice:
      u_corr = u - alpha * div_x(Flux(u))
    where Flux(u) is learned by a small 1D CNN with circular padding.
    """

    def __init__(self, hidden: int = 32, n_layers: int = 3, kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for symmetric padding"
        pad = kernel_size // 2

        layers = []
        in_ch = 1
        for i in range(n_layers - 1):
            layers.append(nn.Conv1d(in_ch, hidden, kernel_size=kernel_size, padding=pad, padding_mode="circular"))
            layers.append(nn.GELU())
            in_ch = hidden
        # output one flux value per grid point, interpreted as flux at i+1/2 aligned with i
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=kernel_size, padding=pad, padding_mode="circular"))
        self.net = nn.Sequential(*layers)

        # Learn positive alpha via softplus
        self._alpha_raw = nn.Parameter(torch.tensor(0.0))  # alpha ~ softplus(raw)

    def alpha(self) -> torch.Tensor:
        return F.softplus(self._alpha_raw) + 1e-6

    def forward(self, u: torch.Tensor, dx: float) -> torch.Tensor:
        """
        u:  [B, T, N] (or [B,1,N] if you want)
        dx: grid spacing

        returns corrected u with same shape.
        """
        if u.dim() == 3:
            B, T, N = u.shape
            uu = u.reshape(B * T, 1, N)  # [B*T,1,N]
            flux = self.net(uu)          # [B*T,1,N] representing F_{i+1/2}
            flux = flux[:, 0, :]         # [B*T,N]

            # divergence: (F_{i+1/2} - F_{i-1/2})/dx
            flux_imhalf = torch.roll(flux, shifts=1, dims=-1)
            div = (flux - flux_imhalf) / dx  # [B*T,N]

            u_corr = uu[:, 0, :] - self.alpha() * div
            return u_corr.reshape(B, T, N)

        elif u.dim() == 2:
            B, N = u.shape
            uu = u[:, None, :]           # [B,1,N]
            flux = self.net(uu)[:, 0, :] # [B,N]
            flux_imhalf = torch.roll(flux, shifts=1, dims=-1)
            div = (flux - flux_imhalf) / dx
            return u - self.alpha() * div

        else:
            raise ValueError("u must have shape [B,T,N] or [B,N]")
