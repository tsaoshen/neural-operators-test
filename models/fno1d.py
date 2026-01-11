import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    1D spectral convolution:
      out(x) = irfft( W(k) * rfft(in(x)) )
    Only the first 'modes' Fourier modes are learned.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Complex weights: [in_channels, out_channels, modes]
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, N]
        returns: [B, C_out, N]
        """
        B, Cin, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # [B, Cin, Nf]
        Nf = x_ft.shape[-1]

        out_ft = torch.zeros(B, self.out_channels, Nf, device=x.device, dtype=torch.cfloat)

        m = min(self.modes, Nf)
        # Multiply low-frequency modes
        # out_ft[:, c_out, k] = sum_{c_in} x_ft[:, c_in, k] * weight[c_in, c_out, k]
        out_ft[:, :, :m] = torch.einsum("bik,iok->bok", x_ft[:, :, :m], self.weight[:, :, :m])

        out = torch.fft.irfft(out_ft, n=N, dim=-1)
        return out


class FNO1d(nn.Module):
    """
    Baseline FNO for operator learning Burgers:
      input channels typically: [u0(x), x, nu]
      output: u(x, t_j) for j=1..T  => [B, T, N]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: int = 32,
        n_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.n_layers = n_layers

        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.spec_convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)])

        self.proj1 = nn.Conv1d(width, 128, kernel_size=1)
        self.proj2 = nn.Conv1d(128, out_channels, kernel_size=1)

        if activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError("activation must be gelu or relu")

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        x_in: [B, in_channels, N]
        returns: [B, out_channels, N]
        """
        x = self.lift(x_in)  # [B,width,N]
        for spec, w in zip(self.spec_convs, self.ws):
            x = self.act(spec(x) + w(x))
        x = self.act(self.proj1(x))
        x = self.proj2(x)
        return x
