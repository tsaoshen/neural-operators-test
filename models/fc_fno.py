import torch
import torch.nn as nn
from models.fno1d import FNO1d
from models.flux_corrector import FluxCorrector1d


class FCFNO1d(nn.Module):
    """
    FC-FNO = Baseline FNO prediction + Conservative Flux Corrector.
    """

    def __init__(
        self,
        fno: FNO1d,
        corrector: FluxCorrector1d,
    ):
        super().__init__()
        self.fno = fno
        self.corrector = corrector

    def forward(self, x_in: torch.Tensor, dx: float) -> torch.Tensor:
        """
        x_in: [B, in_channels, N]
        returns: [B, T, N]
        """
        y = self.fno(x_in)  # [B, T, N] but currently [B, out_channels, N]
        # ensure shape [B,T,N]
        y = self.corrector(y, dx=dx)
        return y
