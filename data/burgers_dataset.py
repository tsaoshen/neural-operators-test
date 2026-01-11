import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional


class BurgersOperatorDataset(Dataset):
    """
    Loads Part-1 saved .pt files:
      {
        "u0": [n, N]    (optional but enabled in your config)
        "u":  [n, T, N]
        "x":  [N]
        "t":  [T]
        "nu": [n, 1]
        ...
      }
    Returns per-sample dict:
      u0: [N]
      u:  [T, N]
      x:  [N]
      t:  [T]
      nu: [1]
    """

    def __init__(self, pt_path: str, device: Optional[torch.device] = None):
        super().__init__()
        data: Dict[str, Any] = torch.load(pt_path, map_location="cpu")
        assert "u" in data, f"Missing 'u' in {pt_path}"
        assert "x" in data and "t" in data, f"Missing x/t in {pt_path}"
        assert "nu" in data, f"Missing 'nu' in {pt_path}"
        assert "u0" in data, (
            "Missing 'u0'. In Part 1 config set output.include_u0: true "
            "or modify model to infer u0 from u[:,0]."
        )

        self.u0 = data["u0"].float()     # [n, N]
        self.u = data["u"].float()       # [n, T, N]
        self.x = data["x"].float()       # [N]
        self.t = data["t"].float()       # [T]
        self.nu = data["nu"].float()     # [n, 1]

        self.device = device

        # Basic checks
        n, N = self.u0.shape
        n2, T, N2 = self.u.shape
        assert n == n2 and N == N2, "Inconsistent u0/u shapes"
        assert self.x.numel() == N, "x length mismatch"
        assert self.t.numel() == T, "t length mismatch"

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "u0": self.u0[idx],    # [N]
            "u": self.u[idx],      # [T, N]
            "nu": self.nu[idx],    # [1]
            "x": self.x,           # [N]
            "t": self.t,           # [T]
        }
        if self.device is not None:
            sample = {k: v.to(self.device) for k, v in sample.items()}
        return sample


def burgers_collate(batch):
    # x and t are identical across samples; take from first
    u0 = torch.stack([b["u0"] for b in batch], dim=0)  # [B,N]
    u = torch.stack([b["u"] for b in batch], dim=0)    # [B,T,N]
    nu = torch.stack([b["nu"] for b in batch], dim=0)  # [B,1]
    x = batch[0]["x"]                                  # [N]
    t = batch[0]["t"]                                  # [T]
    return {"u0": u0, "u": u, "nu": nu, "x": x, "t": t}
