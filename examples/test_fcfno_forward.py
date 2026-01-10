# examples/test_fcfno_forward.py
"""
Quick sanity check: instantiate model and run a forward pass.

Run:
  python examples/test_fcfno_forward.py
"""
import torch
from models import FCFNO1d

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 8
    N = 256
    nt_out = 16

    u0 = torch.randn(B, N, device=device)
    x = torch.linspace(0, 1, steps=N, device=device)
    t = torch.linspace(0, 1, steps=nt_out, device=device)
    nu = torch.full((B, 1), 1e-3, device=device)

    model = FCFNO1d(width=64, modes=32, n_layers=4, nt_out=nt_out, use_nu=True).to(device)
    y = model(u0, x=x, t=t, nu=nu)

    print("u0:", u0.shape)
    print("y :", y.shape)

if __name__ == "__main__":
    main()
