import os
import argparse
import torch
from torch.utils.data import DataLoader

from data.burgers_dataset import BurgersOperatorDataset, burgers_collate
from models.fno1d import FNO1d
from models.flux_corrector import FluxCorrector1d
from models.fc_fno import FCFNO1d


def make_input(u0, x, nu):
    B, N = u0.shape
    x_chan = x[None, :].expand(B, N)
    nu_chan = nu.expand(B, 1).expand(B, N)
    return torch.stack([u0, x_chan, nu_chan], dim=1)  # [B,3,N]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pt", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--use_corrector", action="store_true")

    # Must match training hyperparams
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--corr_hidden", type=int, default=32)
    parser.add_argument("--corr_layers", type=int, default=3)
    parser.add_argument("--corr_kernel", type=int, default=5)

    parser.add_argument("--save_pred", type=str, default="runs/preds.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = BurgersOperatorDataset(args.test_pt, device=None)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=burgers_collate)

    T = ds.u.shape[1]
    in_channels = 3

    fno = FNO1d(in_channels=in_channels, out_channels=T, width=args.width, modes=args.modes, n_layers=args.layers).to(device)

    if args.use_corrector:
        corr = FluxCorrector1d(hidden=args.corr_hidden, n_layers=args.corr_layers, kernel_size=args.corr_kernel).to(device)
        model = FCFNO1d(fno=fno, corrector=corr).to(device)
    else:
        model = fno

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    mse = 0.0
    n = 0
    # store a small subset of preds
    preds = []
    trues = []
    u0s = []

    for batch in loader:
        u0 = batch["u0"].to(device)
        u_true = batch["u"].to(device)
        x = batch["x"].to(device)
        nu = batch["nu"].to(device)
        dx = float((x[1] - x[0]).item())

        x_in = make_input(u0, x, nu)
        y = model(x_in, dx) if args.use_corrector else model(x_in)

        loss = torch.mean((y - u_true) ** 2)
        mse += loss.item() * u0.shape[0]
        n += u0.shape[0]

        if len(preds) < 8:
            preds.append(y[:2].detach().cpu())
            trues.append(u_true[:2].detach().cpu())
            u0s.append(u0[:2].detach().cpu())

    mse /= max(n, 1)
    print(f"[TEST] MSE = {mse:.4e}")

    os.makedirs(os.path.dirname(args.save_pred), exist_ok=True)
    torch.save(
        {
            "pred": torch.cat(preds, dim=0),
            "true": torch.cat(trues, dim=0),
            "u0": torch.cat(u0s, dim=0),
            "x": ds.x,
            "t": ds.t,
            "mse": mse,
        },
        args.save_pred,
    )
    print(f"[OK] saved predictions to {args.save_pred}")


if __name__ == "__main__":
    main()
