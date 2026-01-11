import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.burgers_dataset import BurgersOperatorDataset, burgers_collate
from models.fno1d import FNO1d
from models.flux_corrector import FluxCorrector1d
from models.fc_fno import FCFNO1d


def make_input(u0, x, nu):
    """
    u0: [B,N]
    x:  [N]
    nu: [B,1]
    returns x_in: [B, C, N]
    """
    B, N = u0.shape
    x_chan = x[None, :].expand(B, N)
    nu_chan = nu.expand(B, 1).expand(B, N)

    # channels: [u0, x, nu]
    x_in = torch.stack([u0, x_chan, nu_chan], dim=1)  # [B,3,N]
    return x_in


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse = 0.0
    n = 0
    for batch in loader:
        u0 = batch["u0"].to(device)
        u_true = batch["u"].to(device)  # [B,T,N]
        x = batch["x"].to(device)
        nu = batch["nu"].to(device)

        dx = float((x[1] - x[0]).item())
        x_in = make_input(u0, x, nu)

        y = model(x_in, dx) if isinstance(model, FCFNO1d) else model(x_in)
        # y shape [B,T,N]
        loss = torch.mean((y - u_true) ** 2)
        mse += loss.item() * u0.shape[0]
        n += u0.shape[0]
    return mse / max(n, 1)


def train_one(
    model,
    train_loader,
    val_loader,
    device,
    lr=1e-3,
    epochs=50,
    save_path=None,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            u0 = batch["u0"].to(device)
            u_true = batch["u"].to(device)
            x = batch["x"].to(device)
            nu = batch["nu"].to(device)

            dx = float((x[1] - x[0]).item())
            x_in = make_input(u0, x, nu)

            if isinstance(model, FCFNO1d):
                y = model(x_in, dx)
            else:
                y = model(x_in)

            loss = torch.mean((y - u_true) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item() * u0.shape[0]
            n += u0.shape[0]

        scheduler.step()
        train_mse = running / max(n, 1)
        val_mse = evaluate(model, val_loader, device)

        if val_mse < best_val:
            best_val = val_mse
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({"model": model.state_dict(), "val_mse": best_val}, save_path)

        print(f"Epoch {ep:03d} | train_mse={train_mse:.4e} | val_mse={val_mse:.4e} | best={best_val:.4e}")

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--val_pt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)

    # FNO params
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)

    # Corrector params
    parser.add_argument("--use_corrector", action="store_true")
    parser.add_argument("--corr_hidden", type=int, default=32)
    parser.add_argument("--corr_layers", type=int, default=3)
    parser.add_argument("--corr_kernel", type=int, default=5)

    # Saving
    parser.add_argument("--out_dir", type=str, default="runs/fc_fno")
    parser.add_argument("--name", type=str, default="burgers")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = BurgersOperatorDataset(args.train_pt, device=None)
    val_ds = BurgersOperatorDataset(args.val_pt, device=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=burgers_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=burgers_collate)

    # Determine output channels from dataset
    T = train_ds.u.shape[1]
    in_channels = 3  # u0, x, nu

    fno = FNO1d(
        in_channels=in_channels,
        out_channels=T,
        width=args.width,
        modes=args.modes,
        n_layers=args.layers,
        activation="gelu",
    ).to(device)

    if args.use_corrector:
        corr = FluxCorrector1d(hidden=args.corr_hidden, n_layers=args.corr_layers, kernel_size=args.corr_kernel).to(device)
        model = FCFNO1d(fno=fno, corrector=corr).to(device)
        save_path = os.path.join(args.out_dir, f"{args.name}_fcfno.pt")
        print("[Model] FC-FNO (FNO + conservative flux corrector)")
    else:
        model = fno
        save_path = os.path.join(args.out_dir, f"{args.name}_fno.pt")
        print("[Model] Baseline FNO")

    best = train_one(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        save_path=save_path,
    )

    print(f"[DONE] best_val_mse={best:.4e} | saved: {save_path}")


if __name__ == "__main__":
    main()
