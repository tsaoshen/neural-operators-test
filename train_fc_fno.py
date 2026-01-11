import os
import argparse
import torch
from torch.utils.data import DataLoader

from data.burgers_dataset import BurgersOperatorDataset, burgers_collate
from models.fno1d import FNO1d
from models.flux_corrector import FluxCorrector1d
from models.fc_fno import FCFNO1d
from models.losses_burgers import BurgersWeakResidualLoss, BurgersEntropyInequalityLoss


def make_input(u0, x, nu):
    """
    u0: [B,N]
    x:  [N]
    nu: [B,1]
    returns: [B,3,N] channels [u0, x, nu]
    """
    B, N = u0.shape
    x_chan = x[None, :].expand(B, N)
    nu_chan = nu[:, 0][:, None].expand(B, N)
    return torch.stack([u0, x_chan, nu_chan], dim=1)


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

        if isinstance(model, FCFNO1d):
            y = model(x_in, dx)
        else:
            y = model(x_in)

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
    lambda_weak=0.0,
    lambda_ent=0.0,
    weak_loss=None,
    ent_loss=None,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        running_sup = 0.0
        running_w = 0.0
        running_e = 0.0
        n = 0

        for batch in train_loader:
            u0 = batch["u0"].to(device)
            u_true = batch["u"].to(device)
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            nu = batch["nu"].to(device)

            dx = float((x[1] - x[0]).item())
            x_in = make_input(u0, x, nu)

            if isinstance(model, FCFNO1d):
                y = model(x_in, dx)
            else:
                y = model(x_in)

            # supervised
            sup = torch.mean((y - u_true) ** 2)

            loss = sup
            wv = torch.tensor(0.0, device=device)
            ev = torch.tensor(0.0, device=device)

            if lambda_weak > 0.0:
                assert weak_loss is not None
                wv = weak_loss(y, x, t, nu)
                loss = loss + lambda_weak * wv

            if lambda_ent > 0.0:
                assert ent_loss is not None
                ev = ent_loss(y, x, t)
                loss = loss + lambda_ent * ev

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            B = u0.shape[0]
            running += loss.item() * B
            running_sup += sup.item() * B
            running_w += wv.item() * B
            running_e += ev.item() * B
            n += B

        scheduler.step()
        train_loss = running / max(n, 1)
        train_sup = running_sup / max(n, 1)
        train_w = running_w / max(n, 1)
        train_e = running_e / max(n, 1)

        val_mse = evaluate(model, val_loader, device)

        if val_mse < best_val:
            best_val = val_mse
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "val_mse": best_val,
                        "lambda_weak": lambda_weak,
                        "lambda_ent": lambda_ent,
                    },
                    save_path,
                )

        print(
            f"Epoch {ep:03d} | train_loss={train_loss:.4e} (sup={train_sup:.4e}, "
            f"weak={train_w:.4e}, ent={train_e:.4e}) | val_mse={val_mse:.4e} | best={best_val:.4e}"
        )

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

    # Weak + entropy loss params
    parser.add_argument("--lambda_weak", type=float, default=0.0)
    parser.add_argument("--lambda_ent", type=float, default=0.0)
    parser.add_argument("--weak_pmax", type=int, default=6)
    parser.add_argument("--weak_qmax", type=int, default=6)
    parser.add_argument("--ent_k", type=int, default=16)
    parser.add_argument("--ent_bumps", type=int, default=8)

    # Saving
    parser.add_argument("--out_dir", type=str, default="runs/fc_fno")
    parser.add_argument("--name", type=str, default="burgers")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = BurgersOperatorDataset(args.train_pt, device=None)
    val_ds = BurgersOperatorDataset(args.val_pt, device=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=burgers_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=burgers_collate)

    T = train_ds.u.shape[1]
    in_channels = 3  # [u0, x, nu]

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

    weak_loss = None
    ent_loss = None
    if args.lambda_weak > 0.0:
        weak_loss = BurgersWeakResidualLoss(p_max=args.weak_pmax, q_max=args.weak_qmax, use_cos=True).to(device)
        print(f"[Loss] Weak residual enabled: lambda_weak={args.lambda_weak}")

    if args.lambda_ent > 0.0:
        ent_loss = BurgersEntropyInequalityLoss(n_k=args.ent_k, n_bumps=args.ent_bumps).to(device)
        print(f"[Loss] Entropy inequality enabled: lambda_ent={args.lambda_ent}")

    best = train_one(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        save_path=save_path,
        lambda_weak=args.lambda_weak,
        lambda_ent=args.lambda_ent,
        weak_loss=weak_loss,
        ent_loss=ent_loss,
    )

    print(f"[DONE] best_val_mse={best:.4e} | saved: {save_path}")


if __name__ == "__main__":
    main()
