import torch
from torch.utils.data import Dataset, DataLoader
from models import FCFNO1d


# -------------------------------------------------
# Dataset wrapper
# -------------------------------------------------

class BurgersDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.u0 = data["u0"]          # [n, N]
        self.u  = data["u"]           # [n, T, N]
        self.x  = data["x"]           # [N]
        self.t  = data["t"]           # [T]
        self.nu = data["nu"]           # [n, 1]

    def __len__(self):
        return self.u0.shape[0]

    def __getitem__(self, idx):
        return {
            "u0": self.u0[idx],
            "u":  self.u[idx],
            "nu": self.nu[idx],
        }


# -------------------------------------------------
# Training loop
# -------------------------------------------------

def train_epoch(model, loader, optimizer, x, t, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        u0 = batch["u0"].to(device)      # [B, N]
        u  = batch["u"].to(device)       # [B, T, N]
        nu = batch["nu"].to(device)      # [B, 1]

        optimizer.zero_grad()
        u_pred = model(u0, x=x, t=t, nu=nu)  # [B, T, N]

        loss = torch.mean((u_pred - u) ** 2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, x, t, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        u0 = batch["u0"].to(device)
        u  = batch["u"].to(device)
        nu = batch["nu"].to(device)

        u_pred = model(u0, x=x, t=t, nu=nu)
        loss = torch.mean((u_pred - u) ** 2)
        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load datasets
    # -----------------------------
    train_ds = BurgersDataset("data/cached/burgers_fronts_train.pt")
    val_ds   = BurgersDataset("data/cached/burgers_fronts_val.pt")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16)

    # Shared grids
    x = train_ds.x.to(device)   # [N]
    t = train_ds.t.to(device)   # [T]

    # -----------------------------
    # Model
    # -----------------------------
    model = FCFNO1d(
        width=64,
        modes=32,
        n_layers=4,
        nt_out=len(t),
        use_x=True,
        use_t=True,
        use_nu=True,
        L=1.0,
        alpha_init=0.05,     # conservative correction strength
        learn_alpha=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # -----------------------------
    # Training
    # -----------------------------
    n_epochs = 50
    for ep in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, x, t, device)
        val_loss   = eval_epoch(model, val_loader, x, t, device)

        print(f"[Epoch {ep:03d}] train L2={train_loss:.4e} | val L2={val_loss:.4e}")

    torch.save(model.state_dict(), "fcfno_burgers.pt")
    print("Saved model → fcfno_burgers.pt")


if __name__ == "__main__":
    main()
