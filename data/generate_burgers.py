import os
import math
import yaml
import torch
from dataclasses import dataclass
from typing import Dict, Any

from data.ic.sample_ic import sample_u0_batch
from data.solvers.burgers_weno import solve_burgers_weno
from data.utils.io import save_pt
from data.utils.grids import make_grid_1d, downsample_1d, pick_time_indices


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_split(cfg: Dict[str, Any], split: str) -> int:
    return int(cfg["dataset"][f"n_{split}"])


def generate_split(cfg: Dict[str, Any], split: str, device: torch.device, dtype: torch.dtype):
    # --- Unpack ---
    N_ref = int(cfg["solver"]["N_ref"])
    N_out = int(cfg["output"]["N_out"])
    nt_out = int(cfg["output"]["nt_out"])
    T = float(cfg["T"])
    L = float(cfg["domain"]["L"])
    nu = float(cfg["solver"]["nu"])
    cfl = float(cfg["solver"]["cfl"])
    include_u0 = bool(cfg["output"]["include_u0"])

    n_samples = make_split(cfg, split)
    families = cfg["dataset"]["families"]
    mix_probs = cfg["dataset"]["mix_probs"]

    # --- grids ---
    x_ref = make_grid_1d(N_ref, L=L, device=device, dtype=dtype)  # [N_ref]
    # time indices for snapshotting will be selected after dt is known; we pick indices in solver.

    # --- sample initial conditions on ref grid ---
    u0_ref, meta = sample_u0_batch(
        n_samples=n_samples,
        x=x_ref,
        families=families,
        mix_probs=mix_probs,
        family_cfg=cfg["dataset"],
        device=device,
        dtype=dtype,
    )  # u0_ref: [n, N_ref]

    # --- solve PDE on ref grid ---
    # returns u_ref_snap: [n, nt_out, N_ref] (snapshots including t=0)
    u_ref_snap, t_out = solve_burgers_weno(
        u0=u0_ref,
        L=L,
        T=T,
        nu=nu,
        cfl=cfl,
        nt_out=nt_out,
    )

    # --- downsample spatially to N_out ---
    u_out = downsample_1d(u_ref_snap, N_out)  # [n, nt_out, N_out]
    u0_out = downsample_1d(u0_ref[:, None, :], N_out)[:, 0, :]  # [n, N_out]

    out = {
        "u": u_out.contiguous(),                  # [n, nt_out, N_out]
        "t": t_out.contiguous(),                  # [nt_out]
        "x": make_grid_1d(N_out, L=L, device=device, dtype=dtype).contiguous(),  # [N_out]
        "nu": torch.full((n_samples, 1), nu, device=device, dtype=dtype),
        "meta": meta,
        "split": split,
    }
    if include_u0:
        out["u0"] = u0_out.contiguous()          # [n, N_out]

    return out, (u_ref_snap if split == "test" else None)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/burgers.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))

    device = torch.device(cfg.get("device", "cpu"))
    dtype = DTYPE_MAP[cfg.get("dtype", "float32")]

    out_dir = cfg["save"]["out_dir"]
    name = cfg["save"]["name"]
    os.makedirs(out_dir, exist_ok=True)

    # Generate splits
    for split in ["train", "val", "test"]:
        data_split, u_ref_snap = generate_split(cfg, split, device=device, dtype=dtype)
        save_path = os.path.join(out_dir, f"{name}_{split}.pt")
        # save_pt(save_path, data_split)
        # print(f"[OK] Saved {split}: {save_path}")
        data_split2 = {}
        data_split2['x'] = data_split['u0']
        data_split2['y'] = data_split['u']
        save_pt(save_path, data_split2)
        print(f"[OK] Saved {split}: {save_path}")

        # # Optionally save high-res test
        # if split == "test" and bool(cfg["save"].get("save_hr_test", False)):
        #     N_test_hr = int(cfg["save"].get("N_test_hr", cfg["solver"]["N_ref"]))
        #     # Downsample high-res snapshots to N_test_hr for evaluation
        #     u_hr = downsample_1d(u_ref_snap, N_test_hr).contiguous()
        #     hr = {
        #         "u_hr": u_hr,
        #         "t": data_split["t"],
        #         "x_hr": make_grid_1d(N_test_hr, L=float(cfg["domain"]["L"]), device=device, dtype=dtype),
        #         "nu": data_split["nu"],
        #     }
        #     save_hr_path = os.path.join(out_dir, f"{name}_test_hr{N_test_hr}.pt")
        #     save_pt(save_hr_path, hr)
        #     print(f"[OK] Saved test high-res: {save_hr_path}")


if __name__ == "__main__":
    main()
