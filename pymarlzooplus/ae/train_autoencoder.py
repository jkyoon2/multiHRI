import os
import argparse
import json
import time
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .autoencoder import AutoEncoder, Encoder


class VisualObsDataset(Dataset):
    def __init__(self, shards, normalize=True):
        self.files = shards
        self.normalize = normalize
        self._index = []  # (file_idx, local_idx)
        for fi, fpath in enumerate(self.files):
            with np.load(fpath) as data:
                n = data["visual_obs"].shape[0]
            self._index.extend([(fi, i) for i in range(n)])

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fi, li = self._index[idx]
        with np.load(self.files[fi]) as data:
            arr = data["visual_obs"][li]  # (C,H,W)
        x = th.from_numpy(arr.astype(np.float32))
        if self.normalize:
            x = x / 20.0
        return x


def list_npz_files(root: str):
    shards = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".npz") and fn.startswith("shard_"):
                shards.append(os.path.join(dirpath, fn))
    shards.sort()
    return shards


def train_autoencoder(dataset_root: str,
                      latent_dim: int,
                      batch_size: int,
                      epochs: int,
                      lr: float,
                      weight_decay: float,
                      device: str,
                      target_hw: str,
                      out_dir: str):
    shards = list_npz_files(dataset_root)
    assert len(shards) > 0, f"No shards found under {dataset_root}"

    # Detect (H,W) from first sample for decoder target size
    with np.load(shards[0]) as data:
        c, h, w = data["visual_obs"][0].shape
    hw = (h, w) if target_hw is None else tuple(map(int, target_hw.split("x")))

    ds = VisualObsDataset(shards)
    n_val = max(1, int(0.05 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = th.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AutoEncoder(in_channels=c, target_hw=hw, latent_dim=latent_dim).to(device)
    opt = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    best_val = float("inf")
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x in train_loader:
            x = x.to(device)
            z, x_rec = model(x)
            loss = crit(x_rec, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with th.no_grad():
            for x in val_loader:
                x = x.to(device)
                _, x_rec = model(x)
                loss = crit(x_rec, x)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[AE] epoch {epoch}/{epochs} | train {train_loss:.6f} | val {val_loss:.6f}")

        # Save best encoder weights
        if val_loss < best_val:
            best_val = val_loss
            enc_path = os.path.join(out_dir, "encoder.pt")
            th.save(model.encoder.state_dict(), enc_path)
            meta = {
                "latent_dim": latent_dim,
                "normalize_divisor": 20.0,
                "input_channels": int(c),
                "target_hw": hw,
                "val_mse": float(val_loss),
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--target-hw", type=str, default=None, help="e.g., 6x5; if None, autodetect from dataset")
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "autoencoders", time.strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()

    train_autoencoder(dataset_root=args.dataset_root,
                      latent_dim=args.latent_dim,
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      lr=args.lr,
                      weight_decay=args.weight_decay,
                      device=args.device,
                      target_hw=args.target_hw,
                      out_dir=args.out)


if __name__ == "__main__":
    main()