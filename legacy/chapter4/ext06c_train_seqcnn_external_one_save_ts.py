#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleSeqCNN(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B,4,L)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x).squeeze(-1)
        return x

def load_seq(dataset_dir: Path, split: str):
    for fn in [f"seq_{split}.npy", f"S_{split}.npy", f"Xseq_{split}.npy", f"onehot_{split}.npy"]:
        p = dataset_dir / fn
        if p.exists():
            a = np.load(p, mmap_mode="r")
            return a
    raise FileNotFoundError(f"[BAD] cannot find seq array for split={split} under {dataset_dir}")

def load_y(dataset_dir: Path, split: str):
    for fn in [f"y_{split}.npy", f"y_{split}.tsv", f"y_{split}.tsv.gz"]:
        p = dataset_dir / fn
        if p.exists():
            if fn.endswith(".npy"):
                return np.load(p)
            import pandas as pd
            df = pd.read_csv(p, sep="\t")
            return df.iloc[:,0].to_numpy()
    raise FileNotFoundError(f"[BAD] cannot find y for split={split} under {dataset_dir}")

def to_onehot(x):
    # x: (N,L) int tokens 0..3 OR (N,4,L)
    if x.ndim == 3:
        return x.astype(np.float32)
    if x.ndim == 2:
        N, L = x.shape
        oh = np.zeros((N,4,L), dtype=np.float32)
        xi = x.astype(np.int32)
        for b in range(4):
            oh[:,b,:] = (xi == b)
        return oh
    raise RuntimeError(f"[BAD] unexpected seq shape: {x.shape}")

def train_one(model, dl, opt):
    model.train()
    loss_fn = nn.MSELoss()
    tot = 0.0
    n = 0
    for xb, yb in dl:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        tot += float(loss.item()) * len(yb)
        n += len(yb)
    return tot / max(1,n)

@torch.no_grad()
def eval_pred(model, dl):
    model.eval()
    ys=[]
    ps=[]
    for xb,yb in dl:
        pred = model(xb)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    if len(ps)==0:
        return np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(ps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_ts", action="store_true", help="save model.ts")
    args = ap.parse_args()

    d = Path(args.dataset_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # load
    Xtr = to_onehot(load_seq(d, "train"))
    Xva = to_onehot(load_seq(d, "val"))
    Xte = to_onehot(load_seq(d, "test"))
    ytr = load_y(d, "train").astype(np.float32)
    yva = load_y(d, "val").astype(np.float32)
    yte = load_y(d, "test").astype(np.float32)

    L = Xtr.shape[-1]
    device = torch.device("cpu")

    model = SimpleSeqCNN(L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                       batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                       batch_size=args.batch_size, shuffle=False)
    te_dl = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                       batch_size=args.batch_size, shuffle=False)

    for ep in range(1, args.epochs+1):
        loss = train_one(model, tr_dl, opt)
        if ep % 5 == 0 or ep == args.epochs:
            yv, pv = eval_pred(model, va_dl)
            print(f"[INFO] epoch={ep} train_loss={loss:.4g} val_n={len(yv)}")

    # save torchscript
    if args.save_ts:
        example = torch.zeros((1,4,L), dtype=torch.float32)
        ts = torch.jit.trace(model, example)
        ts.save(str(out/"model.ts"))

    torch.save(model.state_dict(), out/"model.pt")
    (out/"model_meta.json").write_text(json.dumps({"L": int(L)}, indent=2))

    print(f"[DONE] saved -> {out}")

if __name__ == "__main__":
    main()
