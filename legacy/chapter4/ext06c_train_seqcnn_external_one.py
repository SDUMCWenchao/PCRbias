#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, shutil
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr

import torch
import torch.nn as nn
import torch.optim as optim


def save_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def rmse(y, p):
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    if y.size == 0 or p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    if y.size == 0 or p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    if y.size < 2:
        return float("nan")
    ssr = np.sum((y - p) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ssr / sst) if sst > 0 else float("nan")


def safe_spearman(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if y.size < 2 or p.size < 2:
        return float("nan")
    if np.all(y == y[0]) or np.all(p == p[0]):
        return float("nan")
    try:
        return float(spearmanr(y, p).correlation)
    except Exception:
        return float("nan")


def safe_pearson(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if y.size < 2 or p.size < 2:
        return float("nan")
    if np.all(y == y[0]) or np.all(p == p[0]):
        return float("nan")
    try:
        return float(pearsonr(y, p)[0])
    except Exception:
        return float("nan")


class SeqCNN(nn.Module):
    def __init__(self, c1=64, c2=64, k1=7, k2=7, drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, c1, kernel_size=k1, padding=k1 // 2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(
            nn.Linear(c2, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,F)
        x = self.act(self.conv1(x))
        x = self.drop(x)
        x = self.act(self.conv2(x))
        x = self.drop(x)
        x = torch.amax(x, dim=2)  # global max pool
        return self.head(x).squeeze(1)


def iter_batches_csr(X: sparse.csr_matrix, y: np.ndarray, batch: int, shuffle: bool, seed: int):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for i in range(0, n, batch):
        j = idx[i:i + batch]
        xb = torch.from_numpy(X[j].toarray().astype(np.float32))
        yb = torch.from_numpy(y[j].astype(np.float32))
        yield xb, yb


@torch.no_grad()
def predict(model, X: sparse.csr_matrix, batch: int):
    if X.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i + batch].toarray().astype(np.float32))
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.asarray([], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--torch_threads", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = Path(args.dataset_dir)
    out = Path(args.out_dir)

    # resume/skip logic: need BOTH metrics and model.pt
    if out.exists():
        if (out / "metrics.json").exists() and (out / "model.pt").exists():
            print(f"[SKIP] done: {out}")
            return
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    Xtr = sparse.load_npz(d / "X_train.npz").tocsr()
    Xva = sparse.load_npz(d / "X_val.npz").tocsr()
    Xte = sparse.load_npz(d / "X_test.npz").tocsr()
    ytr = np.load(d / "y_train.npy").astype(np.float32)
    yva = np.load(d / "y_val.npy").astype(np.float32)
    yte = np.load(d / "y_test.npy").astype(np.float32)

    if Xtr.shape[0] == 0:
        (out / "metrics.json").write_text(json.dumps({
            "model": "seqcnn",
            "dataset_dir": str(d),
            "status": "skipped",
            "reason": "empty_train_split",
            "X_train_shape": list(Xtr.shape),
            "X_val_shape": list(Xva.shape),
            "X_test_shape": list(Xte.shape),
        }, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[SKIP] empty train split: {out}")
        return

    val_nonempty = (Xva.shape[0] > 0)
    # if val empty, use train as early-stopping signal but record this in metrics
    Xva_fit, yva_fit = (Xva, yva) if val_nonempty else (Xtr, ytr)

    model = SeqCNN()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best = float("inf")
    best_state = None
    bad = 0
    hist = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in iter_batches_csr(Xtr, ytr, args.batch_size, shuffle=True, seed=args.seed + ep):
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        pva_fit = predict(model, Xva_fit, args.batch_size)
        cur = rmse(yva_fit, pva_fit)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)) if losses else float("nan"), "val_rmse_fit": cur})

        if cur < best - 1e-6:
            best = cur
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ptr = predict(model, Xtr, args.batch_size)
    pva = predict(model, Xva, args.batch_size)
    pte = predict(model, Xte, args.batch_size)

    def pack(y, p):
        return {
            "rmse": rmse(y, p),
            "mae": mae(y, p),
            "r2": r2(y, p),
            "spearman": safe_spearman(y, p),
            "pearson": safe_pearson(y, p),
            "n": int(len(y)),
        }

    metrics = {
        "model": "seqcnn",
        "dataset_dir": str(d),
        "val_nonempty": bool(val_nonempty),
        "val_fallback_to_train_for_earlystop": (not val_nonempty),
        "train": pack(ytr, ptr),
        "val": pack(yva, pva),
        "test": pack(yte, pte),
        "history": hist,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    save_tsv(out / "pred_train.tsv", ["y_true", "y_pred"], [[float(a), float(b)] for a, b in zip(ytr, ptr)])
    save_tsv(out / "pred_val.tsv", ["y_true", "y_pred"], [[float(a), float(b)] for a, b in zip(yva, pva)])
    save_tsv(out / "pred_test.tsv", ["y_true", "y_pred"], [[float(a), float(b)] for a, b in zip(yte, pte)])

    torch.save(model.state_dict(), out / "model.pt")
    print(f"[DONE] seqCNN -> {out}")


if __name__ == "__main__":
    main()
