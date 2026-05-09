#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import torch

def load_feature_names(dataset_dir: Path, nfeat: int):
    cand = [
        "feature_names.tsv", "feature_names.txt", "feat_cols.tsv",
        "feature_cols.tsv", "columns.tsv", "X_cols.tsv"
    ]
    for fn in cand:
        p = dataset_dir / fn
        if p.exists():
            df = pd.read_csv(p, sep="\t", header=None)
            names = df.iloc[:, 0].astype(str).tolist()
            if len(names) == nfeat:
                return names
    return [f"f{i}" for i in range(nfeat)]

def load_split(dataset_dir: Path, split: str):
    Xp = dataset_dir / f"X_{split}.npz"
    yp = dataset_dir / f"y_{split}.npy"
    if not Xp.exists():
        raise FileNotFoundError(f"[BAD] missing {Xp}")
    if not yp.exists():
        raise FileNotFoundError(f"[BAD] missing {yp}")
    X = sparse.load_npz(Xp).tocsr()
    y = np.load(yp)
    if X.shape[0] != len(y):
        raise RuntimeError(f"[BAD] X/y mismatch: X={X.shape} y={y.shape}")
    return X, y

def try_load_model(model_dir: Path, device: str):
    # 1) TorchScript 优先
    for fn in ["model.ts", "model_scripted.pt", "model_jit.pt"]:
        p = model_dir / fn
        if p.exists():
            m = torch.jit.load(str(p), map_location=device)
            m.eval()
            return m, str(p)

    # 2) 直接 torch.load 一个 Module
    p = model_dir / "model.pt"
    if not p.exists():
        raise FileNotFoundError(f"[BAD] missing {p}")

    obj = torch.load(str(p), map_location=device)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj, str(p)

    # 3) 可能是 dict（state_dict 等）——无法通用重建
    raise RuntimeError(
        "[BAD] model.pt is not a torch.nn.Module or TorchScript.\n"
        "Please save a TorchScript copy during training, e.g.:\n"
        "  scripted = torch.jit.script(model.cpu())\n"
        "  scripted.save(out_dir/'model.ts')\n"
    )

def forward_pred(model, x):
    # 尝试三种常见输入形状
    try:
        y = model(x)
        return y
    except Exception:
        pass
    try:
        y = model(x.unsqueeze(1))
        return y
    except Exception:
        pass
    y = model(x.unsqueeze(-1))
    return y

@torch.no_grad()
def predict(model, X, batch_size=256, device="cpu"):
    out = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        yb = forward_pred(model, xb)
        yb = yb.detach().view(-1).cpu().numpy()
        out.append(yb)
    if not out:
        return np.array([], dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)

def integrated_gradients(model, X, baseline=None, steps=32, batch_size=128, device="cpu"):
    # X: np.ndarray (n, p)
    n, p = X.shape
    if baseline is None:
        baseline = np.zeros((1, p), dtype=np.float32)
    if baseline.shape[0] == 1:
        baseline = np.repeat(baseline, n, axis=0)

    attrs = np.zeros_like(X, dtype=np.float32)

    for i in range(0, n, batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device, requires_grad=True)
        bb = torch.tensor(baseline[i:i+batch_size], dtype=torch.float32, device=device)

        # Riemann sum
        total_grad = torch.zeros_like(xb)
        for s in range(1, steps + 1):
            alpha = float(s) / steps
            xi = bb + alpha * (xb - bb)
            xi.requires_grad_(True)
            yi = forward_pred(model, xi).view(-1).sum()
            grad = torch.autograd.grad(yi, xi, retain_graph=False, create_graph=False)[0]
            total_grad += grad

        avg_grad = total_grad / steps
        ig = (xb - bb) * avg_grad
        attrs[i:i+batch_size] = ig.detach().cpu().numpy().astype(np.float32)

    return attrs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--explain_n", type=int, default=0, help="0=all rows of split")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="", help="default: <model_dir>/attr_tables")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (model_dir / "attr_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    done_flag = out_dir / f"ig_global_{args.split}.tsv.gz"
    if done_flag.exists() and (not args.force):
        print(f"[SKIP] exists: {done_flag}")
        return

    Xsp, y = load_split(dataset_dir, args.split)
    n = Xsp.shape[0]
    if n == 0:
        raise RuntimeError(f"[BAD] split {args.split} has 0 rows: {dataset_dir}")

    idx = np.arange(n)
    if args.explain_n and args.explain_n < n:
        idx = idx[:args.explain_n]

    X = Xsp[idx].toarray().astype(np.float32)
    ys = y[idx].astype(np.float32)

    model, model_path = try_load_model(model_dir, args.device)

    # 预测（用于 local 表）
    pred = predict(model, X, batch_size=max(64, args.batch_size), device=args.device)

    # IG
    attrs = integrated_gradients(
        model, X,
        baseline=np.zeros((1, X.shape[1]), dtype=np.float32),
        steps=args.steps, batch_size=args.batch_size, device=args.device
    )

    nfeat = X.shape[1]
    feat_names = load_feature_names(dataset_dir, nfeat)

    abs_at = np.abs(attrs)
    df_g = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_ig": abs_at.mean(axis=0),
        "mean_ig": attrs.mean(axis=0),
        "std_abs_ig": abs_at.std(axis=0),
    }).sort_values("mean_abs_ig", ascending=False)
    df_g.to_csv(done_flag, sep="\t", index=False, compression="gzip")

    topk = min(args.topk, nfeat)
    rows = []
    for i in range(len(idx)):
        at = attrs[i]
        top = np.argsort(np.abs(at))[::-1][:topk]
        for r, j in enumerate(top, 1):
            rows.append({
                "row_i": int(idx[i]),
                "rank": int(r),
                "feature": feat_names[j],
                "x": float(X[i, j]),
                "ig": float(at[j]),
                "abs_ig": float(abs(at[j])),
                "y_true": float(ys[i]),
                "y_pred": float(pred[i]) if len(pred)==len(idx) else np.nan,
            })
    df_l = pd.DataFrame(rows)
    out_l = out_dir / f"ig_local_top{topk}_{args.split}.tsv.gz"
    df_l.to_csv(out_l, sep="\t", index=False, compression="gzip")

    meta = {
        "dataset_dir": str(dataset_dir),
        "model_dir": str(model_dir),
        "model_loaded_from": model_path,
        "split": args.split,
        "n_explained": int(len(idx)),
        "n_features": int(nfeat),
        "steps": int(args.steps),
        "global_table": str(done_flag),
        "local_table": str(out_l),
    }
    (out_dir / f"ig_meta_{args.split}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[DONE] seqCNN IG -> {out_dir}")

if __name__ == "__main__":
    main()
