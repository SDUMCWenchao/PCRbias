#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

def load_X(ds: Path, split: str):
    npz = ds / f"X_{split}.npz"
    npy = ds / f"X_{split}.npy"
    if npz.exists():
        return sparse.load_npz(npz).tocsr().astype(np.float32)
    if npy.exists():
        X = np.load(npy, allow_pickle=False)
        return sparse.csr_matrix(X.astype(np.float32, copy=False))
    raise FileNotFoundError(f"missing X_{split}.npz or X_{split}.npy in {ds}")

def r2(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ssr = np.sum((y - p) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ssr / (sst + 1e-12))

def rmse(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.sqrt(np.mean((y - p) ** 2)))

def mae(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean(np.abs(y - p)))

def sign_acc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean((y >= 0) == (p >= 0)))

def spearman(y, p):
    rr = spearmanr(y, p)
    return float(rr.correlation) if np.isfinite(rr.correlation) else float("nan")

def pair_spearman(meta_df, y, p, min_pair_n=30):
    if "pair_id" not in meta_df.columns:
        return float("nan"), 0
    out = []
    for pid, g in meta_df.groupby("pair_id"):
        idx = g.index.values
        if len(idx) < min_pair_n:
            continue
        rr = spearmanr(y[idx], p[idx])
        if np.isfinite(rr.correlation):
            out.append(float(rr.correlation))
    if not out:
        return float("nan"), 0
    return float(np.mean(out)), int(len(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nthread", type=int, default=16)

    ap.add_argument("--max_depth", type=int, default=7)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.6)
    ap.add_argument("--min_child_weight", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--lambda_l2", type=float, default=1.0)
    ap.add_argument("--alpha_l1", type=float, default=0.0)

    ap.add_argument("--num_boost_round", type=int, default=5000)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)

    ap.add_argument("--min_pair_n", type=int, default=10)
    ap.add_argument("--y_clip", type=float, default=6.0, help="clip y_pred into [-y_clip,y_clip] for reporting")

    args = ap.parse_args()

    import xgboost as xgb

    ds = Path(args.dataset_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    Xtr = load_X(ds, "train")
    Xva = load_X(ds, "val")
    Xte = load_X(ds, "test")

    ytr = np.load(ds / "y_train.npy").astype(np.float32, copy=False)
    yva = np.load(ds / "y_val.npy").astype(np.float32, copy=False)
    yte = np.load(ds / "y_test.npy").astype(np.float32, copy=False)

    wtr = np.load(ds / "w_train.npy").astype(np.float32, copy=False)
    wva = np.load(ds / "w_val.npy").astype(np.float32, copy=False)
    wte = np.load(ds / "w_test.npy").astype(np.float32, copy=False)

    meta_va = pd.read_csv(ds / "meta_val.tsv.gz", sep="\t", compression="gzip")
    meta_te = pd.read_csv(ds / "meta_test.tsv.gz", sep="\t", compression="gzip")

    # Feature names (optional but recommended)
    feat_names = None
    fn = ds / "feature_names.txt"
    if fn.exists():
        feat_names = [x.strip() for x in fn.read_text(encoding="utf-8").splitlines() if x.strip()]

    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr, feature_names=feat_names)
    dva = xgb.DMatrix(Xva, label=yva, weight=wva, feature_names=feat_names)
    dte = xgb.DMatrix(Xte, label=yte, weight=wte, feature_names=feat_names)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": args.seed,
        "nthread": args.nthread,
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "lambda": args.lambda_l2,
        "alpha": args.alpha_l1,
        "tree_method": "hist",
    }

    t0 = time.time()
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=args.num_boost_round,
        evals=[(dva, "val")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=100
    )
    train_sec = time.time() - t0

    # Predict
    pv = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    pt = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))

    if args.y_clip > 0:
        pv = np.clip(pv, -args.y_clip, args.y_clip)
        pt = np.clip(pt, -args.y_clip, args.y_clip)

    # Pair metrics
    vps, vpn = pair_spearman(meta_va, yva, pv, min_pair_n=args.min_pair_n)
    tps, tpn = pair_spearman(meta_te, yte, pt, min_pair_n=args.min_pair_n)

    metrics = {
        "val": {
            "r2": r2(yva, pv),
            "rmse": rmse(yva, pv),
            "mae": mae(yva, pv),
            "spearman": spearman(yva, pv),
            "sign_acc": sign_acc(yva, pv),
            "n": int(len(yva)),
        },
        "test": {
            "r2": r2(yte, pt),
            "rmse": rmse(yte, pt),
            "mae": mae(yte, pt),
            "spearman": spearman(yte, pt),
            "sign_acc": sign_acc(yte, pt),
            "n": int(len(yte)),
        },
        "val_pair": {"pair_spearman_mean": vps, "pair_spearman_n": vpn, "min_pair_n": args.min_pair_n},
        "test_pair": {"pair_spearman_mean": tps, "pair_spearman_n": tpn, "min_pair_n": args.min_pair_n},
        "train_time_sec": float(train_sec),
        "best_iteration": int(booster.best_iteration),
        "config": vars(args),
        "xgb_params": params,
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Save model
    booster.save_model(str(out / "model.json"))

    # Save preds
    def dump_pred(split, meta, y, p):
        df = meta.copy()
        df["y_true"] = y
        df["y_pred"] = p
        df.to_csv(out / f"pred_{split}.tsv.gz", sep="\t", index=False, compression="gzip")

    dump_pred("val", meta_va, yva, pv)
    dump_pred("test", meta_te, yte, pt)

    # Feature importance
    if feat_names is None:
        # fallback names
        feat_names = [f"f{i}" for i in range(Xtr.shape[1])]

    gain = booster.get_score(importance_type="gain")
    cover = booster.get_score(importance_type="cover")
    weight = booster.get_score(importance_type="weight")

    rows = []
    for i, name in enumerate(feat_names):
        key = name if name in gain else f"f{i}"
        rows.append({
            "feature": name,
            "gain": float(gain.get(key, 0.0)),
            "cover": float(cover.get(key, 0.0)),
            "weight": float(weight.get(key, 0.0)),
        })
    imp = pd.DataFrame(rows).sort_values(["gain","weight","cover"], ascending=False)
    imp.to_csv(out / "feature_importance.tsv", sep="\t", index=False)

    print(f"[DONE] XGB -> {out}  test_r2={metrics['test']['r2']:.4f}  test_spear={metrics['test']['spearman']:.4f}")

if __name__ == "__main__":
    main()
