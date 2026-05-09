#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import sparse
except Exception:
    sparse = None

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_X(dataset_dir: Path, split: str):
    npz = dataset_dir / f"X_{split}.npz"
    npy = dataset_dir / f"X_{split}.npy"
    if npz.exists():
        if sparse is None:
            raise RuntimeError("scipy required for .npz")
        return sparse.load_npz(npz)
    if npy.exists():
        return np.load(npy)
    raise FileNotFoundError(f"X_{split} not found in {dataset_dir}")


def supports_param(cls, param_name: str) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
        return param_name in sig.parameters
    except Exception:
        return False


def spearman_fast(y, p):
    # rank corr without scipy.stats
    y = y.astype(np.float64, copy=False)
    p = p.astype(np.float64, copy=False)
    yr = np.argsort(np.argsort(y)).astype(np.float64, copy=False)
    pr = np.argsort(np.argsort(p)).astype(np.float64, copy=False)
    yr -= yr.mean()
    pr -= pr.mean()
    denom = np.sqrt((yr * yr).sum()) * np.sqrt((pr * pr).sum())
    if denom == 0:
        return float("nan")
    return float((yr * pr).sum() / denom)


def metrics(y, pred):
    mae = float(mean_absolute_error(y, pred))
    mse = float(mean_squared_error(y, pred))  # old sklearn compatible
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y, pred))
    sp = float(spearman_fast(y, pred))
    acc_sign = float(np.mean((y > 0) == (pred > 0)))
    return {"r2": r2, "mae": mae, "rmse": rmse, "spearman": sp, "sign_acc": acc_sign}


def pairwise_spearman(meta_df: pd.DataFrame, y: np.ndarray, pred: np.ndarray, min_n: int = 30):
    # within each pair_id, compute Spearman; then average
    df = meta_df[["pair_id"]].copy()
    df["y"] = y
    df["pred"] = pred
    out = []
    for pid, g in df.groupby("pair_id", sort=False):
        if len(g) < min_n:
            continue
        out.append(spearman_fast(g["y"].to_numpy(dtype=np.float32), g["pred"].to_numpy(dtype=np.float32)))
    if not out:
        return {"pair_spearman_mean": float("nan"), "pair_spearman_n": 0}
    return {"pair_spearman_mean": float(np.nanmean(out)), "pair_spearman_n": int(len(out))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", choices=["rf", "etr"], default="rf")

    ap.add_argument("--n_estimators", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=64)
    ap.add_argument("--min_samples_leaf", type=int, default=5)
    ap.add_argument("--max_features", type=float, default=0.25)
    ap.add_argument("--max_depth", type=int, default=0)

    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--max_samples", type=float, default=0.6)
    ap.add_argument("--random_state", type=int, default=1)

    ap.add_argument("--min_pair_n", type=int, default=30)
    args = ap.parse_args()

    d = Path(args.dataset_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    Xtr = load_X(d, "train")
    Xva = load_X(d, "val")
    Xte = load_X(d, "test")

    ytr = np.load(d / "y_train.npy").astype(np.float32, copy=False)
    yva = np.load(d / "y_val.npy").astype(np.float32, copy=False)
    yte = np.load(d / "y_test.npy").astype(np.float32, copy=False)

    wtr = np.load(d / "w_train.npy").astype(np.float32, copy=False) if (d / "w_train.npy").exists() else np.ones_like(ytr)
    wva = np.load(d / "w_val.npy").astype(np.float32, copy=False) if (d / "w_val.npy").exists() else np.ones_like(yva)
    wte = np.load(d / "w_test.npy").astype(np.float32, copy=False) if (d / "w_test.npy").exists() else np.ones_like(yte)

    mva = pd.read_csv(d / "meta_val.tsv.gz", sep="\t")
    mte = pd.read_csv(d / "meta_test.tsv.gz", sep="\t")

    cls = RandomForestRegressor if args.model == "rf" else ExtraTreesRegressor

    kwargs = dict(
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state,
        max_depth=None if args.max_depth == 0 else args.max_depth,
        bootstrap=bool(args.bootstrap),
    )

    if args.bootstrap and supports_param(cls, "max_samples"):
        kwargs["max_samples"] = float(args.max_samples)
    elif args.bootstrap:
        print("[WARN] sklearn too old: max_samples not supported; ignoring")

    if args.model == "rf" and args.bootstrap and supports_param(cls, "oob_score"):
        kwargs["oob_score"] = True

    model = cls(**kwargs)

    if sparse is not None and sparse.issparse(Xtr):
        Xtr_fit, Xva_fit, Xte_fit = Xtr.tocsc(), Xva.tocsc(), Xte.tocsc()
    else:
        Xtr_fit, Xva_fit, Xte_fit = Xtr, Xva, Xte

    print(f"[INFO] fit {args.model} X_train={getattr(Xtr_fit,'shape',None)}")
    model.fit(Xtr_fit, ytr, sample_weight=wtr)

    pva = model.predict(Xva_fit).astype(np.float32, copy=False)
    pte = model.predict(Xte_fit).astype(np.float32, copy=False)

    out_metrics = {
        "val": metrics(yva, pva),
        "test": metrics(yte, pte),
        "val_pair": pairwise_spearman(mva, yva, pva, min_n=args.min_pair_n),
        "test_pair": pairwise_spearman(mte, yte, pte, min_n=args.min_pair_n),
        "oob_r2": float(getattr(model, "oob_score_", np.nan)),
        "config": vars(args),
    }

    (out / "metrics.json").write_text(json.dumps(out_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # dump preds (for later plots)
    mva2 = mva.copy()
    mva2["y"] = yva
    mva2["pred"] = pva
    mva2.to_csv(out / "pred_val.tsv.gz", sep="\t", index=False, compression="gzip")

    mte2 = mte.copy()
    mte2["y"] = yte
    mte2["pred"] = pte
    mte2.to_csv(out / "pred_test.tsv.gz", sep="\t", index=False, compression="gzip")

    # feature importance (gini)
    fn = d / "feature_names.txt"
    if fn.exists() and hasattr(model, "feature_importances_"):
        names = [x.strip() for x in fn.read_text(encoding="utf-8").splitlines() if x.strip()]
        imp = np.asarray(model.feature_importances_, dtype=np.float64)
        k = min(len(names), len(imp))
        order = np.argsort(-imp[:k])
        with (out / "feature_importance.tsv").open("w", encoding="utf-8") as w:
            w.write("rank\tfeature\timportance\n")
            for i, j in enumerate(order[:2000], start=1):
                w.write(f"{i}\t{names[j]}\t{imp[j]:.10g}\n")

    import joblib
    joblib.dump(model, out / f"model_{args.model}.joblib")

    print(f"[DONE] {out}")
    print(json.dumps(out_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
