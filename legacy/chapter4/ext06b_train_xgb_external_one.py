#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math, shutil
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb


def _rmse(y, p):
    return float(math.sqrt(mean_squared_error(y, p)))


def _safe_corr_spearman(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if y.size < 2 or p.size < 2:
        return float("nan")
    if np.all(y == y[0]) or np.all(p == p[0]):
        return float("nan")
    try:
        return float(spearmanr(y, p).correlation)
    except Exception:
        return float("nan")


def _safe_corr_pearson(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if y.size < 2 or p.size < 2:
        return float("nan")
    if np.all(y == y[0]) or np.all(p == p[0]):
        return float("nan")
    try:
        return float(pearsonr(y, p)[0])
    except Exception:
        return float("nan")


def _safe_r2(y, p):
    if len(y) < 2:
        return float("nan")
    try:
        return float(r2_score(y, p))
    except Exception:
        return float("nan")


def _save_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def _load_names(d: Path, nfeat: int):
    fp = d / "feature_names.tsv"
    if fp.exists():
        names = [x.strip() for x in fp.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(names) == nfeat:
            return names
    return [f"f{i}" for i in range(nfeat)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_jobs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--n_estimators", type=int, default=4000)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--min_child_weight", type=float, default=1.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)
    args = ap.parse_args()

    d = Path(args.dataset_dir)
    out = Path(args.out_dir)

    # resume/skip logic
    if out.exists():
        if (out / "metrics.json").exists():
            print(f"[SKIP] done: {out}")
            return
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    Xtr = sparse.load_npz(d / "X_train.npz").tocsr()
    Xva = sparse.load_npz(d / "X_val.npz").tocsr()
    Xte = sparse.load_npz(d / "X_test.npz").tocsr()
    ytr = np.load(d / "y_train.npy")
    yva = np.load(d / "y_val.npy")
    yte = np.load(d / "y_test.npy")

    if Xtr.shape[0] == 0:
        (out / "metrics.json").write_text(json.dumps({
            "model": "xgb",
            "dataset_dir": str(d),
            "status": "skipped",
            "reason": "empty_train_split",
            "X_train_shape": list(Xtr.shape),
            "X_val_shape": list(Xva.shape),
            "X_test_shape": list(Xte.shape),
        }, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[SKIP] empty train split: {out}")
        return

    names = _load_names(d, Xtr.shape[1])

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    evals = []
    val_nonempty = (Xva.shape[0] > 0)
    if val_nonempty:
        dval = xgb.DMatrix(Xva, label=yva)
        evals = [(dtrain, "train"), (dval, "val")]
    else:
        evals = [(dtrain, "train")]

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "lambda": float(args.reg_lambda),
        "min_child_weight": float(args.min_child_weight),
        "seed": int(args.seed),
        "nthread": int(args.n_jobs),
        "tree_method": "hist",
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(args.n_estimators),
        evals=evals,
        early_stopping_rounds=(int(args.early_stopping_rounds) if val_nonempty else None),
        verbose_eval=False,
    )

    def eval_split(X, y, split):
        if X.shape[0] == 0:
            _save_tsv(out / f"pred_{split}.tsv", ["y_true", "y_pred"], [])
            return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                    "spearman": float("nan"), "pearson": float("nan"), "n": 0}
        dm = xgb.DMatrix(X)
        p = booster.predict(dm)
        _save_tsv(out / f"pred_{split}.tsv", ["y_true", "y_pred"], [[float(a), float(b)] for a, b in zip(y, p)])
        return {
            "rmse": _rmse(y, p),
            "mae": float(mean_absolute_error(y, p)),
            "r2": _safe_r2(y, p),
            "spearman": _safe_corr_spearman(y, p),
            "pearson": _safe_corr_pearson(y, p),
            "n": int(len(y)),
        }

    metrics = {
        "model": "xgb",
        "dataset_dir": str(d),
        "val_nonempty": bool(val_nonempty),
        "best_iteration": int(getattr(booster, "best_iteration", -1) or -1),
        "train": eval_split(Xtr, ytr, "train"),
        "val": eval_split(Xva, yva, "val"),
        "test": eval_split(Xte, yte, "test"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    booster.save_model(str(out / "model.json"))

    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    rows = []
    for i, nm in enumerate(names):
        fid = f"f{i}"
        rows.append([nm, float(gain.get(fid, 0.0)), float(weight.get(fid, 0.0))])
    rows.sort(key=lambda x: x[1], reverse=True)
    _save_tsv(out / "feature_importance.tsv", ["feature", "gain", "weight"], rows)

    print(f"[DONE] XGB -> {out}")


if __name__ == "__main__":
    main()
