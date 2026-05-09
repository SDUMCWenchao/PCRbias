#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math, shutil
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
    ap.add_argument("--n_estimators", type=int, default=1200)
    ap.add_argument("--n_jobs", type=int, default=64)
    ap.add_argument("--min_samples_leaf", type=int, default=2)
    ap.add_argument("--max_features", type=float, default=0.3)
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--max_samples", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    d = Path(args.dataset_dir)
    out = Path(args.out_dir)

    # resume/skip logic
    if out.exists():
        if (out / "metrics.json").exists():
            print(f"[SKIP] done: {out}")
            return
        if args.force:
            shutil.rmtree(out)
        else:
            # incomplete dir -> clean and rerun
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
            "model": "rf",
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

    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        max_samples=(args.max_samples if args.bootstrap else None),
    )

    rf.fit(Xtr, ytr)

    def eval_split(X, y, split):
        if X.shape[0] == 0:
            _save_tsv(out / f"pred_{split}.tsv", ["y_true", "y_pred"], [])
            return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                    "spearman": float("nan"), "pearson": float("nan"), "n": 0}
        p = rf.predict(X)
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
        "model": "rf",
        "dataset_dir": str(d),
        "X_train_shape": list(Xtr.shape),
        "X_val_shape": list(Xva.shape),
        "X_test_shape": list(Xte.shape),
        "train": eval_split(Xtr, ytr, "train"),
        "val": eval_split(Xva, yva, "val"),
        "test": eval_split(Xte, yte, "test"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    joblib.dump(rf, out / "model.joblib")

    imp = getattr(rf, "feature_importances_", None)
    if imp is not None:
        rows = sorted([[names[i], float(imp[i])] for i in range(len(names))], key=lambda x: x[1], reverse=True)
        _save_tsv(out / "feature_importance.tsv", ["feature", "importance"], rows)

    print(f"[DONE] RF -> {out}")


if __name__ == "__main__":
    main()
