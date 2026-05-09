#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = pearsonr = None

try:
    import xgboost as xgb
except Exception:
    xgb = None


def log(msg: str):
    print(msg, flush=True)


def rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def corr(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]; p = p[m]
    if y.size < 20 or spearmanr is None or pearsonr is None:
        return {"spearman_r": None, "spearman_p": None, "pearson_r": None, "pearson_p": None}
    if np.nanmin(y) == np.nanmax(y) or np.nanmin(p) == np.nanmax(p):
        return {"spearman_r": None, "spearman_p": None, "pearson_r": None, "pearson_p": None}
    sr = spearmanr(y, p)
    pr = pearsonr(y, p)
    return {
        "spearman_r": float(sr.correlation) if hasattr(sr, "correlation") else float(sr[0]),
        "spearman_p": float(sr.pvalue) if hasattr(sr, "pvalue") else float(sr[1]),
        "pearson_r": float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0]),
        "pearson_p": float(pr.pvalue) if hasattr(pr, "pvalue") else float(pr[1]),
    }


def load_dataset(dataset_dir: Path):
    Xtr = sparse.load_npz(dataset_dir / "X_train.npz").tocsr()
    ytr = np.load(dataset_dir / "y_train.npy")
    Xva = sparse.load_npz(dataset_dir / "X_val.npz").tocsr()
    yva = np.load(dataset_dir / "y_val.npy")
    Xte = sparse.load_npz(dataset_dir / "X_test.npz").tocsr()
    yte = np.load(dataset_dir / "y_test.npy")

    fn = (dataset_dir / "feature_names.tsv").read_text(encoding="utf-8").strip().splitlines()
    return Xtr, ytr, Xva, yva, Xte, yte, fn


def save_preds(out_dir: Path, split: str, y: np.ndarray, p: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"y": y.astype(np.float32), "pred": p.astype(np.float32)})
    df.to_csv(out_dir / f"preds_{split}.tsv.gz", sep="\t", index=False, compression="gzip")


def write_importance(out_fp: Path, names: list[str], imp: np.ndarray, kind: str):
    df = pd.DataFrame({"feature": names, f"importance_{kind}": imp.astype(np.float64)})
    df = df.sort_values(f"importance_{kind}", ascending=False)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, sep="\t", index=False)


def train_rf(Xtr, ytr, Xva, yva, Xte, yte, feat_names, args, out_dir: Path):
    n_jobs = int(args.n_jobs)
    model = RandomForestRegressor(
        n_estimators=int(args.rf_n_estimators),
        random_state=int(args.seed),
        n_jobs=n_jobs,
        min_samples_leaf=int(args.rf_min_samples_leaf),
        bootstrap=bool(args.rf_bootstrap),
        max_samples=float(args.rf_max_samples) if args.rf_bootstrap else None,
        max_features=float(args.rf_max_features),
    )
    t0 = time.time()
    model.fit(Xtr, ytr)
    fit_sec = time.time() - t0

    p_tr = model.predict(Xtr)
    p_va = model.predict(Xva)
    p_te = model.predict(Xte)

    metrics = {
        "model": "rf",
        "fit_seconds": float(fit_sec),
        "train": {"n": int(len(ytr)), "r2": float(r2_score(ytr, p_tr)), "rmse": rmse(ytr, p_tr), "mae": float(mean_absolute_error(ytr, p_tr)), **corr(ytr, p_tr)},
        "val":   {"n": int(len(yva)), "r2": float(r2_score(yva, p_va)), "rmse": rmse(yva, p_va), "mae": float(mean_absolute_error(yva, p_va)), **corr(yva, p_va)},
        "test":  {"n": int(len(yte)), "r2": float(r2_score(yte, p_te)), "rmse": rmse(yte, p_te), "mae": float(mean_absolute_error(yte, p_te)), **corr(yte, p_te)},
        "params": {
            "n_estimators": int(args.rf_n_estimators),
            "min_samples_leaf": int(args.rf_min_samples_leaf),
            "bootstrap": bool(args.rf_bootstrap),
            "max_samples": float(args.rf_max_samples) if args.rf_bootstrap else None,
            "max_features": float(args.rf_max_features),
            "n_jobs": n_jobs,
            "seed": int(args.seed),
        },
    }

    # outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, out_dir / "model.joblib", compress=3)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_preds(out_dir, "train", ytr, p_tr)
    save_preds(out_dir, "val", yva, p_va)
    save_preds(out_dir, "test", yte, p_te)

    imp = getattr(model, "feature_importances_", None)
    if imp is not None and len(imp) == len(feat_names):
        write_importance(out_dir / "feature_importance.tsv", feat_names, np.asarray(imp), "mdi")

    (out_dir / "DONE.ok").write_text("ok\n", encoding="utf-8")
    log(f"[DONE] RF -> {out_dir}")


def train_xgb(Xtr, ytr, Xva, yva, Xte, yte, feat_names, args, out_dir: Path):
    if xgb is None:
        raise RuntimeError("[BAD] xgboost not installed in this env.")

    n_jobs = int(args.n_jobs)
    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        eta=float(args.xgb_eta),
        max_depth=int(args.xgb_max_depth),
        min_child_weight=float(args.xgb_min_child_weight),
        subsample=float(args.xgb_subsample),
        colsample_bytree=float(args.xgb_colsample),
        reg_lambda=float(args.xgb_lambda),
        reg_alpha=float(args.xgb_alpha),
        tree_method=str(args.xgb_tree_method),
        nthread=n_jobs,
        seed=int(args.seed),
    )

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feat_names)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=feat_names)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=feat_names)

    t0 = time.time()
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=int(args.xgb_num_boost_round),
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=int(args.xgb_early_stopping),
        verbose_eval=False,
    )
    fit_sec = time.time() - t0

    p_tr = booster.predict(dtr, iteration_range=(0, booster.best_iteration + 1))
    p_va = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    p_te = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))

    metrics = {
        "model": "xgb",
        "fit_seconds": float(fit_sec),
        "best_iteration": int(booster.best_iteration),
        "train": {"n": int(len(ytr)), "r2": float(r2_score(ytr, p_tr)), "rmse": rmse(ytr, p_tr), "mae": float(mean_absolute_error(ytr, p_tr)), **corr(ytr, p_tr)},
        "val":   {"n": int(len(yva)), "r2": float(r2_score(yva, p_va)), "rmse": rmse(yva, p_va), "mae": float(mean_absolute_error(yva, p_va)), **corr(yva, p_va)},
        "test":  {"n": int(len(yte)), "r2": float(r2_score(yte, p_te)), "rmse": rmse(yte, p_te), "mae": float(mean_absolute_error(yte, p_te)), **corr(yte, p_te)},
        "params": params | {
            "num_boost_round": int(args.xgb_num_boost_round),
            "early_stopping_rounds": int(args.xgb_early_stopping),
            "n_jobs": n_jobs,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(out_dir / "model.json")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_preds(out_dir, "train", ytr, p_tr)
    save_preds(out_dir, "val", yva, p_va)
    save_preds(out_dir, "test", yte, p_te)

    # importance (gain)
    score = booster.get_score(importance_type="gain")
    imp = np.zeros(len(feat_names), dtype=np.float64)
    f2i = {f: i for i, f in enumerate(feat_names)}
    for k, v in score.items():
        # xgboost feature name might be "f123" if names lost
        if k in f2i:
            imp[f2i[k]] = float(v)
        elif k.startswith("f") and k[1:].isdigit():
            j = int(k[1:])
            if 0 <= j < len(imp):
                imp[j] = float(v)
    write_importance(out_dir / "feature_importance.tsv", feat_names, imp, "gain")

    (out_dir / "DONE.ok").write_text("ok\n", encoding="utf-8")
    log(f"[DONE] XGB -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", choices=["rf", "xgb"], required=True)

    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)

    # RF defaults (你之前用的风格)
    ap.add_argument("--rf_n_estimators", type=int, default=1200)
    ap.add_argument("--rf_min_samples_leaf", type=int, default=2)
    ap.add_argument("--rf_bootstrap", action="store_true")
    ap.add_argument("--rf_max_samples", type=float, default=0.5)
    ap.add_argument("--rf_max_features", type=float, default=0.3)

    # XGB defaults（稳健 + 资源友好）
    ap.add_argument("--xgb_tree_method", default="hist")
    ap.add_argument("--xgb_eta", type=float, default=0.05)
    ap.add_argument("--xgb_max_depth", type=int, default=8)
    ap.add_argument("--xgb_min_child_weight", type=float, default=5.0)
    ap.add_argument("--xgb_subsample", type=float, default=0.8)
    ap.add_argument("--xgb_colsample", type=float, default=0.6)
    ap.add_argument("--xgb_lambda", type=float, default=1.0)
    ap.add_argument("--xgb_alpha", type=float, default=0.0)
    ap.add_argument("--xgb_num_boost_round", type=int, default=5000)
    ap.add_argument("--xgb_early_stopping", type=int, default=150)

    ap.add_argument("--force", action="store_true", help="ignore DONE.ok and rerun")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    done_flag = out_dir / "DONE.ok"
    if done_flag.exists() and not args.force:
        log(f"[SKIP] DONE.ok exists: {out_dir}")
        return

    Xtr, ytr, Xva, yva, Xte, yte, feat_names = load_dataset(dataset_dir)

    # set threads defensively
    os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
    os.environ["MKL_NUM_THREADS"] = str(args.n_jobs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.n_jobs)

    log(f"[INFO] model={args.model} X_train={Xtr.shape} X_val={Xva.shape} X_test={Xte.shape} n_jobs={args.n_jobs}")

    if args.model == "rf":
        train_rf(Xtr, ytr, Xva, yva, Xte, yte, feat_names, args, out_dir)
    else:
        train_xgb(Xtr, ytr, Xva, yva, Xte, yte, feat_names, args, out_dir)


if __name__ == "__main__":
    main()
