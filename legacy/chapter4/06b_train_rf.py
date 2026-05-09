#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import inspect
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy import sparse
except Exception:
    sparse = None


def load_X(dataset_dir: Path, split: str):
    npz = dataset_dir / f"X_{split}.npz"
    npy = dataset_dir / f"X_{split}.npy"
    if npz.exists():
        if sparse is None:
            raise RuntimeError("scipy is required to load .npz sparse matrices, but scipy is not available.")
        return sparse.load_npz(npz)
    if npy.exists():
        return np.load(npy)
    raise FileNotFoundError(f"X for split={split} not found in {dataset_dir}")


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Spearman = Pearson of ranks
    # robust enough for large arrays without scipy.stats
    a = y_true.astype(np.float64, copy=False)
    b = y_pred.astype(np.float64, copy=False)
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    ar = ar.astype(np.float64, copy=False)
    br = br.astype(np.float64, copy=False)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.sqrt((ar * ar).sum()) * np.sqrt((br * br).sum())
    if denom == 0:
        return float("nan")
    return float((ar * br).sum() / denom)


def supports_param(cls, param_name: str) -> bool:
    try:
        sig = inspect.signature(cls.__init__)
        return param_name in sig.parameters
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="analysis_results/05_ModelInputs/{no_kmer,kmer_only,all}")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", choices=["rf", "etr"], default="rf")

    ap.add_argument("--n_estimators", type=int, default=800)
    ap.add_argument("--n_jobs", type=int, default=64)
    ap.add_argument("--max_features", default="auto", help="auto|sqrt|log2|float(0-1)")
    ap.add_argument("--min_samples_leaf", type=int, default=2)
    ap.add_argument("--max_depth", type=int, default=0, help="0 means None")

    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--max_samples", type=float, default=0.5, help="only if bootstrap and sklearn supports it")
    ap.add_argument("--random_state", type=int, default=1)

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

    # parse max_features
    mf = args.max_features
    if mf not in ("auto", "sqrt", "log2"):
        mf = float(mf)

    max_depth = None if args.max_depth == 0 else int(args.max_depth)

    if args.model == "rf":
        cls = RandomForestRegressor
    else:
        cls = ExtraTreesRegressor

    kwargs = dict(
        n_estimators=int(args.n_estimators),
        n_jobs=int(args.n_jobs),
        random_state=int(args.random_state),
        max_features=mf,
        min_samples_leaf=int(args.min_samples_leaf),
        max_depth=max_depth,
        bootstrap=bool(args.bootstrap),
    )

    # oob_score only makes sense for RF and bootstrap
    if args.model == "rf" and args.bootstrap and supports_param(cls, "oob_score"):
        kwargs["oob_score"] = True

    # max_samples exists only in newer sklearn; auto-detect
    if args.bootstrap and supports_param(cls, "max_samples"):
        kwargs["max_samples"] = float(args.max_samples)
    elif args.bootstrap:
        print("[WARN] sklearn is too old: model does not support max_samples; ignoring --max_samples")

    model = cls(**kwargs)

    # Sparse handling: CSC is typically faster for tree splits
    if sparse is not None and sparse.issparse(Xtr):
        Xtr_fit = Xtr.tocsc()
        Xva_fit = Xva.tocsc()
        Xte_fit = Xte.tocsc()
    else:
        Xtr_fit, Xva_fit, Xte_fit = Xtr, Xva, Xte

    print(f"[INFO] fitting {args.model}  X_train={getattr(Xtr_fit, 'shape', None)}")
    model.fit(Xtr_fit, ytr)

    def eval_split(X, y):
        pred = model.predict(X).astype(np.float32, copy=False)
        r2 = float(r2_score(y, pred))
        mae = float(mean_absolute_error(y, pred))
        mse = float(mean_squared_error(y, pred))  # compatible with old sklearn
        rmse = float(np.sqrt(mse))
        sp = float(spearman_corr(y, pred))
        return pred, {"r2": r2, "mae": mae, "rmse": rmse, "spearman": sp}

    pva, mva = eval_split(Xva_fit, yva)
    pte, mte = eval_split(Xte_fit, yte)

    metrics = {"val": mva, "test": mte}

    if hasattr(model, "oob_score_") and args.bootstrap:
        metrics["oob_r2"] = float(model.oob_score_)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.save(out / "pred_val.npy", pva)
    np.save(out / "pred_test.npy", pte)

    # feature importance
    fn = d / "feature_names.txt"
    if fn.exists() and hasattr(model, "feature_importances_"):
        names = [x.strip() for x in fn.read_text(encoding="utf-8").splitlines() if x.strip()]
        imp = np.asarray(model.feature_importances_, dtype=np.float64)
        k = min(len(names), len(imp))
        order = np.argsort(-imp[:k])

        with (out / "feature_importance.tsv").open("w", encoding="utf-8") as w:
            w.write("rank\tfeature\timportance\n")
            for i, j in enumerate(order[:1000], start=1):
                w.write(f"{i}\t{names[j]}\t{imp[j]:.10g}\n")

    # save model
    import joblib
    joblib.dump(model, out / f"model_{args.model}.joblib")

    print("[DONE] metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
