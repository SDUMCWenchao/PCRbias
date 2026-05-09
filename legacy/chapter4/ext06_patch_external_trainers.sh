#!/usr/bin/env bash
set -euo pipefail

ROOT="/datapool/zhangw/duwenchao/var/2511_PCR_Bias/scripts"

############################
# RF (robust empty splits + silent corr)
############################
cat > "${ROOT}/ext06a_train_rf_external_one.py" <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math, shutil, warnings
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def _nan():
    return float("nan")


def rmse(y, p):
    if len(y) == 0:
        return _nan()
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    if len(y) == 0:
        return _nan()
    return float(mean_absolute_error(y, p))


def r2(y, p):
    if len(y) < 2:
        return _nan()
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    sst = np.sum((y - np.mean(y)) ** 2)
    if sst <= 0:
        return _nan()
    ssr = np.sum((y - p) ** 2)
    return float(1.0 - ssr / sst)


def safe_spearman(y, p):
    if len(y) < 2:
        return _nan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(spearmanr(y, p).correlation)
        except Exception:
            return _nan()


def safe_pearson(y, p):
    if len(y) < 2:
        return _nan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(pearsonr(y, p)[0])
        except Exception:
            return _nan()


def save_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_preds(path: Path, y, p):
    # always write file, even if empty
    rows = []
    if len(y) > 0:
        rows = [[float(a), float(b)] for a, b in zip(y, p)]
    save_tsv(path, ["y_true", "y_pred"], rows)


def load_names(d: Path, nfeat: int):
    fp = d / "feature_names.tsv"
    if fp.exists():
        names = [x.strip() for x in fp.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(names) == nfeat:
            return names
    return [f"f{i}" for i in range(nfeat)]


def eval_split(rf, X, y, out_dir: Path, split: str):
    n = int(X.shape[0])
    if n == 0:
        write_preds(out_dir / f"pred_{split}.tsv", [], [])
        return {"rmse": _nan(), "mae": _nan(), "r2": _nan(), "spearman": _nan(), "pearson": _nan(), "n": 0}

    p = rf.predict(X)
    write_preds(out_dir / f"pred_{split}.tsv", y, p)
    return {
        "rmse": rmse(y, p),
        "mae": mae(y, p),
        "r2": r2(y, p),
        "spearman": safe_spearman(y, p),
        "pearson": safe_pearson(y, p),
        "n": n,
    }


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

    if out.exists():
        if args.force:
            shutil.rmtree(out)
        else:
            raise RuntimeError(f"[BAD] out_dir exists (use --force): {out}")
    out.mkdir(parents=True, exist_ok=True)

    Xtr = sparse.load_npz(d / "X_train.npz").tocsr()
    Xva = sparse.load_npz(d / "X_val.npz").tocsr()
    Xte = sparse.load_npz(d / "X_test.npz").tocsr()
    ytr = np.load(d / "y_train.npy")
    yva = np.load(d / "y_val.npy")
    yte = np.load(d / "y_test.npy")

    if Xtr.shape[0] == 0:
        raise RuntimeError(f"[BAD] train split is empty: {d}")

    names = load_names(d, Xtr.shape[1])

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

    metrics = {
        "model": "rf",
        "dataset_dir": str(d),
        "X_train_shape": list(Xtr.shape),
        "X_val_shape": list(Xva.shape),
        "X_test_shape": list(Xte.shape),
        "train": eval_split(rf, Xtr, ytr, out, "train"),
        "val": eval_split(rf, Xva, yva, out, "val"),
        "test": eval_split(rf, Xte, yte, out, "test"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    joblib.dump(rf, out / "model.joblib")

    imp = rf.feature_importances_
    rows = sorted([[names[i], float(imp[i])] for i in range(len(names))], key=lambda x: x[1], reverse=True)
    save_tsv(out / "feature_importance.tsv", ["feature", "importance"], rows)

    print(f"[DONE] RF -> {out}")


if __name__ == "__main__":
    main()
PY
chmod +x "${ROOT}/ext06a_train_rf_external_one.py"


############################
# XGB (use xgboost.train for old versions + robust empty splits)
############################
cat > "${ROOT}/ext06b_train_xgb_external_one.py" <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math, shutil, warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb


def _nan():
    return float("nan")


def rmse(y, p):
    if len(y) == 0:
        return _nan()
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    if len(y) == 0:
        return _nan()
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    if len(y) < 2:
        return _nan()
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    sst = np.sum((y - np.mean(y)) ** 2)
    if sst <= 0:
        return _nan()
    ssr = np.sum((y - p) ** 2)
    return float(1.0 - ssr / sst)


def safe_spearman(y, p):
    if len(y) < 2:
        return _nan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(spearmanr(y, p).correlation)
        except Exception:
            return _nan()


def safe_pearson(y, p):
    if len(y) < 2:
        return _nan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(pearsonr(y, p)[0])
        except Exception:
            return _nan()


def save_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_preds(path: Path, y, p):
    rows = []
    if len(y) > 0:
        rows = [[float(a), float(b)] for a, b in zip(y, p)]
    save_tsv(path, ["y_true", "y_pred"], rows)


def load_names(d: Path, nfeat: int):
    fp = d / "feature_names.tsv"
    if fp.exists():
        names = [x.strip() for x in fp.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(names) == nfeat:
            return names
    return [f"f{i}" for i in range(nfeat)]


def predict_best(booster: xgb.Booster, dm: xgb.DMatrix):
    bi = getattr(booster, "best_iteration", None)
    if bi is not None and isinstance(bi, int) and bi >= 0:
        try:
            return booster.predict(dm, iteration_range=(0, bi + 1))
        except TypeError:
            pass
        try:
            nt = getattr(booster, "best_ntree_limit", None)
            if nt:
                return booster.predict(dm, ntree_limit=nt)
        except Exception:
            pass
    return booster.predict(dm)


def eval_split(booster, X, y, out_dir: Path, split: str, feature_names):
    n = int(X.shape[0])
    if n == 0:
        write_preds(out_dir / f"pred_{split}.tsv", [], [])
        return {"rmse": _nan(), "mae": _nan(), "r2": _nan(), "spearman": _nan(), "pearson": _nan(), "n": 0}
    dm = xgb.DMatrix(X, label=y, feature_names=feature_names)
    p = predict_best(booster, dm)
    write_preds(out_dir / f"pred_{split}.tsv", y, p)
    return {
        "rmse": rmse(y, p),
        "mae": mae(y, p),
        "r2": r2(y, p),
        "spearman": safe_spearman(y, p),
        "pearson": safe_pearson(y, p),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_jobs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--n_estimators", type=int, default=4000)          # num_boost_round
    ap.add_argument("--learning_rate", type=float, default=0.03)       # eta
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--min_child_weight", type=float, default=1.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)
    args = ap.parse_args()

    d = Path(args.dataset_dir)
    out = Path(args.out_dir)

    if out.exists():
        if args.force:
            shutil.rmtree(out)
        else:
            raise RuntimeError(f"[BAD] out_dir exists (use --force): {out}")
    out.mkdir(parents=True, exist_ok=True)

    Xtr = sparse.load_npz(d / "X_train.npz").tocsr()
    Xva = sparse.load_npz(d / "X_val.npz").tocsr()
    Xte = sparse.load_npz(d / "X_test.npz").tocsr()
    ytr = np.load(d / "y_train.npy")
    yva = np.load(d / "y_val.npy")
    yte = np.load(d / "y_test.npy")

    if Xtr.shape[0] == 0:
        raise RuntimeError(f"[BAD] train split is empty: {d}")

    feature_names = load_names(d, Xtr.shape[1])

    dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_names)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.reg_lambda,
        "min_child_weight": args.min_child_weight,
        "seed": args.seed,
        "nthread": args.n_jobs,
        "tree_method": "hist",
    }

    evals = [(dtrain, "train")]
    if Xva.shape[0] > 0:
        dval = xgb.DMatrix(Xva, label=yva, feature_names=feature_names)
        evals.append((dval, "val"))
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.n_estimators,
            evals=evals,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=False,
        )
    else:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.n_estimators,
            evals=evals,
            verbose_eval=False,
        )

    metrics = {
        "model": "xgb",
        "dataset_dir": str(d),
        "best_iteration": int(getattr(booster, "best_iteration", -1) or -1),
        "train": eval_split(booster, Xtr, ytr, out, "train", feature_names),
        "val": eval_split(booster, Xva, yva, out, "val", feature_names),
        "test": eval_split(booster, Xte, yte, out, "test", feature_names),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    booster.save_model(str(out / "model.json"))

    # feature importance (gain/weight) robust mapping f{idx} -> name
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")

    rows = []
    for i, nm in enumerate(feature_names):
        key = f"f{i}"
        rows.append([nm, float(gain.get(key, 0.0)), float(weight.get(key, 0.0))])
    rows.sort(key=lambda x: x[1], reverse=True)
    save_tsv(out / "feature_importance.tsv", ["feature", "gain", "weight"], rows)

    print(f"[DONE] XGB -> {out}")


if __name__ == "__main__":
    main()
PY
chmod +x "${ROOT}/ext06b_train_xgb_external_one.py"


############################
# seqCNN (robust empty splits)
############################
cat > "${ROOT}/ext06c_train_seqcnn_external_one.py" <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, shutil, warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, pearsonr

import torch
import torch.nn as nn
import torch.optim as optim


def _nan():
    return float("nan")


def save_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_preds(path: Path, y, p):
    rows = []
    if len(y) > 0:
        rows = [[float(a), float(b)] for a, b in zip(y, p)]
    save_tsv(path, ["y_true", "y_pred"], rows)


def rmse(y, p):
    if len(y) == 0:
        return _nan()
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    if len(y) == 0:
        return _nan()
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    if len(y) < 2:
        return _nan()
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    sst = np.sum((y - np.mean(y)) ** 2)
    if sst <= 0:
        return _nan()
    ssr = np.sum((y - p) ** 2)
    return float(1.0 - ssr / sst)


def safe_spearman(y, p):
    if len(y) < 2:
        return _nan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(spearmanr(y, p).correlation)
        except Exception:
            return _nan()


def safe_pearson(y, p):
    if len(y) < 2:
        return _nan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(pearsonr(y, p)[0])
        except Exception:
            return _nan()


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
        x = x.unsqueeze(1)   # (B,1,F)
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
        return np.zeros((0,), dtype=np.float32)
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i:i + batch].toarray().astype(np.float32))
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def pack_metrics(y, p):
    return {
        "rmse": rmse(y, p),
        "mae": mae(y, p),
        "r2": r2(y, p),
        "spearman": safe_spearman(y, p),
        "pearson": safe_pearson(y, p),
        "n": int(len(y)),
    }


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

    if out.exists():
        if args.force:
            shutil.rmtree(out)
        else:
            raise RuntimeError(f"[BAD] out_dir exists (use --force): {out}")
    out.mkdir(parents=True, exist_ok=True)

    Xtr = sparse.load_npz(d / "X_train.npz").tocsr()
    Xva = sparse.load_npz(d / "X_val.npz").tocsr()
    Xte = sparse.load_npz(d / "X_test.npz").tocsr()
    ytr = np.load(d / "y_train.npy").astype(np.float32)
    yva = np.load(d / "y_val.npy").astype(np.float32)
    yte = np.load(d / "y_test.npy").astype(np.float32)

    if Xtr.shape[0] == 0:
        raise RuntimeError(f"[BAD] train split is empty: {d}")

    model = SeqCNN()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    use_val = (Xva.shape[0] > 0)
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

        # early stopping only if val exists
        if use_val:
            pva = predict(model, Xva, args.batch_size)
            cur = rmse(yva, pva)
            hist.append({"epoch": ep, "train_loss": float(np.mean(losses)) if losses else _nan(), "val_rmse": cur})
            if cur < best - 1e-6:
                best = cur
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
        else:
            # no val: just log train loss, no early stop
            hist.append({"epoch": ep, "train_loss": float(np.mean(losses)) if losses else _nan(), "val_rmse": _nan()})

    if best_state is not None:
        model.load_state_dict(best_state)

    ptr = predict(model, Xtr, args.batch_size)
    pva = predict(model, Xva, args.batch_size)
    pte = predict(model, Xte, args.batch_size)

    metrics = {
        "model": "seqcnn",
        "dataset_dir": str(d),
        "train": pack_metrics(ytr, ptr),
        "val": pack_metrics(yva, pva),
        "test": pack_metrics(yte, pte),
        "history": hist,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    write_preds(out / "pred_train.tsv", ytr, ptr)
    write_preds(out / "pred_val.tsv", yva, pva)
    write_preds(out / "pred_test.tsv", yte, pte)

    torch.save(model.state_dict(), out / "model.pt")
    print(f"[DONE] seqCNN -> {out}")


if __name__ == "__main__":
    main()
PY
chmod +x "${ROOT}/ext06c_train_seqcnn_external_one.py"

echo "[DONE] patched 3 trainers:"
echo "  - ext06a_train_rf_external_one.py"
echo "  - ext06b_train_xgb_external_one.py"
echo "  - ext06c_train_seqcnn_external_one.py"
PY

chmod +x "${ROOT}/ext06_patch_external_trainers.sh"
echo "[DONE] patch script ready: ${ROOT}/ext06_patch_external_trainers.sh"
