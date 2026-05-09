#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def load_X(d: Path, split: str):
    npz = d / f"X_{split}.npz"
    npy = d / f"X_{split}.npy"
    if npz.exists():
        return sparse.load_npz(npz), "npz"
    if npy.exists():
        return np.load(npy, allow_pickle=False), "npy"
    raise FileNotFoundError(f"missing X_{split}.npz or X_{split}.npy in {d}")


def save_X(out: Path, split: str, X, fmt: str):
    if fmt == "npz":
        sparse.save_npz(out / f"X_{split}.npz", X.tocsr())
    else:
        np.save(out / f"X_{split}.npy", np.asarray(X, dtype=np.float32))


def topk_by_abs(y: np.ndarray, frac: float, seed: int = 1):
    n = len(y)
    if n == 0:
        return np.array([], dtype=np.int64)
    k = int(math.ceil(n * frac))
    k = max(1, min(k, n))

    a = np.abs(y)
    # 取 top-k 的阈值（含并列会略多）
    if k == n:
        idx = np.arange(n, dtype=np.int64)
        return idx
    # 用 partition 拿到第 k 大的阈值
    kth = np.partition(a, n - k)[n - k]
    cand = np.where(a >= kth)[0]

    # 如果并列导致 cand 太多，随机下采样到 k（可复现）
    if len(cand) > k:
        rng = np.random.default_rng(seed)
        cand = rng.choice(cand, size=k, replace=False)

    return np.sort(cand.astype(np.int64))


def subset_meta(meta_path: Path, idx: np.ndarray):
    df = pd.read_csv(meta_path, sep="\t", compression="gzip")
    return df.iloc[idx].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3")
    ap.add_argument("--out_root", default="analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--fracs", default="0.01,0.005,0.001", help="top fractions by |y|, e.g. 0.01,0.005,0.001")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    project = Path(args.project_dir)
    in_root = project / args.inputs_root
    out_root = project / args.out_root
    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
    if not fracs:
        raise SystemExit("[ERROR] no fracs provided")

    # 找到所有 dataset 目录（含 config.json + y_train.npy）
    ds_dirs = []
    for d in in_root.rglob("*"):
        if not d.is_dir():
            continue
        if (d / "y_train.npy").exists() and (d / "config.json").exists() and (d / "feature_names.txt").exists():
            ds_dirs.append(d)
    ds_dirs = sorted(ds_dirs)

    if not ds_dirs:
        raise SystemExit(f"[ERROR] no datasets found under {in_root}")

    print(f"[INFO] datasets={len(ds_dirs)}  fracs={fracs}")

    for d in ds_dirs:
        rel = d.relative_to(in_root)  # species/locus/variant
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))

        # load common files
        feat_names = (d / "feature_names.txt").read_text(encoding="utf-8")
        for frac in fracs:
            tag = f"top{int(round(frac*100000))/1000:g}p".replace(".", "p")  # 0.5% -> top0p5p
            out = out_root / tag / rel
            out.mkdir(parents=True, exist_ok=True)

            new_cfg = dict(cfg)
            new_cfg["topbias_frac"] = frac
            new_cfg["topbias_rule"] = "per-split top fraction by |y| (true log2fc)"
            new_cfg["topbias_seed"] = args.seed

            for split in ["train", "val", "test"]:
                y = np.load(d / f"y_{split}.npy", allow_pickle=False).astype(np.float32, copy=False)
                w = np.load(d / f"w_{split}.npy", allow_pickle=False).astype(np.float32, copy=False)
                X, fmt = load_X(d, split)

                idx = topk_by_abs(y, frac, seed=args.seed + (0 if split=="train" else 17 if split=="val" else 31))

                # subset
                y2 = y[idx]
                w2 = w[idx]
                if sparse.issparse(X):
                    X2 = X[idx, :]
                else:
                    X2 = X[idx, :]

                # save
                save_X(out, split, X2, fmt)
                np.save(out / f"y_{split}.npy", y2.astype(np.float32, copy=False))
                np.save(out / f"w_{split}.npy", w2.astype(np.float32, copy=False))

                m2 = subset_meta(d / f"meta_{split}.tsv.gz", idx)
                m2.to_csv(out / f"meta_{split}.tsv.gz", sep="\t", index=False, compression="gzip")

                new_cfg[f"n_{split}"] = int(len(y2))

            (out / "feature_names.txt").write_text(feat_names, encoding="utf-8")
            (out / "config.json").write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            print(f"[DONE] {rel} -> {out_root/tag/rel}  n_train/val/test={new_cfg['n_train']}/{new_cfg['n_val']}/{new_cfg['n_test']}")

    print("[DONE] all subsets created.")


if __name__ == "__main__":
    main()
