#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, gzip, shutil
from pathlib import Path
import numpy as np
import pandas as pd

def load_y(train_dir: Path):
    # 支持 y_train.npy 或 y_train.tsv(.gz)
    p = train_dir / "y_train.npy"
    if p.exists():
        y = np.load(p)
        return y, "npy"
    for fn in ["y_train.tsv.gz","y_train.tsv"]:
        p = train_dir / fn
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            # 默认取第一列
            y = df.iloc[:,0].to_numpy()
            return y, "tsv"
    raise FileNotFoundError(f"[BAD] cannot find y_train in {train_dir}")

def slice_and_save_file(fin: Path, fout: Path, idx, kind):
    fout.parent.mkdir(parents=True, exist_ok=True)
    if fin.suffix == ".npy":
        arr = np.load(fin)
        np.save(fout, arr[idx])
        return
    if fin.suffix == ".npz":
        # sparse matrix
        from scipy import sparse
        X = sparse.load_npz(fin)
        sparse.save_npz(fout, X[idx])
        return
    # tsv / tsv.gz
    if fin.name.endswith(".tsv.gz"):
        df = pd.read_csv(fin, sep="\t")
        df.iloc[idx].to_csv(fout, sep="\t", index=False, compression="gzip")
        return
    if fin.name.endswith(".tsv"):
        df = pd.read_csv(fin, sep="\t")
        df.iloc[idx].to_csv(fout, sep="\t", index=False)
        return
    # 其他文件：直接复制
    shutil.copy2(fin, fout)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_root", required=True, help="e.g. external_test/analysis_results/05_ModelInputs_external_topbias_resplit_v1_resplit_v1")
    ap.add_argument("--out_root", required=True, help="new dir, won't overwrite old")
    ap.add_argument("--tags", default="top1p,top0p5p,top0p1p")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train_frac", type=float, default=0.80)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--min_val", type=int, default=1)
    ap.add_argument("--min_test", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    in_root = Path(args.inputs_root)
    out_root = Path(args.out_root)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    if out_root.exists():
        if not args.force:
            raise RuntimeError(f"[BAD] out_root exists: {out_root} (use --force)")
        shutil.rmtree(out_root)

    rng = np.random.default_rng(args.seed)
    # 找所有 dataset_dir：包含 y_train.npy 或 y_train.tsv(.gz)
    candidates = []
    for tag in tags:
        tag_dir = in_root / tag
        if not tag_dir.exists():
            continue
        for ypath in tag_dir.rglob("y_train.npy"):
            candidates.append(ypath.parent)
        for ypath in tag_dir.rglob("y_train.tsv.gz"):
            candidates.append(ypath.parent)
        for ypath in tag_dir.rglob("y_train.tsv"):
            candidates.append(ypath.parent)

    # 去重
    seen=set()
    dataset_dirs=[]
    for d in candidates:
        if str(d) not in seen:
            seen.add(str(d))
            dataset_dirs.append(d)

    print(f"[INFO] datasets found = {len(dataset_dirs)}")

    bad_small = 0
    for d in sorted(dataset_dirs):
        # d: .../<tag>/<compare>/<variant>
        rel = d.relative_to(in_root)
        out_d = out_root / rel

        y, yfmt = load_y(d)
        n = len(y)
        if n < (1 + args.min_val + args.min_test):
            print(f"[WARN] skip too-small n={n}: {rel}")
            bad_small += 1
            continue

        idx = np.arange(n)
        rng.shuffle(idx)

        n_val = max(args.min_val, int(round(n * args.val_frac)))
        n_test = max(args.min_test, int(round(n * args.test_frac)))
        # 保证至少 1 个 train
        if n - n_val - n_test < 1:
            n_val = args.min_val
            n_test = args.min_test
            if n - n_val - n_test < 1:
                # 再兜底：压缩 val/test
                n_val = 1
                n_test = 1
        n_train = n - n_val - n_test

        tr = idx[:n_train]
        va = idx[n_train:n_train+n_val]
        te = idx[n_train+n_val:n_train+n_val+n_test]

        out_d.mkdir(parents=True, exist_ok=True)
        # 保存 split index 便于复现
        pd.DataFrame({"split":["train"]*len(tr)+["val"]*len(va)+["test"]*len(te),
                      "i": np.concatenate([tr,va,te])}).to_csv(out_d/"split_index.tsv", sep="\t", index=False)

        # 对所有 *_train.* 文件切分
        for fin in d.iterdir():
            if fin.is_dir():
                continue
            name = fin.name
            if "_val." in name or "_test." in name:
                continue
            if "_train." in name:
                base = name.replace("_train.", "_{}.")
                slice_and_save_file(fin, out_d / base.format("train"), tr, "train")
                slice_and_save_file(fin, out_d / base.format("val"), va, "val")
                slice_and_save_file(fin, out_d / base.format("test"), te, "test")
            else:
                shutil.copy2(fin, out_d / name)

        print(f"[DONE] {rel}  n={n} -> train/val/test={len(tr)}/{len(va)}/{len(te)}")

    print(f"[DONE] out_root -> {out_root}  (skipped_small={bad_small})")

if __name__ == "__main__":
    main()
