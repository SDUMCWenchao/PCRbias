#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, shutil
from pathlib import Path

import numpy as np
from scipy import sparse


def load_names(p: Path):
    return [x.strip() for x in (p / "feature_names.tsv").read_text(encoding="utf-8").splitlines() if x.strip()]


def save_variant(dst: Path, src: Path, col_idx, names):
    dst.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        X = sparse.load_npz(src / f"X_{split}.npz").tocsr()
        Xs = X[:, col_idx]
        sparse.save_npz(dst / f"X_{split}.npz", Xs)
        np.save(dst / f"y_{split}.npy", np.load(src / f"y_{split}.npy"))
    (dst / "feature_names.tsv").write_text("\n".join(names) + "\n", encoding="utf-8")
    info = {}
    if (src / "dataset_info.json").exists():
        info = json.loads((src / "dataset_info.json").read_text(encoding="utf-8"))
    info["derived_from"] = str(src)
    info["n_features"] = int(len(names))
    (dst / "dataset_info.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_root", required=True, help=".../external_test/analysis_results/05_ModelInputs_external_topbias_resplit_v1_resplit_v1")
    ap.add_argument("--tags", required=True, help="top1p,top0p5p,top0p1p")
    ap.add_argument("--k_list", default="4,5,6,7,8")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.inputs_root)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]

    built = 0
    for tag in tags:
        tag_dir = root / tag
        if not tag_dir.exists():
            print(f"[WARN] missing tag dir: {tag_dir}")
            continue
        for compare_dir in sorted([p for p in tag_dir.iterdir() if p.is_dir()]):
            src = compare_dir / "kmer_only"
            if not src.exists():
                continue
            names = load_names(src)

            # names may be "k4_head30_..." OR "feat_k4_head30_..." (robust)
            def which_k(nm: str):
                x = nm
                if x.startswith("feat_"):
                    x = x[5:]
                if x.startswith("k") and "_" in x:
                    try:
                        return int(x.split("_", 1)[0][1:])
                    except Exception:
                        return None
                return None

            k_of = [which_k(n) for n in names]

            for k in k_list:
                idx = [i for i, kk in enumerate(k_of) if kk == k]
                if not idx:
                    print(f"[WARN] {tag}/{compare_dir.name}: no columns for k={k}")
                    continue
                dst = compare_dir / f"kmer_only_k{k}"
                if dst.exists():
                    if args.force:
                        shutil.rmtree(dst)
                    else:
                        print(f"[SKIP] exists: {dst}")
                        continue
                save_variant(dst, src, idx, [names[i] for i in idx])
                built += 1
                print(f"[DONE] built {tag}/{compare_dir.name}/kmer_only_k{k}  nfeat={len(idx)}")

    print(f"[DONE] total built datasets = {built}")


if __name__ == "__main__":
    main()
