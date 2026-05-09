#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

def load_X(ds: Path, split: str):
    npz = ds / f"X_{split}.npz"
    npy = ds / f"X_{split}.npy"
    if npz.exists():
        return sparse.load_npz(npz).tocsr()
    if npy.exists():
        X = np.load(npy, allow_pickle=False)
        return sparse.csr_matrix(X.astype(np.float32, copy=False))
    raise FileNotFoundError(f"missing X_{split}.npz or X_{split}.npy in {ds}")

def read_feat_names(ds: Path):
    fn = ds / "feature_names.txt"
    if not fn.exists():
        raise FileNotFoundError(f"missing feature_names.txt in {ds}")
    return [x.strip() for x in fn.read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--tags", default="top1p,top0p5p,top0p1p")
    ap.add_argument("--out_name", default="real_kmer_only_all")
    args = ap.parse_args()

    project = Path(args.project_dir)
    root = project / args.inputs_root
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    made = 0
    for tag in tags:
        tag_dir = root / tag
        if not tag_dir.exists():
            print(f"[WARN] missing tag dir: {tag_dir}")
            continue

        for sp_dir in sorted(tag_dir.glob("*")):
            if not sp_dir.is_dir():
                continue
            sp = sp_dir.name
            for lc_dir in sorted(sp_dir.glob("*")):
                if not lc_dir.is_dir():
                    continue
                lc = lc_dir.name

                k_dirs = []
                for k in range(1, 9):
                    ds = lc_dir / f"kmer_only_k{k}"
                    if ds.exists() and (ds / "y_train.npy").exists():
                        k_dirs.append(ds)

                if len(k_dirs) == 0:
                    print(f"[WARN] {tag}/{sp}/{lc}: no kmer_only_k1..k8 datasets found -> skip")
                    continue

                out_ds = lc_dir / args.out_name
                out_ds.mkdir(parents=True, exist_ok=True)

                # copy y/w/meta from first available k
                ref = k_dirs[0]
                for split in ["train", "val", "test"]:
                    for fn in [f"y_{split}.npy", f"w_{split}.npy", f"meta_{split}.tsv.gz"]:
                        src = ref / fn
                        if not src.exists():
                            raise FileNotFoundError(f"missing {src}")
                        (out_ds / fn).write_bytes(src.read_bytes())

                # build X by hstack
                feat_all = []
                for split in ["train", "val", "test"]:
                    X_parts = []
                    for ds in k_dirs:
                        X_parts.append(load_X(ds, split))
                    X = sparse.hstack(X_parts, format="csr")
                    sparse.save_npz(out_ds / f"X_{split}.npz", X)
                for ds in k_dirs:
                    feat_all.extend(read_feat_names(ds))

                # ensure unique feature names (normally already unique because names start with k1_/k2_/...)
                seen = {}
                feat_unique = []
                for f in feat_all:
                    if f not in seen:
                        seen[f] = 0
                        feat_unique.append(f)
                    else:
                        seen[f] += 1
                        feat_unique.append(f"{f}__dup{seen[f]}")
                (out_ds / "feature_names.txt").write_text("\n".join(feat_unique) + "\n", encoding="utf-8")

                cfg = {
                    "variant": args.out_name,
                    "source_variants": [d.name for d in k_dirs],
                    "note": "hstack of kmer_only_k1..k8 feature matrices (topbias datasets)",
                }
                (out_ds / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

                print(f"[DONE] built {tag}/{sp}/{lc}/{args.out_name}  nfeat={len(feat_unique)}  from={cfg['source_variants']}")
                made += 1

    print(f"[DONE] total built datasets = {made}")

if __name__ == "__main__":
    main()
