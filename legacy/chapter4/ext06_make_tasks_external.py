#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

NEEDED = ["X_train.npz", "X_val.npz", "X_test.npz", "y_train.npy", "y_val.npy", "y_test.npy", "feature_names.tsv"]

def v2dir(v: str) -> str:
    v = v.strip()
    if v in ("no_kmer", "kmer_only", "all"):
        return v
    if v.startswith("k") and v[1:].isdigit():
        return f"kmer_only_k{int(v[1:])}"
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--tags", required=True)        # top1p,top0p5p,top0p1p
    ap.add_argument("--variants", required=True)    # no_kmer,kmer_only,all,k4..k8
    ap.add_argument("--model", required=True, choices=["rf", "xgb", "seqcnn"])
    ap.add_argument("--out_tsv", required=True)     # tasks file (NO header)
    args = ap.parse_args()

    inputs_root = Path(args.inputs_root)
    out_root = Path(args.out_root)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = []
    for tag in tags:
        tdir = inputs_root / tag
        if not tdir.exists():
            continue
        for comp in sorted([p for p in tdir.iterdir() if p.is_dir()]):
            for v in variants:
                vdir = v2dir(v)
                dset = comp / vdir
                if not dset.exists():
                    continue
                ok = True
                for fn in NEEDED:
                    if not (dset / fn).exists():
                        ok = False
                        break
                if not ok:
                    continue
                out_dir = out_root / tag / comp.name / args.model / vdir
                rows.append((str(dset), str(out_dir), tag, comp.name, vdir))

    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tsv, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"[DONE] model={args.model} tasks={len(rows)} -> {args.out_tsv}")

if __name__ == "__main__":
    main()
