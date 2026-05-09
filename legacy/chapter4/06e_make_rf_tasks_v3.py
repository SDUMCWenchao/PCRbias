#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

VARIANT_ORDER = [
    "no_kmer",
    "no_kmer_noprimer",
    "kmer_only_all",
    "kmer_only_k1","kmer_only_k2","kmer_only_k3","kmer_only_k4",
    "kmer_only_k5","kmer_only_k6","kmer_only_k7","kmer_only_k8",
    "all",
    "all_noprimer",
]

def has_dataset(d: Path) -> bool:
    return (d / "y_train.npy").exists() and ((d / "X_train.npz").exists() or (d / "X_train.npy").exists())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3")
    ap.add_argument("--models_root", default="analysis_results/06_Models_v3")
    ap.add_argument("--out_tasks", default="analysis_results/06_Models_v3/rf_tasks.tsv")
    ap.add_argument("--only_core4", action="store_true",
                    help="only run 4 key variants: no_kmer, no_kmer_noprimer, kmer_only_all, all_noprimer")
    args = ap.parse_args()

    project = Path(args.project_dir)
    inputs_root = project / args.inputs_root
    models_root = project / args.models_root
    out_tasks = project / args.out_tasks
    out_tasks.parent.mkdir(parents=True, exist_ok=True)

    variants = VARIANT_ORDER
    if args.only_core4:
        variants = ["no_kmer", "no_kmer_noprimer", "kmer_only_all", "all_noprimer"]

    tasks = []
    for sp_dir in sorted(inputs_root.glob("*")):
        if not sp_dir.is_dir():
            continue
        sp = sp_dir.name
        for lc_dir in sorted(sp_dir.glob("*")):
            if not lc_dir.is_dir():
                continue
            lc = lc_dir.name
            for v in variants:
                ds = lc_dir / v
                if not ds.exists():
                    continue
                if not has_dataset(ds):
                    continue
                out = models_root / sp / lc / "rf" / v
                tasks.append((sp, lc, v, str(ds), str(out)))

    with out_tasks.open("w", encoding="utf-8") as w:
        # 无 header，便于 array 用行号读取
        for sp, lc, v, ds, out in tasks:
            w.write(f"{sp}\t{lc}\t{v}\t{ds}\t{out}\n")

    print(f"[DONE] tasks = {len(tasks)} -> {out_tasks}")
    if len(tasks) == 0:
        print("[WARN] no tasks found. Check inputs_root path.")

if __name__ == "__main__":
    main()
