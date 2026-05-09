#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

VARIANTS = [
    "no_kmer", "kmer_only", "all",
    "kmer_only_k4", "kmer_only_k5", "kmer_only_k6", "kmer_only_k7", "kmer_only_k8",
]

def exists_any(dirpath: Path, names):
    for n in names:
        if (dirpath / n).exists():
            return dirpath / n
    return None

def detect_rf_model(model_dir: Path):
    return exists_any(model_dir, ["model.joblib", "model.pkl", "rf.joblib"])

def detect_xgb_model(model_dir: Path):
    return exists_any(model_dir, ["model.json", "model.ubj", "model.bin", "xgb.json", "xgb.ubj"])

def detect_seqcnn_model(model_dir: Path):
    p = model_dir / "model.pt"
    return p if p.exists() else None

def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", required=True, help=".../external_test")
    ap.add_argument("--inputs_root", required=True, help=".../05_ModelInputs_external_topbias_resplit_v1")
    ap.add_argument("--models_root", required=True, help=".../06_Models_external_topbias_v2_resplit_v1")
    ap.add_argument("--out_dir", required=True, help="where to write tasks_missing_*.tsv and summary")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    args = ap.parse_args()

    project_dir = Path(args.project_dir)
    inputs_root = Path(args.inputs_root)
    models_root = Path(args.models_root)
    out_dir = Path(args.out_dir)
    split = args.split

    rf_rows = []
    xgb_rows = []
    ig_rows = []

    summary = []
    tags = sorted([p for p in models_root.glob("top*") if p.is_dir()])

    for tag_dir in tags:
        tag = tag_dir.name
        compares = sorted([p for p in tag_dir.iterdir() if p.is_dir()])
        for comp_dir in compares:
            compare_id = comp_dir.name

            # model groups
            rf_root = comp_dir / "rf"
            xgb_root = comp_dir / "xgb"
            cnn_root = comp_dir / "seqcnn"

            for variant in VARIANTS:
                dataset_dir = inputs_root / tag / compare_id / variant

                # -------- RF SHAP --------
                if rf_root.is_dir():
                    mdir = rf_root / variant
                    if mdir.is_dir() and dataset_dir.exists():
                        need = mdir / "shap_tables" / f"rf_shap_{split}.tsv"
                        have_model = detect_rf_model(mdir) is not None
                        if have_model and (not need.exists()):
                            rf_rows.append((tag, compare_id, variant, str(dataset_dir), str(mdir)))

                # -------- XGB SHAP --------
                if xgb_root.is_dir():
                    mdir = xgb_root / variant
                    if mdir.is_dir() and dataset_dir.exists():
                        need = mdir / "shap_tables" / f"xgb_shap_{split}.tsv"
                        have_model = detect_xgb_model(mdir) is not None
                        if have_model and (not need.exists()):
                            xgb_rows.append((tag, compare_id, variant, str(dataset_dir), str(mdir)))

                # -------- seqCNN IG --------
                if cnn_root.is_dir():
                    mdir = cnn_root / variant
                    if mdir.is_dir() and dataset_dir.exists():
                        need1 = mdir / "attr_tables" / f"ig_region_{split}.tsv"
                        need2 = mdir / "attr_tables" / f"ig_base_{split}.tsv"
                        have_model = detect_seqcnn_model(mdir) is not None
                        if have_model and (not (need1.exists() and need2.exists())):
                            ig_rows.append((tag, compare_id, variant, str(dataset_dir), str(mdir)))

    out_dir.mkdir(parents=True, exist_ok=True)

    write_tsv(out_dir / "tasks_missing_rf.tsv",
              ["tag","compare_id","variant","dataset_dir","model_dir"], rf_rows)
    write_tsv(out_dir / "tasks_missing_xgb.tsv",
              ["tag","compare_id","variant","dataset_dir","model_dir"], xgb_rows)
    write_tsv(out_dir / "tasks_missing_ig.tsv",
              ["tag","compare_id","variant","dataset_dir","model_dir"], ig_rows)

    # summary
    summary.append(("rf_missing", len(rf_rows)))
    summary.append(("xgb_missing", len(xgb_rows)))
    summary.append(("ig_missing", len(ig_rows)))

    with open(out_dir / "missing_summary.tsv", "w") as f:
        f.write("item\tcount\n")
        for k,v in summary:
            f.write(f"{k}\t{v}\n")

    print("[DONE] wrote:")
    print("  ", out_dir / "tasks_missing_rf.tsv", "n=", len(rf_rows))
    print("  ", out_dir / "tasks_missing_xgb.tsv", "n=", len(xgb_rows))
    print("  ", out_dir / "tasks_missing_ig.tsv", "n=", len(ig_rows))
    print("  ", out_dir / "missing_summary.tsv")

if __name__ == "__main__":
    main()
