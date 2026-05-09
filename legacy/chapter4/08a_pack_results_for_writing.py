#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pack small, upload-friendly tables for writing the Results section.

It will:
1) copy key plot tables (metrics + SHAP/IG long tables) into a new folder (no overwrite)
2) optionally scan model folders to extract/aggregate y_true/y_pred on test
3) optionally export k-mer feature name lists (if feature_cols files exist)

Output: analysis_results/_results_textpack/<timestamp>/{tables...} + .tar.gz
"""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import time
from typing import Optional, Tuple, List, Dict


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def copy_if_exists(src: Path, dst_dir: Path):
    if src.exists():
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return True
    return False


def read_tsv(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, r in enumerate(reader):
            rows.append(r)
            if max_rows is not None and i + 1 >= max_rows:
                break
    return rows


def write_tsv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def quantiles(vals: List[float], qs=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> Dict[str, float]:
    if not vals:
        return {f"q{int(q*100):02d}": float("nan") for q in qs}
    v = sorted(vals)
    n = len(v)
    out = {}
    for q in qs:
        idx = int(round(q * (n - 1)))
        out[f"q{int(q*100):02d}"] = v[idx]
    return out


def try_load_pred_file(model_dir: Path) -> Optional[Path]:
    # Try common prediction filenames
    candidates = [
        model_dir / "pred_test.tsv.gz",
        model_dir / "pred_test.tsv",
        model_dir / "predictions_test.tsv.gz",
        model_dir / "predictions_test.tsv",
        model_dir / "test_pred.tsv.gz",
        model_dir / "test_pred.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def summarize_preds(pred_path: Path) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    """
    Expect columns containing y_true and y_pred (names may vary).
    We'll detect from typical names.
    """
    rows = read_tsv(pred_path)
    if not rows:
        return 0, {}, {}
    cols = rows[0].keys()
    y_true_key = None
    y_pred_key = None

    for k in ["y_true", "y", "log2fc_true", "true"]:
        if k in cols:
            y_true_key = k
            break
    for k in ["y_pred", "pred", "log2fc_pred", "prediction"]:
        if k in cols:
            y_pred_key = k
            break
    if y_true_key is None or y_pred_key is None:
        # cannot parse
        return len(rows), {}, {}

    y_true = []
    y_pred = []
    resid = []
    for r in rows:
        try:
            yt = float(r[y_true_key])
            yp = float(r[y_pred_key])
        except Exception:
            continue
        y_true.append(yt)
        y_pred.append(yp)
        resid.append(yp - yt)

    return len(y_true), quantiles(y_true), quantiles(resid)


def scan_feature_cols(dataset_dir: Path) -> Optional[Path]:
    # Try common filenames storing feature names
    candidates = [
        dataset_dir / "feature_cols.txt",
        dataset_dir / "feature_cols.tsv",
        dataset_dir / "columns.txt",
        dataset_dir / "cols.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_feature_cols(path: Path) -> List[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    cols = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # allow tsv with header
            if "\t" in s and len(cols) == 0 and s.lower().startswith("feature"):
                continue
            cols.append(s.split("\t")[0])
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=".", help="project root (default: .)")
    ap.add_argument("--plot_tables_dir", default="analysis_results/07_PlotTables_v3_topbias",
                    help="folder containing plot tables")
    ap.add_argument("--models_root", default="analysis_results/06_Models_v3_topbias",
                    help="root folder of trained models")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3_topbias",
                    help="root folder of datasets")
    ap.add_argument("--out_root", default="analysis_results/_results_textpack",
                    help="where to write the pack (no overwrite)")
    ap.add_argument("--export_preds", action="store_true", help="also export prediction summaries if pred files exist")
    ap.add_argument("--export_feature_lists", action="store_true", help="also export feature list summaries if feature_cols exist")
    args = ap.parse_args()

    project = Path(args.project_dir).resolve()
    plot_dir = (project / args.plot_tables_dir).resolve()
    models_root = (project / args.models_root).resolve()
    inputs_root = (project / args.inputs_root).resolve()
    out_dir = (project / args.out_root / now_tag()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) copy key plot tables
    key_files = [
        "01_master_metrics_ci.tsv",
        "04_shap_global_full_long.tsv",
        "05_shap_region_full_long.tsv",
        "06_ig_region_long.tsv",
        "07_ig_pos_from_end_long.tsv",
        "08_feature_importance_long.tsv",
    ]
    copied = []
    missing = []
    for fn in key_files:
        src = plot_dir / fn
        if copy_if_exists(src, out_dir):
            copied.append(fn)
        else:
            missing.append(fn)

    # 2) write a small README
    readme = out_dir / "README.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(f"Packed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"project_dir: {project}\n")
        f.write(f"plot_tables_dir: {plot_dir}\n")
        f.write(f"models_root: {models_root}\n")
        f.write(f"inputs_root: {inputs_root}\n")
        f.write(f"copied_plot_tables: {copied}\n")
        f.write(f"missing_plot_tables: {missing}\n")

    # 3) optionally summarize predictions
    if args.export_preds:
        pred_rows = []
        # walk model dirs: <tag>/<species>/<locus>/<model>/<variant>/
        if models_root.exists():
            for pred_path in models_root.rglob("pred_test.tsv.gz"):
                model_dir = pred_path.parent
                n, yq, rq = summarize_preds(pred_path)
                if not yq or not rq:
                    continue
                rel = model_dir.relative_to(models_root)
                parts = rel.parts
                if len(parts) < 5:
                    continue
                tag, species, locus, model, variant = parts[0], parts[1], parts[2], parts[3], parts[4]
                row = {"tag": tag, "species": species, "locus": locus, "model": model, "variant": variant, "n": n}
                row.update({f"y_{k}": f"{v:.6g}" for k, v in yq.items()})
                row.update({f"resid_{k}": f"{v:.6g}" for k, v in rq.items()})
                pred_rows.append(row)

        if pred_rows:
            fields = ["tag","species","locus","model","variant","n"] + \
                     [f"y_q{q:02d}" for q in [0,25,50,75,90,95,99,100]] + \
                     [f"resid_q{q:02d}" for q in [0,25,50,75,90,95,99,100]]
            # adjust fieldnames to match the quantiles() function keys
            fields = ["tag","species","locus","model","variant","n"] + \
                     [f"y_q{q:02d}" for q in [0,25,50,75,90,95,99,100]] + \
                     [f"resid_q{q:02d}" for q in [0,25,50,75,90,95,99,100]]
            # Actually produced keys are y_q00 etc; keep consistent:
            fields = ["tag","species","locus","model","variant","n"] + \
                     [f"y_q{q:02d}" for q in [0,25,50,75,90,95,99,100]] + \
                     [f"resid_q{q:02d}" for q in [0,25,50,75,90,95,99,100]]

            # remap from dict keys q00 -> q00
            # write dynamically:
            all_fields = set()
            for r in pred_rows:
                all_fields |= set(r.keys())
            fieldnames = [c for c in ["tag","species","locus","model","variant","n"] if c in all_fields] + \
                         sorted([c for c in all_fields if c not in {"tag","species","locus","model","variant","n"}])
            write_tsv(out_dir / "pred_summaries_test.tsv", pred_rows, fieldnames)

    # 4) optionally export feature lists summaries (counts + k-mer breakdown)
    if args.export_feature_lists:
        feat_rows = []
        if inputs_root.exists():
            # dataset dirs: <tag>/<species>/<locus>/<variant>/
            for dataset_dir in inputs_root.glob("*/*/*/*"):
                rel = dataset_dir.relative_to(inputs_root)
                parts = rel.parts
                if len(parts) != 4:
                    continue
                tag, species, locus, variant = parts
                fpath = scan_feature_cols(dataset_dir)
                if not fpath:
                    continue
                cols = read_feature_cols(fpath)
                nfeat = len(cols)
                nkmer = sum(1 for c in cols if ("kmer" in c.lower()) or ("__k" in c.lower()) or c.lower().startswith("k"))
                nprimer = sum(1 for c in cols if c.lower().startswith("feat_pr_") or "primer" in c.lower())
                feat_rows.append({
                    "tag": tag, "species": species, "locus": locus, "variant": variant,
                    "n_features": nfeat, "n_kmer_like": nkmer, "n_primer_like": nprimer,
                    "feature_cols_file": str(fpath.name),
                })
                # also dump full list (gz) per dataset (still small generally)
                out_list = out_dir / "feature_lists" / tag / species / locus
                out_list.mkdir(parents=True, exist_ok=True)
                gzpath = out_list / f"{variant}.features.txt.gz"
                with gzip.open(gzpath, "wt", encoding="utf-8") as g:
                    for c in cols:
                        g.write(c + "\n")

        if feat_rows:
            fieldnames = ["tag","species","locus","variant","n_features","n_kmer_like","n_primer_like","feature_cols_file"]
            write_tsv(out_dir / "feature_list_manifest.tsv", feat_rows, fieldnames)

    # 5) tar.gz the pack
    tar_path = out_dir.with_suffix(".tar.gz")
    # create tar using system tar if available
    os.system(f"tar -czf {tar_path} -C {out_dir.parent} {out_dir.name}")

    print(f"[DONE] pack dir: {out_dir}")
    print(f"[DONE] tarball : {tar_path}")
    print(f"[INFO] copied plot tables: {len(copied)}  missing: {len(missing)}")
    if args.export_preds:
        print("[INFO] exported prediction summaries (if any).")
    if args.export_feature_lists:
        print("[INFO] exported feature list summaries (if any).")


if __name__ == "__main__":
    main()
