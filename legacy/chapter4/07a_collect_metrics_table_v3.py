#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def safe_get(d, *keys, default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def parse_path(p: Path):
    # expected:
    # .../06_Models_v3_topbias/<tag>/<species>/<locus>/<model>/<variant>/metrics.json
    parts = p.parts
    tag = species = locus = model = variant = None
    try:
        i = parts.index("06_Models_v3_topbias")
        tag, species, locus, model, variant = parts[i+1:i+6]
    except Exception:
        # fallback: search for /topXp/ pattern
        for j in range(len(parts)):
            if parts[j].startswith("top") and j+4 < len(parts):
                tag, species, locus, model, variant = parts[j:j+5]
                break
    return tag, species, locus, model, variant

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--models_root", default="analysis_results/06_Models_v3_topbias")
    ap.add_argument("--out_tsv", default="analysis_results/06_Models_v3_topbias/metrics_detail.tsv")
    ap.add_argument("--out_summary_tsv", default="analysis_results/06_Models_v3_topbias/metrics_summary.tsv")
    args = ap.parse_args()

    project = Path(args.project_dir)
    root = project / args.models_root
    out_tsv = project / args.out_tsv
    out_sum = project / args.out_summary_tsv
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for mj in root.rglob("metrics.json"):
        try:
            obj = json.loads(mj.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] skip unreadable: {mj} ({e})")
            continue

        tag, sp, lc, model, variant = parse_path(mj)
        cfg = obj.get("config", {})

        row = {
            "tag": tag,
            "species": sp,
            "locus": lc,
            "model": model,
            "variant": variant,
            "path": str(mj.parent),
            # val
            "val_r2": safe_get(obj, "val", "r2"),
            "val_rmse": safe_get(obj, "val", "rmse"),
            "val_mae": safe_get(obj, "val", "mae"),
            "val_spearman": safe_get(obj, "val", "spearman"),
            "val_sign_acc": safe_get(obj, "val", "sign_acc"),
            # test
            "test_r2": safe_get(obj, "test", "r2"),
            "test_rmse": safe_get(obj, "test", "rmse"),
            "test_mae": safe_get(obj, "test", "mae"),
            "test_spearman": safe_get(obj, "test", "spearman"),
            "test_sign_acc": safe_get(obj, "test", "sign_acc"),
            # pair
            "val_pair_spear_mean": safe_get(obj, "val_pair", "pair_spearman_mean"),
            "val_pair_n": safe_get(obj, "val_pair", "pair_spearman_n"),
            "test_pair_spear_mean": safe_get(obj, "test_pair", "pair_spearman_mean"),
            "test_pair_n": safe_get(obj, "test_pair", "pair_spearman_n"),
            # rf extras (if any)
            "oob_r2": obj.get("oob_r2", np.nan),
            "n_estimators": cfg.get("n_estimators", np.nan),
            "min_samples_leaf": cfg.get("min_samples_leaf", np.nan),
            "max_features": cfg.get("max_features", np.nan),
            "bootstrap": cfg.get("bootstrap", np.nan),
            "max_samples": cfg.get("max_samples", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"[ERROR] no metrics.json found under {root}")

    # nicer sort
    df["tag"] = df["tag"].astype(str)
    df = df.sort_values(["tag","species","locus","model","variant"], kind="stable").reset_index(drop=True)

    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[DONE] detail -> {out_tsv}  rows={len(df)}")

    # summary: mean across species×locus (per tag/variant/model)
    g = df.groupby(["tag","model","variant"], dropna=False)
    sumdf = g.agg(
        n=("test_r2","count"),
        test_r2_mean=("test_r2","mean"),
        test_r2_median=("test_r2","median"),
        test_spear_mean=("test_spearman","mean"),
        test_sign_mean=("test_sign_acc","mean"),
        test_rmse_mean=("test_rmse","mean"),
    ).reset_index().sort_values(["tag","model","variant"])
    sumdf.to_csv(out_sum, sep="\t", index=False)
    print(f"[DONE] summary -> {out_sum}  rows={len(sumdf)}")

if __name__ == "__main__":
    main()
