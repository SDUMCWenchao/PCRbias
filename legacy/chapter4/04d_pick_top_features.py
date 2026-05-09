#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--top_n", type=int, default=200)
    ap.add_argument("--min_pairs", type=int, default=1)
    ap.add_argument("--by", choices=["stability_score","mean_abs_corr","median_abs_corr"], default="stability_score")
    args = ap.parse_args()

    project = Path(args.project_dir)
    in_fp = project / "analysis_results" / "04_Stats" / "meta_feature_summary.tsv"
    out_fp = project / "analysis_results" / "04_Stats" / f"top_features_{args.top_n}.txt"

    rows = []
    with in_fp.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            n = int(row["n_pairs"])
            if n < args.min_pairs:
                continue
            rows.append(row)

    rows.sort(key=lambda x: float(x[args.by]), reverse=True)
    picked = rows[:args.top_n]

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with out_fp.open("w", encoding="utf-8") as fo:
        for row in picked:
            fo.write(row["feature"] + "\n")

    print(f"[DONE] picked {len(picked)} features -> {out_fp}")

if __name__ == "__main__":
    main()
