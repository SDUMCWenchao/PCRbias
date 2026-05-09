#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meta summary across pairs:
Input: pair_feature_corr.tsv.gz
Output: meta_feature_summary.tsv

Metrics per feature:
- n_pairs
- mean_abs_corr
- median_abs_corr
- frac_pos_corr
- stability_score = mean_abs_corr * (1 - |frac_pos_corr - 0.5|*2 )?  (penalize sign flipping)
Here: stability = mean_abs_corr * (2*abs(frac_pos_corr-0.5))  -> higher when sign consistent
"""

from __future__ import annotations
import argparse, csv, gzip, statistics
from collections import defaultdict
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    args = ap.parse_args()

    project = Path(args.project_dir)
    in_fp = project / "analysis_results" / "04_Stats" / "pair_feature_corr.tsv.gz"
    out_fp = project / "analysis_results" / "04_Stats" / "meta_feature_summary.tsv"

    feats = defaultdict(list)  # feat -> list[corr]
    with gzip.open(in_fp, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            feat = row["feature"]
            c = float(row["pearson_corr"])
            feats[feat].append(c)

    with out_fp.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["feature","n_pairs","mean_abs_corr","median_abs_corr","frac_pos_corr","stability_score"])
        for feat, cs in feats.items():
            n = len(cs)
            abs_cs = [abs(x) for x in cs]
            mean_abs = sum(abs_cs)/n
            med_abs = statistics.median(abs_cs)
            frac_pos = sum(1 for x in cs if x > 0)/n
            # sign consistency factor: 0..1, 1 when all same sign
            sign_cons = 2*abs(frac_pos - 0.5)
            stability = mean_abs * sign_cons
            w.writerow([feat, n, f"{mean_abs:.6g}", f"{med_abs:.6g}", f"{frac_pos:.6g}", f"{stability:.6g}"])

    print(f"[DONE] {out_fp}")

if __name__ == "__main__":
    main()
