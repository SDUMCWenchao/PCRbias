#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, gzip, statistics
from collections import defaultdict
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--in_name", default="pair_feature_shift_top200.tsv.gz")
    ap.add_argument("--out_name", default="meta_feature_shift_top200.tsv")
    args = ap.parse_args()

    project = Path(args.project_dir)
    in_fp  = project / "analysis_results" / "04_Stats" / args.in_name
    out_fp = project / "analysis_results" / "04_Stats" / args.out_name

    deltas = defaultdict(list)
    effects = defaultdict(list)

    with gzip.open(in_fp, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            feat = row["feature"]
            d = row["delta_yes_minus_no"]
            e = row["effect_d"]
            if d != "NA":
                deltas[feat].append(float(d))
            if e != "NA":
                effects[feat].append(float(e))

    with out_fp.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow([
            "feature","n_pairs",
            "median_delta","mean_abs_delta","frac_pos_delta",
            "median_abs_effect_d"
        ])
        for feat, ds in deltas.items():
            n = len(ds)
            frac_pos = sum(1 for x in ds if x > 0) / n
            mean_abs = sum(abs(x) for x in ds) / n
            med = statistics.median(ds)
            med_abs_d = statistics.median([abs(x) for x in effects.get(feat, [])]) if effects.get(feat) else float("nan")
            w.writerow([
                feat, n,
                f"{med:.6g}", f"{mean_abs:.6g}", f"{frac_pos:.6g}",
                f"{med_abs_d:.6g}" if med_abs_d==med_abs_d else "NA"
            ])

    print(f"[DONE] {out_fp} features={len(deltas)}")

if __name__ == "__main__":
    main()
