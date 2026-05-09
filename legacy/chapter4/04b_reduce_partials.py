#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4A (REDUCE):
Merge partial stats files to compute per (pair_id, feature):
- weighted Pearson corr between feature (x) and log2fc (y)
- n_points and total weight
Outputs:
  analysis_results/04_Stats/pair_feature_corr.tsv.gz
"""

from __future__ import annotations
import argparse, csv, gzip, math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

def read_partial(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            yield row

def pearson_from_sums(w, x_sum, x2_sum, y_sum, y2_sum, xy_sum):
    # cov = E[xy]-E[x]E[y]
    if w <= 0:
        return 0.0
    ex = x_sum / w
    ey = y_sum / w
    ex2 = x2_sum / w
    ey2 = y2_sum / w
    exy = xy_sum / w
    vx = ex2 - ex*ex
    vy = ey2 - ey*ey
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = exy - ex*ey
    return cov / math.sqrt(vx*vy)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    args = ap.parse_args()

    project = Path(args.project_dir)
    part_dir = project / "analysis_results" / "04_Stats" / "partials"
    out_dir = project / "analysis_results" / "04_Stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fp = out_dir / "pair_feature_corr.tsv.gz"

    files = sorted(part_dir.glob("chunk_*.partial.tsv.gz"))
    if not files:
        raise FileNotFoundError(f"No partials in {part_dir}")

    # sums[(pair,feat)] = [n, w, ysum, y2sum, xsum, x2sum, xysum]
    sums: Dict[Tuple[str,str], list] = defaultdict(lambda: [0,0.0,0.0,0.0,0.0,0.0,0.0])

    for fp in files:
        for row in read_partial(fp):
            pid = row["pair_id"]; feat = row["feature"]
            key = (pid, feat)
            a = sums[key]
            a[0] += int(row["n"])
            a[1] += float(row["w_sum"])
            a[2] += float(row["y_sum"])
            a[3] += float(row["y2_sum"])
            a[4] += float(row["x_sum"])
            a[5] += float(row["x2_sum"])
            a[6] += float(row["xy_sum"])

    with gzip.open(out_fp, "wt", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["pair_id","feature","n_points","w_sum","pearson_corr"])
        for (pid, feat), a in sums.items():
            n, wsum, ysum, y2sum, xsum, x2sum, xysum = a
            corr = pearson_from_sums(wsum, xsum, x2sum, ysum, y2sum, xysum)
            w.writerow([pid, feat, n, f"{wsum:.12g}", f"{corr:.12g}"])

    print(f"[DONE] wrote {out_fp} rows={len(sums)}")

if __name__ == "__main__":
    main()
