#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4A (MAP):
Read one training chunk (chunk_XXXX.train.tsv.gz),
accumulate per (pair_id, feature) weighted sums needed to compute:

- weighted mean of feature in YES and NO, and delta_mean (YES-NO)
- weighted variance of delta (optional)
- correlation between feature and log2fc (weighted Pearson, optional)
- Spearman correlation is expensive; we do Spearman in Step 4B (pair-level) using binned ranks approach (optional).
  Here we compute weighted Pearson as a fast proxy + save raw sums.

Weight w = count_yes + count_no  (only sequences present contribute)

Output: analysis_results/04_Stats/partials/<chunk>.partial.tsv.gz
Each row: pair_id, feature,
  n, w_sum,
  y_sum, y2_sum, x_sum, x2_sum, xy_sum
Where:
  x = feature value
  y = log2fc
"""

from __future__ import annotations
import argparse, csv, gzip, math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--chunk_name", required=True, help="chunk_0001")
    ap.add_argument("--min_weight", type=float, default=1.0, help="drop rows with w < min_weight")
    args = ap.parse_args()

    project = Path(args.project_dir)
    in_dir = project / "analysis_results" / "03_DataWeaver" / "training_chunks"
    out_dir = project / "analysis_results" / "04_Stats" / "partials"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_fp = in_dir / f"{args.chunk_name}.train.tsv.gz"
    if not in_fp.exists():
        raise FileNotFoundError(in_fp)

    out_fp = out_dir / f"{args.chunk_name}.partial.tsv.gz"
    if out_fp.exists() and out_fp.stat().st_size > 0:
        print(f"[SKIP] {out_fp.name} exists")
        return

    # accum[(pair, feature)] = [n, wsum, ysum, y2sum, xsum, x2sum, xysum]
    accum: Dict[Tuple[str,str], list] = defaultdict(lambda: [0,0.0,0.0,0.0,0.0,0.0,0.0])

    with gzip.open(in_fp, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        fields = r.fieldnames or []
        base_cols = {"pair_id","log2fc","count_yes","count_no"}
        feat_cols = [c for c in fields if (c not in base_cols and c not in {"yes_file_id","no_file_id","Seq_ID","total_yes","total_no","rel_yes","rel_no"})]

        # 只保留数值型 feature（跳过 tm_method、碱基字符等）
        # 判定：在前若干行中能转 float
        # 简化：运行时逐值判断，不能转的跳过
        for row in r:
            pid = row["pair_id"]
            y = float(row["log2fc"])
            cy = int(row["count_yes"]); cn = int(row["count_no"])
            w = float(cy + cn)
            if w < args.min_weight:
                continue

            for fc in feat_cols:
                v = row.get(fc, "")
                if v == "" or v is None:
                    continue
                # 过滤非数值
                try:
                    x = float(v)
                except Exception:
                    continue

                key = (pid, fc)
                a = accum[key]
                a[0] += 1
                a[1] += w
                a[2] += w * y
                a[3] += w * y * y
                a[4] += w * x
                a[5] += w * x * x
                a[6] += w * x * y

    with gzip.open(out_fp, "wt", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["pair_id","feature","n","w_sum","y_sum","y2_sum","x_sum","x2_sum","xy_sum"])
        for (pid, feat), a in accum.items():
            w.writerow([pid, feat, a[0], f"{a[1]:.12g}", f"{a[2]:.12g}", f"{a[3]:.12g}", f"{a[4]:.12g}", f"{a[5]:.12g}", f"{a[6]:.12g}"])

    print(f"[DONE] {args.chunk_name}: partial rows={len(accum)} -> {out_fp.name}")

if __name__ == "__main__":
    main()
