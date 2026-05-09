#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make union kmer vocab (per k) from kmervocab/kmer_vocab.tsv (scope,k,kmer,df,total_count).
Union across scopes (all/head/tail), then keep top per k by summed df (and total_count tiebreak).

Output:
analysis_results/02_Features/kmervocab/kmer_union.tsv  (k, kmer, df_sum, total_sum)
"""

from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--in_vocab", default=None)
    ap.add_argument("--min_k", type=int, default=6)
    ap.add_argument("--max_k", type=int, default=8)
    ap.add_argument("--max_per_k", type=int, default=4000, help="union after scopes; avoid explosion")
    args = ap.parse_args()

    project = Path(args.project_dir)
    feat_dir = project / "analysis_results" / "02_Features"
    in_fp = Path(args.in_vocab) if args.in_vocab else (feat_dir / "kmervocab" / "kmer_vocab.tsv")
    out_fp = feat_dir / "kmervocab" / "kmer_union.tsv"
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    acc = defaultdict(lambda: [0,0])  # (k,kmer)->[df_sum,total_sum]
    with in_fp.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            k = int(row["k"])
            if k < args.min_k or k > args.max_k:
                continue
            kmer = row["kmer"]
            df = int(float(row["df"]))
            tc = int(float(row["total_count"]))
            a = acc[(k,kmer)]
            a[0] += df
            a[1] += tc

    # keep top per k
    byk = defaultdict(list)
    for (k,kmer),(df,tc) in acc.items():
        byk[k].append((kmer, df, tc))

    with out_fp.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["k","kmer","df_sum","total_sum"])
        kept = 0
        for k in sorted(byk):
            items = byk[k]
            items.sort(key=lambda x: (x[1], x[2]), reverse=True)
            items = items[:args.max_per_k]
            for kmer, df, tc in items:
                w.writerow([k, kmer, df, tc])
                kept += 1

    print(f"[DONE] union vocab -> {out_fp} (kept={kept})")

if __name__ == "__main__":
    main()
