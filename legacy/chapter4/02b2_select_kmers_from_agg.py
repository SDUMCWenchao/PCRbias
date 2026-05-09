#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select k-mer vocab from aggregated chunk files:
input:  kmer_agg_chunks/*.kmer_agg.tsv.gz  (scope,k,kmer,df,total_count)
output: kmervocab/kmer_vocab.tsv

Selection:
- sum df and total_count across chunks
- for each (scope,k): keep df>=min_df then take top max_per_k by (df, total_count)
"""

from __future__ import annotations
import argparse, csv, gzip
from collections import defaultdict
from pathlib import Path

def read_agg(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            yield row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--min_df", type=int, default=10)
    ap.add_argument("--max_per_k", type=int, default=2000)
    args = ap.parse_args()

    project = Path(args.project_dir)
    feat = project / "analysis_results" / "02_Features"
    in_dir = feat / "kmer_agg_chunks"
    out_dir = feat / "kmervocab"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kmer_vocab.tsv"

    files = sorted(in_dir.glob("*.kmer_agg.tsv.gz"))
    if not files:
        raise FileNotFoundError(f"No agg files in {in_dir}. Run 02b1_submit_agg_kmers.sh first.")

    df = defaultdict(int)     # key=(scope,k,kmer)
    total = defaultdict(int)

    for fp in files:
        for row in read_agg(fp):
            scope = row["scope"]
            k = int(row["k"])
            kmer = row["kmer"]
            d = int(row["df"])
            t = int(row["total_count"])
            key = (scope,k,kmer)
            df[key] += d
            total[key] += t

    grouped = defaultdict(list)
    for (scope,k,kmer), d in df.items():
        if d >= args.min_df:
            grouped[(scope,k)].append((kmer, d, total[(scope,k,kmer)]))

    selected = []
    for (scope,k), items in grouped.items():
        items.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for kmer, d, t in items[:args.max_per_k]:
            selected.append((scope, k, kmer, d, t))

    with out_path.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["scope","k","kmer","df","total_count"])
        w.writerows(selected)

    print(f"[DONE] vocab={len(selected)} -> {out_path}")

if __name__ == "__main__":
    main()
