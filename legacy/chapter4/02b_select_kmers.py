#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2B: Select filtered k-mers from sparse json tables.

Inputs:
  analysis_results/02_Features/kmer_sparse_chunks/*.kmer.tsv.gz

Outputs:
  analysis_results/02_Features/kmervocab/kmer_vocab.tsv
Columns:
  scope(kmer_all|kmer_head|kmer_tail)   k   kmer   df   total_count

Selection strategy (defaults):
- for each (scope, k):
    keep k-mers with df >= min_df
    then take top max_per_k by df (tie-break by total_count)

This keeps vocabulary bounded and avoids explosion.
"""

from __future__ import annotations
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

def read_tsv_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            yield r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--feature_dir", default=None, help="analysis_results/02_Features (default under project)")
    ap.add_argument("--min_df", type=int, default=10, help="min document frequency per k-mer")
    ap.add_argument("--max_per_k", type=int, default=2000, help="max k-mers kept per (scope,k)")
    args = ap.parse_args()

    project = Path(args.project_dir)
    feature_dir = Path(args.feature_dir) if args.feature_dir else (project / "analysis_results" / "02_Features")
    in_dir = feature_dir / "kmer_sparse_chunks"
    out_dir = feature_dir / "kmervocab"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kmer_vocab.tsv"

    # df and total_count
    df = defaultdict(int)          # key=(scope,k,kmer)
    total = defaultdict(int)

    files = sorted(in_dir.glob("*.kmer.tsv.gz"))
    if not files:
        raise FileNotFoundError(f"No kmer sparse chunks found in: {in_dir}")

    for fp in files:
        for row in read_tsv_gz(fp):
            for scope_key, col in [("kmer_all","kmer_all_json"), ("kmer_head","kmer_head_json"), ("kmer_tail","kmer_tail_json")]:
                js = row.get(col, "") or "{}"
                try:
                    obj = json.loads(js)
                except Exception:
                    obj = {}
                # obj: {"1": {"A": 10,...}, "2": {...}}
                for k_str, cmap in obj.items():
                    k = int(k_str)
                    if not isinstance(cmap, dict):
                        continue
                    for kmer, cnt in cmap.items():
                        key = (scope_key, k, kmer)
                        # df: count once per seq (row), so if cnt>0, increment
                        df[key] += 1
                        total[key] += int(cnt)

    # select
    selected = []
    grouped = defaultdict(list)
    for (scope,k,kmer), d in df.items():
        if d >= args.min_df:
            grouped[(scope,k)].append((kmer, d, total[(scope,k,kmer)]))

    for (scope,k), items in grouped.items():
        # sort by df desc, total_count desc
        items.sort(key=lambda x: (x[1], x[2]), reverse=True)
        items = items[:args.max_per_k]
        for kmer, d, t in items:
            selected.append((scope, k, kmer, d, t))

    # write
    with out_path.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["scope", "k", "kmer", "df", "total_count"])
        for row in selected:
            w.writerow(row)

    print(f"[DONE] vocab={len(selected)} -> {out_path}")
    print(f"[INFO] min_df={args.min_df} max_per_k={args.max_per_k}")

if __name__ == "__main__":
    main()
