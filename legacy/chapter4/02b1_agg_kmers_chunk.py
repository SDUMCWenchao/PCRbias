#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build per-chunk aggregated kmer stats from sparse json file:
input:  <chunk>.kmer.tsv.gz   (columns: Seq_ID, kmer_all_json, kmer_head_json, kmer_tail_json)
output: <chunk>.kmer_agg.tsv.gz (columns: scope, k, kmer, df, total_count)

df = number of sequences where kmer appears (cnt>0)
total_count = sum of counts across sequences
"""

from __future__ import annotations
import argparse, csv, gzip, json
from collections import defaultdict
from pathlib import Path

# faster json if available
try:
    import orjson  # type: ignore
    def loads(x: str):
        return orjson.loads(x)
except Exception:
    def loads(x: str):
        return json.loads(x)

def read_tsv_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            yield r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_kmer", required=True, help="input chunk kmer.tsv.gz")
    ap.add_argument("--out_agg", required=True, help="output chunk kmer_agg.tsv.gz")
    args = ap.parse_args()

    in_fp = Path(args.in_kmer)
    out_fp = Path(args.out_agg)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    df = defaultdict(int)     # key=(scope,k,kmer)
    total = defaultdict(int)

    for row in read_tsv_gz(in_fp):
        for scope, col in (("kmer_all", "kmer_all_json"),
                           ("kmer_head","kmer_head_json"),
                           ("kmer_tail","kmer_tail_json")):
            s = row.get(col, "") or "{}"
            try:
                obj = loads(s)
            except Exception:
                obj = {}
            # obj: {"1": {"A": 10,...}, "2": {...}}
            for k_str, cmap in obj.items():
                if not isinstance(cmap, dict):
                    continue
                k = int(k_str)
                for kmer, cnt in cmap.items():
                    cnt = int(cnt)
                    if cnt <= 0:
                        continue
                    key = (scope, k, kmer)
                    df[key] += 1
                    total[key] += cnt

    with gzip.open(out_fp, "wt", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["scope", "k", "kmer", "df", "total_count"])
        for (scope,k,kmer), d in df.items():
            w.writerow([scope, k, kmer, d, total[(scope,k,kmer)]])

    print(f"[DONE] {in_fp.name} -> {out_fp.name} rows={len(df)}")

if __name__ == "__main__":
    main()
