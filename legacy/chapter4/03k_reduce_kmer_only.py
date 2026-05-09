#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reduce kmer-only chunk sums:
Input:
- analysis_results/03_DataWeaver/kmer_chunk_sums/chunk_*.kmer_sums.tsv.gz
- analysis_results/03_DataWeaver/sample_totals.tsv
- analysis_results/03_DataWeaver/pairs.tsv

Outputs:
- analysis_results/03_DataWeaver/kmer_only/sample_kmer_norm.tsv.gz
- analysis_results/03_DataWeaver/kmer_only/pair_kmer_log2fc.tsv.gz
- analysis_results/03_DataWeaver/kmer_only/top_hits_per_pair.tsv
"""

from __future__ import annotations
import argparse, csv, gzip, math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

def read_totals(path: Path) -> Dict[str,int]:
    d={}
    with path.open("r", encoding="utf-8") as f:
        r=csv.DictReader(f, delimiter="\t")
        for row in r:
            d[row["file_id"]] = int(row["total_reads"])
    return d

def read_pairs(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        r=csv.DictReader(f, delimiter="\t")
        return [row for row in r]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--topn", type=int, default=50)
    args = ap.parse_args()

    project = Path(args.project_dir)
    weaver = project / "analysis_results" / "03_DataWeaver"
    in_dir = weaver / "kmer_chunk_sums"
    totals_tsv = weaver / "sample_totals.tsv"
    pairs_tsv = weaver / "pairs.tsv"

    out_dir = weaver / "kmer_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("chunk_*.kmer_sums.tsv.gz"))
    if not files:
        raise FileNotFoundError(f"No chunk sums in {in_dir}. Run 03k_submit_kmer_only.sh first.")

    totals = read_totals(totals_tsv)
    pairs = read_pairs(pairs_tsv)

    # aggregate sums: (file_id, scope, k, kmer) -> sum_count
    sums = defaultdict(int)

    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                fid = row["file_id"]
                scope = row["scope"]
                k = int(row["k"])
                kmer = row["kmer"]
                v = int(row["sum_count"])
                sums[(fid, scope, k, kmer)] += v

    # compute norms (per read)
    norm = {}
    for (fid, scope, k, kmer), v in sums.items():
        t = totals.get(fid, 0)
        if t <= 0:
            continue
        norm[(fid, scope, k, kmer)] = v / t

    sample_out = out_dir / "sample_kmer_norm.tsv.gz"
    with gzip.open(sample_out, "wt", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["file_id","scope","k","kmer","sum_count","total_reads","norm_per_read"])
        for (fid, scope, k, kmer), v in sums.items():
            t = totals.get(fid, 0)
            if t <= 0:
                continue
            w.writerow([fid, scope, k, kmer, v, t, f"{v/t:.12g}"])

    pair_out = out_dir / "pair_kmer_log2fc.tsv.gz"
    top_out = out_dir / "top_hits_per_pair.tsv"
    with gzip.open(pair_out, "wt", encoding="utf-8") as fo, top_out.open("w", encoding="utf-8", newline="") as ftop:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["pair_id","yes_file_id","no_file_id","scope","k","kmer","yes_norm","no_norm","log2fc"])

        wt = csv.writer(ftop, delimiter="\t")
        wt.writerow(["pair_id","direction","rank","scope","k","kmer","log2fc","yes_norm","no_norm"])

        # build the union of kmers present in any sample (reduces loops)
        all_feats = set((scope,k,kmer) for (_,scope,k,kmer) in sums.keys())

        for p in pairs:
            pid = p["pair_id"]
            yid = p["yes_file_id"]
            nid = p["no_file_id"]

            hits_pos = []
            hits_neg = []

            for scope,k,kmer in all_feats:
                y = norm.get((yid, scope, k, kmer), 0.0)
                n = norm.get((nid, scope, k, kmer), 0.0)
                l2 = math.log2((y + args.eps) / (n + args.eps))
                w.writerow([pid, yid, nid, scope, k, kmer, f"{y:.12g}", f"{n:.12g}", f"{l2:.12g}"])

                if l2 > 0:
                    hits_pos.append((l2, scope, k, kmer, y, n))
                elif l2 < 0:
                    hits_neg.append((l2, scope, k, kmer, y, n))

            hits_pos.sort(key=lambda x: x[0], reverse=True)
            hits_neg.sort(key=lambda x: x[0])  # most negative first

            for i, (l2, scope, k, kmer, y, n) in enumerate(hits_pos[:args.topn], start=1):
                wt.writerow([pid, "up_in_yes", i, scope, k, kmer, f"{l2:.12g}", f"{y:.12g}", f"{n:.12g}"])
            for i, (l2, scope, k, kmer, y, n) in enumerate(hits_neg[:args.topn], start=1):
                wt.writerow([pid, "down_in_yes", i, scope, k, kmer, f"{l2:.12g}", f"{y:.12g}", f"{n:.12g}"])

    print("[DONE]")
    print(f"  sample norms: {sample_out}")
    print(f"  pair log2fc : {pair_out}")
    print(f"  top hits    : {top_out}")

if __name__ == "__main__":
    main()
