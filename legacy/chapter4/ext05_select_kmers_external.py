#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from collections import Counter, defaultdict
from pathlib import Path

def fasta_iter(path: Path):
    name=None; seq=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name=line[1:].split()[0]
                seq=[]
            else:
                seq.append(line.upper())
        if name is not None:
            yield name, "".join(seq)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--min_k", type=int, default=1)
    ap.add_argument("--max_k", type=int, default=8)
    ap.add_argument("--min_support", type=int, default=2, help="document frequency >= this")
    ap.add_argument("--top_per_k", type=int, default=200)
    ap.add_argument("--kmer_cap_total", type=int, default=1600)
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    fa = Path(args.fasta)
    df = {k: Counter() for k in range(args.min_k, args.max_k+1)}

    n=0
    for _, s in fasta_iter(fa):
        n += 1
        L = len(s)
        for k in range(args.min_k, args.max_k+1):
            if L < k: 
                continue
            seen = set()
            for i in range(L-k+1):
                km = s[i:i+k]
                if "N" in km: 
                    continue
                seen.add(km)
            for km in seen:
                df[k][km] += 1

    # select
    sel = []
    for k in range(args.min_k, args.max_k+1):
        items = [(km,c) for km,c in df[k].items() if c >= args.min_support]
        items.sort(key=lambda x: (-x[1], x[0]))
        items = items[:args.top_per_k]
        for km,c in items:
            sel.append((k, km, c))

    # global cap
    sel.sort(key=lambda x: (-x[2], x[0], x[1]))
    sel = sel[:args.kmer_cap_total]

    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("k\tkmer\tsupport\n")
        for k, km, c in sel:
            f.write(f"{k}\t{km}\t{c}\n")

    print(f"[DONE] seqs={n}  selected={len(sel)} -> {out}")

if __name__ == "__main__":
    main()
