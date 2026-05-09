#!/usr/bin/env python3
from pathlib import Path
import argparse, math, multiprocessing as mp
import pandas as pd

def gc_content(seq):
    seq = str(seq).upper()
    return 0.0 if len(seq) == 0 else (seq.count("G") + seq.count("C")) / len(seq)

def at_skew(seq):
    seq = str(seq).upper()
    a, t = seq.count("A"), seq.count("T")
    den = a + t
    return 0.0 if den == 0 else (a - t) / den

def gc_skew(seq):
    seq = str(seq).upper()
    g, c = seq.count("G"), seq.count("C")
    den = g + c
    return 0.0 if den == 0 else (g - c) / den

def shannon_entropy(seq):
    seq = str(seq).upper()
    if len(seq) == 0:
        return 0.0
    probs = []
    for base in "ATGC":
        p = seq.count(base) / len(seq)
        if p > 0:
            probs.append(p)
    return -sum(p * math.log(p, 2) for p in probs)

def lz_complexity(seq):
    seq = str(seq).upper()
    n = len(seq)
    if n == 0:
        return 0
    i, k, l = 0, 1, 1
    c = 1
    while True:
        if i + k > n or l + k > n:
            c += 1
            break
        if seq[i:i+k] == seq[l:l+k]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            i += 1
            if i == l:
                c += 1
                l += k
                if l >= n:
                    break
                i = 0
                k = 1
    return c

def max_homopolymer(seq):
    seq = str(seq).upper()
    if len(seq) == 0:
        return 0
    best, cur = 1, 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best

def process_row(r):
    seqs = {"global": r["sequence"], "head": r["head"], "mid1": r["mid1"], "mid2": r["mid2"], "mid3": r["mid3"], "tail": r["tail"]}
    row = {"sequence_id": r["sequence_id"], "marker": r["marker"]}
    for region, seq in seqs.items():
        row[f"len_{region}"] = len(seq)
        row[f"gc_{region}"] = gc_content(seq)
        row[f"at_skew_{region}"] = at_skew(seq)
        row[f"gc_skew_{region}"] = gc_skew(seq)
        row[f"entropy_{region}"] = shannon_entropy(seq)
        row[f"lz_{region}"] = lz_complexity(seq)
        row[f"homopolymer_max_{region}"] = max_homopolymer(seq)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence-regions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threads", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--chunksize", type=int, default=100)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.sequence_regions, sep="\t", dtype=str).fillna("")
    with mp.Pool(processes=args.threads) as pool:
        rows = list(pool.imap(process_row, df.to_dict("records"), chunksize=args.chunksize))
    pd.DataFrame(rows).to_csv(outdir / "global_regional_features.tsv", sep="\t", index=False)
    print(outdir / "global_regional_features.tsv")

if __name__ == "__main__":
    main()
