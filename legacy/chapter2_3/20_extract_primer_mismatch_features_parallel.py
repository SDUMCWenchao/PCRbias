#!/usr/bin/env python3
from pathlib import Path
import argparse, multiprocessing as mp
import pandas as pd

PRIMER_DICT = {}

def revcomp(seq):
    table = str.maketrans("ACGTRYMKBDHVN", "TGCAYRKMVHDBN")
    return seq.upper().translate(table)[::-1]

def mismatch_positions(a, b):
    a = a.upper(); b = b.upper()
    n = min(len(a), len(b))
    return [i for i in range(n) if a[i] != b[i]]

def weighted_score(mismatch_pos, length):
    if length == 0:
        return 0.0
    s = 0.0
    for pos in mismatch_pos:
        dist = length - 1 - pos
        s += 1.0 / (dist + 1)
    return s

def init_worker(primer_dict):
    global PRIMER_DICT
    PRIMER_DICT = primer_dict

def process_row(r):
    marker = r["marker"]
    if marker not in PRIMER_DICT:
        return None
    fwd, rev = PRIMER_DICT[marker]
    rev_rc = revcomp(rev)
    head_slice = r["head"].upper()[:len(fwd)]
    tail_slice = r["tail"].upper()[-len(rev_rc):] if len(rev_rc) > 0 else ""
    fwd_mm = mismatch_positions(head_slice, fwd)
    rev_mm = mismatch_positions(tail_slice, rev_rc)
    return {
        "sequence_id": r["sequence_id"],
        "marker": marker,
        "forward_primer_len": len(fwd),
        "reverse_primer_len": len(rev),
        "mismatch_fwd_total": len(fwd_mm),
        "mismatch_rev_total": len(rev_mm),
        "mismatch_total": len(fwd_mm) + len(rev_mm),
        "mismatch_fwd_weighted_score": weighted_score(fwd_mm, len(fwd)),
        "mismatch_rev_weighted_score": weighted_score(rev_mm, len(rev_rc)),
        "mismatch_weighted_score": weighted_score(fwd_mm, len(fwd)) + weighted_score(rev_mm, len(rev_rc)),
        "mismatch_3prime_total": int((len(fwd)-1 in fwd_mm) if len(fwd)>0 else 0) + int((len(rev_rc)-1 in rev_mm) if len(rev_rc)>0 else 0),
        "mismatch_fwd_positions": ",".join(map(str, fwd_mm)),
        "mismatch_rev_positions": ",".join(map(str, rev_mm)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence-regions", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threads", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--chunksize", type=int, default=200)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    regions = pd.read_csv(args.sequence_regions, sep="\t", dtype=str).fillna("")
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str).fillna("")
    primer_map = meta[["marker","forward_primer_5to3","reverse_primer_5to3"]].drop_duplicates("marker")
    primers = {r["marker"]: (r["forward_primer_5to3"].upper(), r["reverse_primer_5to3"].upper()) for _, r in primer_map.iterrows()}
    with mp.Pool(processes=args.threads, initializer=init_worker, initargs=(primers,)) as pool:
        rows = [x for x in pool.imap(process_row, regions.to_dict("records"), chunksize=args.chunksize) if x is not None]
    pd.DataFrame(rows).to_csv(outdir / "primer_mismatch_features.tsv", sep="\t", index=False)
    print(outdir / "primer_mismatch_features.tsv")

if __name__ == "__main__":
    main()
