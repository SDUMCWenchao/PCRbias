#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, gzip, re, json
from pathlib import Path
import numpy as np
import joblib

def iter_fasta(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    name = None
    seq = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name is not None:
            yield name, "".join(seq)

def encode_seq(s: str):
    # A,C,G,T -> 0..3, others -> 4
    m = {"A":0,"C":1,"G":2,"T":3,
         "a":0,"c":1,"g":2,"t":3}
    out = np.fromiter((m.get(ch, 4) for ch in s), dtype=np.uint8, count=len(s))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_len", type=int, default=0, help="optional clamp/pad length; 0=auto max")
    args = ap.parse_args()

    fasta = Path(args.fasta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # pass1: count + maxlen
    n = 0
    maxlen = 0
    for _, seq in iter_fasta(fasta):
        n += 1
        if len(seq) > maxlen:
            maxlen = len(seq)
    if args.max_len and args.max_len > 0:
        L = args.max_len
    else:
        L = maxlen

    tokens_path = out_dir / "seq_tokens.npy"
    ids_path = out_dir / "seq_ids.txt"
    map_path = out_dir / "seq_id_to_row.pkl"
    meta_path = out_dir / "meta.json"

    arr = np.lib.format.open_memmap(tokens_path, mode="w+", dtype=np.uint8, shape=(n, L))
    arr[:] = 4  # pad token

    ids = []
    # pass2: fill
    i = 0
    for name, seq in iter_fasta(fasta):
        ids.append(name)
        t = encode_seq(seq)
        if len(t) >= L:
            arr[i, :] = t[:L]
        else:
            arr[i, :len(t)] = t
        i += 1

    ids_path.write_text("\n".join(ids) + "\n")

    # mapping
    id2row = {sid: idx for idx, sid in enumerate(ids)}
    joblib.dump(id2row, map_path, compress=3)

    meta = {"n": n, "L": L, "pad_token": 4, "fasta": str(fasta)}
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[DONE] seqbank -> {out_dir}")
    print(f"[INFO] n={n} L={L} tokens={tokens_path} map={map_path}")

if __name__ == "__main__":
    main()
