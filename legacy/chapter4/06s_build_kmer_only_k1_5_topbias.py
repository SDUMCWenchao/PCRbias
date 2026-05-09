#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

BASE2 = {'A':0,'C':1,'G':2,'T':3}

def fasta_fetch(fasta_path: Path, wanted_ids: set):
    seqs = {}
    cur_id = None
    buf = []
    keep = False
    with fasta_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None and keep:
                    seqs[cur_id] = "".join(buf).upper()
                cur_id = line[1:].strip().split()[0]
                buf = []
                keep = cur_id in wanted_ids
            else:
                if keep:
                    buf.append(line.strip())
        if cur_id is not None and keep:
            seqs[cur_id] = "".join(buf).upper()
    return seqs

def mid_bins_split(s: str, head: int, tail: int, mid_bins: int):
    L = len(s)
    head_s = s[:min(head, L)]
    tail_s = s[max(0, L-tail):] if tail > 0 else ""
    mid_start = min(head, L)
    mid_end = max(0, L-tail)
    if mid_end < mid_start:
        mid_end = mid_start
    mid_s = s[mid_start:mid_end]
    bins = []
    m = len(mid_s)
    for b in range(mid_bins):
        a = (b * m) // mid_bins
        z = ((b + 1) * m) // mid_bins
        bins.append(mid_s[a:z])
    regions = [("head30", head_s)]
    for i, bb in enumerate(bins):
        regions.append((f"mid{i}", bb))
    regions.append((f"tail30", tail_s))
    return regions

def rolling_kmer_codes(seq: str, k: int):
    # yields integer code for each valid A/C/G/T kmer; reset across invalid chars
    mask = (1 << (2*k)) - 1
    code = 0
    valid = 0
    for ch in seq:
        v = BASE2.get(ch, None)
        if v is None:
            code = 0
            valid = 0
            continue
        code = ((code << 2) | v) & mask
        valid += 1
        if valid >= k:
            yield code

def int_to_kmer(x: int, k: int):
    m = ["A","C","G","T"]
    out = []
    for _ in range(k):
        out.append(m[x & 3])
        x >>= 2
    return "".join(reversed(out))

def build_X(meta: pd.DataFrame, seqs: dict, k: int, head: int, tail: int, mid_bins: int, x_mode: str):
    # columns: region-major then kmer-code order
    regions = ["head30"] + [f"mid{i}" for i in range(mid_bins)] + ["tail30"]
    R = len(regions)
    Vk = 4 ** k
    ncol = R * Vk

    row_idx = []
    col_idx = []
    data = []

    for i, sid in enumerate(meta["Seq_ID"].astype(str).tolist()):
        s = seqs.get(sid, "")
        if not s:
            continue  # keep row empty
        regs = mid_bins_split(s, head, tail, mid_bins)

        if x_mode == "presence":
            cols = set()
            for r_i, (rname, rseq) in enumerate(regs):
                base = r_i * Vk
                for code in rolling_kmer_codes(rseq, k):
                    cols.add(base + code)
            for c in cols:
                row_idx.append(i)
                col_idx.append(c)
                data.append(1.0)
        else:  # count
            counts = {}
            for r_i, (rname, rseq) in enumerate(regs):
                base = r_i * Vk
                for code in rolling_kmer_codes(rseq, k):
                    c = base + code
                    counts[c] = counts.get(c, 0) + 1
            for c, v in counts.items():
                row_idx.append(i)
                col_idx.append(c)
                data.append(float(v))

    X = sparse.coo_matrix(
        (np.array(data, dtype=np.float32),
         (np.array(row_idx, dtype=np.int64), np.array(col_idx, dtype=np.int64))),
        shape=(len(meta), ncol),
        dtype=np.float32
    ).tocsr()
    return X, regions

def write_dataset(out_ds: Path, ref_ds: Path, Xs: dict, feat_names: list, cfg: dict):
    out_ds.mkdir(parents=True, exist_ok=True)
    # copy y/w/meta
    for split in ["train","val","test"]:
        for fn in [f"y_{split}.npy", f"w_{split}.npy", f"meta_{split}.tsv.gz"]:
            (out_ds / fn).write_bytes((ref_ds / fn).read_bytes())
        sparse.save_npz(out_ds / f"X_{split}.npz", Xs[split].tocsr())
    (out_ds / "feature_names.txt").write_text("\n".join(feat_names) + "\n", encoding="utf-8")
    (out_ds / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--tags", default="top1p,top0p5p,top0p1p")
    ap.add_argument("--fasta", default="analysis_results/01_Sequences/ALL_UNIQUE_SEQUENCES.fasta")
    ap.add_argument("--head_win", type=int, default=30)
    ap.add_argument("--tail_win", type=int, default=30)
    ap.add_argument("--mid_bins", type=int, default=3)
    ap.add_argument("--k_max", type=int, default=5)
    ap.add_argument("--x_mode", default="presence", choices=["presence","count"])
    ap.add_argument("--ref_variant", default="no_kmer_noprimer",
                    help="copy y/w/meta splits from this variant dir under each species/locus")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    project = Path(args.project_dir)
    root = project / args.inputs_root
    fasta_path = project / args.fasta
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    for tag in tags:
        tag_dir = root / tag
        if not tag_dir.exists():
            print(f"[WARN] missing tag dir: {tag_dir}")
            continue

        for sp_dir in sorted(tag_dir.glob("*")):
            if not sp_dir.is_dir():
                continue
            sp = sp_dir.name
            for lc_dir in sorted(sp_dir.glob("*")):
                if not lc_dir.is_dir():
                    continue
                lc = lc_dir.name

                ref_ds = lc_dir / args.ref_variant
                if not (ref_ds / "meta_train.tsv.gz").exists():
                    print(f"[WARN] missing ref variant {args.ref_variant} for {tag}/{sp}/{lc} -> skip")
                    continue

                # load metas to get wanted Seq_ID set
                metas = {}
                wanted = set()
                for split in ["train","val","test"]:
                    m = pd.read_csv(ref_ds / f"meta_{split}.tsv.gz", sep="\t", compression="gzip")
                    if "Seq_ID" not in m.columns:
                        raise SystemExit(f"[ERROR] meta_{split} has no Seq_ID in {ref_ds}")
                    metas[split] = m
                    wanted.update(m["Seq_ID"].astype(str).tolist())

                print(f"[INFO] fetching sequences: {tag}/{sp}/{lc} wanted={len(wanted)}")
                seqs = fasta_fetch(fasta_path, wanted)

                miss = len(wanted) - len(seqs)
                if miss > 0:
                    print(f"[WARN] {tag}/{sp}/{lc} missing sequences in fasta: {miss} (these rows will be all-zero)")

                for k in range(1, args.k_max + 1):
                    out_ds = lc_dir / f"kmer_only_k{k}"
                    if (out_ds / "X_train.npz").exists() and not args.force:
                        continue

                    Xs = {}
                    # build split matrices
                    for split in ["train","val","test"]:
                        X, regions = build_X(metas[split], seqs, k, args.head_win, args.tail_win, args.mid_bins, args.x_mode)
                        Xs[split] = X

                    # feature names
                    Vk = 4 ** k
                    feat_names = []
                    reg_names = ["head30"] + [f"mid{i}" for i in range(args.mid_bins)] + ["tail30"]
                    for rname in reg_names:
                        for code in range(Vk):
                            feat_names.append(f"k{k}_{rname}_{int_to_kmer(code, k)}")

                    cfg = {
                        "variant": f"kmer_only_k{k}",
                        "k": k,
                        "x_mode": args.x_mode,
                        "head_win": args.head_win,
                        "tail_win": args.tail_win,
                        "mid_bins": args.mid_bins,
                        "note": "k<=5 uses full enumeration of all 4^k kmers per region; built for topbias only"
                    }
                    write_dataset(out_ds, ref_ds, Xs, feat_names, cfg)
                    print(f"[DONE] built {tag}/{sp}/{lc}/kmer_only_k{k}  nfeat={len(feat_names)}")

    print("[DONE] all k1..k5 datasets built.")

if __name__ == "__main__":
    main()
