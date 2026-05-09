#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import gzip
from collections import Counter

# -------------------------
# utils
# -------------------------
def read_tsv(p: Path):
    if str(p).endswith(".gz"):
        return pd.read_csv(p, sep="\t", compression="gzip")
    return pd.read_csv(p, sep="\t")

def write_tsv(df: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, sep="\t", index=False)

def load_y_array(p: Path):
    # support .npy or .npz (single array)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        if len(z.files) != 1:
            raise ValueError(f"npz has multiple arrays: {p} -> {z.files}")
        return z[z.files[0]]
    raise ValueError(f"unsupported y format: {p}")

def iupac_map():
    return {
        "A": set("A"), "C": set("C"), "G": set("G"), "T": set("T"),
        "R": set("AG"), "Y": set("CT"), "S": set("GC"), "W": set("AT"),
        "K": set("GT"), "M": set("AC"), "B": set("CGT"), "D": set("AGT"),
        "H": set("ACT"), "V": set("ACG"), "N": set("ACGT"),
    }

def revcomp(seq: str):
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]

def iupac_match_prefix(s: str, primer: str, L: int):
    # exact match with IUPAC ambiguity on primer side, length L
    mp = iupac_map()
    s = s.upper()
    primer = primer.upper()
    if len(s) < L or len(primer) < L:
        return False
    for i in range(L):
        b = s[i]
        p = primer[i]
        if b not in "ACGT":
            return False
        if b not in mp.get(p, set()):
            return False
    return True

def stream_fasta(path: Path):
    name = None
    seq_chunks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq_chunks)
                name = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if name is not None:
            yield name, "".join(seq_chunks)

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=".")
    ap.add_argument("--plot_tables_dir", default="analysis_results/07_PlotTables_v3_topbias")
    ap.add_argument("--models_root", default="analysis_results/06_Models_v3_topbias")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--samples_meta", default="samples_meta.tsv",
                    help="path to samples_meta.tsv (will also try analysis_results/**/samples_meta.tsv)")
    ap.add_argument("--fasta", default="analysis_results/01_Sequences/ALL_UNIQUE_SEQUENCES.fasta")
    ap.add_argument("--out_dir", default="analysis_results/07_PlotTables_v3_topbias/A_tables")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--make_pairs_tables", action="store_true",
                    help="build A2 pair manifest tables from samples_meta.tsv")
    ap.add_argument("--make_y_tables", action="store_true",
                    help="build A3/A4 y distribution + topbias thresholds from model inputs y_*.npy")
    ap.add_argument("--make_primer_tables", action="store_true",
                    help="build A5 primer residue + head20 fingerprint from ALL_UNIQUE_SEQUENCES.fasta")
    ap.add_argument("--head", type=int, default=30)
    ap.add_argument("--tail", type=int, default=30)
    ap.add_argument("--primer_L", type=int, default=20)
    ap.add_argument("--fingerprint_L", type=int, default=20)
    ap.add_argument("--fingerprint_topN", type=int, default=20)

    args = ap.parse_args()

    proj = Path(args.project_dir).resolve()
    plot_dir = (proj / args.plot_tables_dir).resolve()
    out_dir = (proj / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    master = plot_dir / "01_master_metrics_ci.tsv"
    if not master.exists():
        raise FileNotFoundError(f"missing {master}")
    dfm = read_tsv(master)

    # -------- A1: group overview (dedupe to tag×species×locus) --------
    a1 = dfm.drop_duplicates(subset=["tag","species","locus"])[
        ["tag","species","locus","n_test","n_pairs","flag_small_test","flag_tiny_test","flag_few_pairs"]
    ].sort_values(["tag","species","locus"])
    p_a1 = out_dir / "A1_group_overview.tsv"
    if p_a1.exists() and not args.overwrite:
        raise FileExistsError(f"exists: {p_a1} (use --overwrite)")
    write_tsv(a1, p_a1)

    # -------- A1b: variant catalog (what ran) --------
    a1b = (dfm.groupby(["tag","species","locus","model"])["variant"]
           .apply(lambda x: ",".join(sorted(set(x))))
           .reset_index()
           .rename(columns={"variant":"variants"}))
    p_a1b = out_dir / "A1b_variant_catalog.tsv"
    if p_a1b.exists() and not args.overwrite:
        raise FileExistsError(f"exists: {p_a1b} (use --overwrite)")
    write_tsv(a1b, p_a1b)

    # -------- A1c: wide presence matrix (nice for Supplementary table) --------
    pres = dfm[["tag","species","locus","model","variant"]].copy()
    pres["present"] = 1
    a1c = pres.pivot_table(index=["tag","species","locus","model"],
                           columns="variant", values="present", fill_value=0, aggfunc="max").reset_index()
    p_a1c = out_dir / "A1c_variant_presence_matrix.tsv"
    if p_a1c.exists() and not args.overwrite:
        raise FileExistsError(f"exists: {p_a1c} (use --overwrite)")
    write_tsv(a1c, p_a1c)

    # -------- A split sizes from metrics.json (train/val/test n) --------
    # robust: infer model_dir via pred_path, then read metrics.json in same dir
    rows = []
    for _, r in dfm.drop_duplicates(subset=["tag","species","locus","model","variant"]).iterrows():
        pred_path = (proj / str(r["pred_path"])).resolve()
        mdir = pred_path.parent
        mj = mdir / "metrics.json"
        if not mj.exists():
            continue
        d = json.loads(mj.read_text())
        rows.append({
            "tag": r["tag"], "species": r["species"], "locus": r["locus"], "model": r["model"], "variant": r["variant"],
            "train_n": d.get("train_n", np.nan),
            "val_n": d.get("val_n", np.nan),
            "test_n": d.get("test_n", np.nan),
        })
    a_split = pd.DataFrame(rows)
    p_as = out_dir / "A_split_sizes_from_metrics.tsv"
    if p_as.exists() and not args.overwrite:
        raise FileExistsError(f"exists: {p_as} (use --overwrite)")
    write_tsv(a_split, p_as)

    # -------- optional: A2 pair tables from samples_meta.tsv --------
    if args.make_pairs_tables:
        cand = [proj / args.samples_meta]
        # try to auto-find if user keeps it elsewhere
        cand += list(proj.glob("analysis_results/**/samples_meta.tsv"))
        meta_path = None
        for c in cand:
            if c.exists():
                meta_path = c
                break
        if meta_path is None:
            raise FileNotFoundError("samples_meta.tsv not found. Pass --samples_meta explicitly.")

        meta = read_tsv(meta_path)

        # try common column names (won't crash if missing)
        # you can rename columns in meta to fit these keys if needed
        colmap = {}
        for k in ["species","locus","pcr","pcr_status","is_pcr","PCR","individual","individual_id","pair_id","file_id","sample_id","n_individuals"]:
            if k in meta.columns:
                colmap[k]=k

        # minimal manifest: group counts and skip n_individuals==1
        if "n_individuals" in meta.columns:
            meta["skip_no_control"] = (pd.to_numeric(meta["n_individuals"], errors="coerce") <= 1).astype(int)
        else:
            meta["skip_no_control"] = 0

        # If you already have explicit yes/no pairing columns, keep them
        keep_cols = [c for c in ["species","locus","pair_id","individual_id","file_id","pcr","pcr_status","n_individuals","skip_no_control"] if c in meta.columns]
        a2_manifest = meta[keep_cols].copy() if keep_cols else meta.copy()
        p_a2 = out_dir / "A2_samples_meta_subset.tsv"
        if p_a2.exists() and not args.overwrite:
            raise FileExistsError(f"exists: {p_a2} (use --overwrite)")
        write_tsv(a2_manifest, p_a2)

        # summary counts
        grp_cols = [c for c in ["species","locus"] if c in meta.columns]
        if grp_cols:
            a2_sum = meta.groupby(grp_cols).agg(
                n_rows=("skip_no_control","size"),
                n_skip=("skip_no_control","sum"),
            ).reset_index()
            p_a2s = out_dir / "A2_group_counts_and_skips.tsv"
            if p_a2s.exists() and not args.overwrite:
                raise FileExistsError(f"exists: {p_a2s} (use --overwrite)")
            write_tsv(a2_sum, p_a2s)

    # -------- optional: A3/A4 y distribution + topbias threshold --------
    if args.make_y_tables:
        inputs_root = (proj / args.inputs_root).resolve()
        rec_q = []
        rec_hist = []

        qs = [0,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.995,0.999,1.0]
        bins = np.linspace(-6, 6, 61)  # y_clip=6 的常用范围；如果你改过 y_clip，可自行改这个

        base_variant = "no_kmer"

        for tag in sorted(dfm["tag"].unique()):
            for sp in sorted(dfm["species"].unique()):
                for lc in sorted(dfm["locus"].unique()):
                    ddir = inputs_root / tag / sp / lc / base_variant
                    y_all = []
                    for split in ["train","val","test"]:
                        yfile_npy = ddir / f"y_{split}.npy"
                        yfile_npz = ddir / f"y_{split}.npz"
                        yfile = yfile_npy if yfile_npy.exists() else (yfile_npz if yfile_npz.exists() else None)
                        if yfile is None:
                            continue
                        y = load_y_array(yfile).astype(float)
                        y_all.append(y)

                        # hist
                        h, e = np.histogram(y, bins=bins)
                        for i in range(len(h)):
                            rec_hist.append({
                                "tag":tag,"species":sp,"locus":lc,"split":split,
                                "bin_left": float(e[i]), "bin_right": float(e[i+1]),
                                "count": int(h[i])
                            })

                        # quantiles per split
                        qv = np.quantile(y, qs)
                        row = {"tag":tag,"species":sp,"locus":lc,"split":split}
                        for q, val in zip(qs, qv):
                            row[f"q{q}"] = float(val)
                        row["min_abs_y"] = float(np.min(np.abs(y)))
                        rec_q.append(row)

                    # combined threshold estimate (min abs over all splits)
                    if y_all:
                        yy = np.concatenate(y_all)
                        rec_q.append({
                            "tag":tag,"species":sp,"locus":lc,"split":"all",
                            "min_abs_y": float(np.min(np.abs(yy))),
                            **{f"q{q}": float(v) for q, v in zip(qs, np.quantile(yy, qs))}
                        })

        dfq = pd.DataFrame(rec_q)
        dfh = pd.DataFrame(rec_hist)

        p_q = out_dir / "A3A4_y_quantiles_and_thresholds.tsv"
        p_h = out_dir / "A3A4_y_hist.tsv"
        if (p_q.exists() or p_h.exists()) and not args.overwrite:
            raise FileExistsError(f"exists: {p_q} or {p_h} (use --overwrite)")
        write_tsv(dfq, p_q)
        write_tsv(dfh, p_h)

    # -------- optional: A5 primer residue + fingerprint --------
    if args.make_primer_tables:
        fasta = (proj / args.fasta).resolve()
        if not fasta.exists():
            raise FileNotFoundError(f"missing fasta: {fasta}")

        primers = {
            "12S_F": "GGGATTAGATACCCCACTATGCYTA",
            "12S_R": "GAGGGTGACGGGCGGTGT",
            "16S_F": "ACCAAAAACATCACCTCYAGCAT",
            "16S_R": "AATAGGATTGCGCTGTTATCCCTA",
        }

        head_counts = Counter()
        tail_counts = Counter()
        head20 = Counter()
        total = 0

        for _, seq in stream_fasta(fasta):
            seq = seq.strip().upper()
            if not seq:
                continue
            total += 1
            hwin = seq[:args.head]
            twin = seq[-args.tail:] if len(seq) >= args.tail else seq

            # head: match primer at window start
            for name, pr in primers.items():
                if iupac_match_prefix(hwin, pr, args.primer_L):
                    head_counts[name] += 1

            # tail: match revcomp(primer) at tail-window start (same logic as你之前打印“TAIL revcomp at window start”)
            for name, pr in primers.items():
                prc = revcomp(pr)
                if iupac_match_prefix(twin, prc, args.primer_L):
                    tail_counts[name] += 1

            # fingerprint head20
            if len(seq) >= args.fingerprint_L:
                head20[seq[:args.fingerprint_L]] += 1

        rec = []
        for name in primers.keys():
            rec.append({
                "primer": name,
                "where": "HEAD",
                "match_L": args.primer_L,
                "count": int(head_counts.get(name, 0)),
                "frac": float(head_counts.get(name, 0) / total if total else 0.0),
                "total_seqs": int(total),
            })
        for name in primers.keys():
            rec.append({
                "primer": name,
                "where": "TAIL_revcomp",
                "match_L": args.primer_L,
                "count": int(tail_counts.get(name, 0)),
                "frac": float(tail_counts.get(name, 0) / total if total else 0.0),
                "total_seqs": int(total),
            })
        dfp = pd.DataFrame(rec)

        topN = head20.most_common(args.fingerprint_topN)
        dff = pd.DataFrame([{
            "rank": i+1,
            "headN": k,
            "count": int(v),
            "frac": float(v/total if total else 0.0)
        } for i,(k,v) in enumerate(topN)])

        p_p = out_dir / "A5_primer_residue.tsv"
        p_f = out_dir / "A5_head_fingerprint_topN.tsv"
        if (p_p.exists() or p_f.exists()) and not args.overwrite:
            raise FileExistsError(f"exists: {p_p} or {p_f} (use --overwrite)")
        write_tsv(dfp, p_p)
        write_tsv(dff, p_f)

    print("[DONE] A tables ->", out_dir)
    print("  - A1_group_overview.tsv")
    print("  - A1b_variant_catalog.tsv")
    print("  - A1c_variant_presence_matrix.tsv")
    print("  - A_split_sizes_from_metrics.tsv")
    print("  - (optional) A2_* , A3A4_* , A5_*")

if __name__ == "__main__":
    main()
