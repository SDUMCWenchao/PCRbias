#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def norm_id(x: str) -> str:
    s = str(x).strip().split("/")[-1]
    s = s.replace(".gz", "")
    for suf in [".fastq", ".fq", ".fasta", ".fa", ".tsv", ".txt", ".csv"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    for suf in [".trim", ".trimmed", ".merged", ".clean", ".filtered"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--chunks_dir", default="analysis_results/03_DataWeaver/training_chunks")
    ap.add_argument("--samples_meta", default="scripts/samples_meta.tsv")
    ap.add_argument("--max_files", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    project = Path(args.project_dir)
    chunks = project / args.chunks_dir
    meta = pd.read_csv(project / args.samples_meta, sep="\t")

    # build allowed pairs strictly from meta: species/locus + pcr yes/no + n_individuals>1
    allowed = set()
    for (sp, lc), g in meta.groupby(["species", "locus"], sort=False):
        g2 = g[g["n_individuals"].astype(int) > 1]
        ys = g2[g2["pcr"].astype(str) == "yes"]["file_id"].astype(str).tolist()
        ns = g2[g2["pcr"].astype(str) == "no"]["file_id"].astype(str).tolist()
        if len(ys) == 1 and len(ns) == 1:
            allowed.add((ys[0], ns[0]))  # (yes, no)
        else:
            print(f"[WARN] ambiguous allowed pair for {sp}/{lc}: yes={ys} no={ns}")

    print("[INFO] allowed pairs (yes,no):")
    for a in sorted(allowed):
        print("   ", a)

    files = sorted(chunks.glob("chunk_*.train.tsv.gz")) + sorted(chunks.glob("chunk_*.train.tsv"))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"[ERROR] no chunk files in {chunks}")

    bad = Counter()
    ok = Counter()
    same_pcr = Counter()
    meta_map = {str(r.file_id): r for r in meta.itertuples(index=False)}

    for fp in files:
        df = pd.read_csv(fp, sep="\t", compression="gzip" if str(fp).endswith(".gz") else None,
                         usecols=["yes_file_id", "no_file_id"])
        if df.empty:
            continue
        y = df["yes_file_id"].astype(str).map(norm_id)
        n = df["no_file_id"].astype(str).map(norm_id)

        for yy, nn in zip(y.tolist(), n.tolist()):
            ry = meta_map.get(yy)
            rn = meta_map.get(nn)
            if ry is None or rn is None:
                bad[(yy, nn)] += 1
                continue
            # strict condition: same species & locus, and pcr differs
            if (ry.species != rn.species) or (ry.locus != rn.locus):
                bad[(yy, nn)] += 1
                continue
            if str(ry.pcr) == str(rn.pcr):
                same_pcr[(yy, nn)] += 1
                continue

            # must match allowed set (either orientation)
            if (yy, nn) in allowed:
                ok[(yy, nn)] += 1
            elif (nn, yy) in allowed:
                # reversed in file; still "allowed comparison" but orientation is swapped
                ok[(yy, nn)] += 1
            else:
                bad[(yy, nn)] += 1

    print(f"\n[INFO] ok_pairs={len(ok)}  bad_pairs={len(bad)}  same_pcr_pairs={len(same_pcr)}")

    if same_pcr:
        print("\n[WARN] pcr same on both sides (should not be compared): top10")
        for (k, v) in same_pcr.most_common(10):
            print("   ", k, v)

    if bad:
        print("\n[ERROR] found disallowed comparisons (species/locus mismatch OR not in allowed set): top20")
        for (k, v) in bad.most_common(20):
            print("   ", k, v)

    print("\n[DONE] audit finished.")


if __name__ == "__main__":
    main()
