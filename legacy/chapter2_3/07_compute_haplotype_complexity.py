#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def shannon(p):
    p = np.asarray([x for x in p if x > 0], dtype=float)
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0

def simpson_1_minus_d(p):
    p = np.asarray([x for x in p if x > 0], dtype=float)
    return float(1.0 - np.square(p).sum()) if len(p) else 0.0

def pielou_evenness(h, richness):
    return 0.0 if richness <= 1 else float(h / np.log(richness))

def make_prefix_species(group_name: str) -> str:
    if str(group_name).startswith("Ea"):
        return "Equus_asinus"
    if str(group_name).startswith("Bt"):
        return "Bos_taurus"
    if str(group_name).startswith("Ss"):
        return "Sus_scrofa"
    return "Unknown"

def main():
    ap = argparse.ArgumentParser(description="Compute intra-sample haplotype complexity metrics.")
    ap.add_argument("--abundance", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-relative-abundance", type=float, default=0.01, help="Default 0.01 for main-line chapter analysis")
    ap.add_argument("--count-min", type=int, default=2)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.abundance, sep="\t")
    df = df[
        (df["sample_type"] == "intra_mix") &
        (df["is_core_analysis"].astype(str).str.lower() == "yes") &
        (df["count"] >= args.count_min) &
        (df["relative_abundance"] >= args.min_relative_abundance)
    ].copy()

    rows = []
    for keys, sub in df.groupby(["sample_id","marker","group_name","sample_type","species_scope"], dropna=False):
        sub = sub.sort_values("relative_abundance", ascending=False).copy()
        p = sub["count"] / sub["count"].sum()
        richness = int(sub["sequence_id"].nunique())
        H = shannon(p)
        S = simpson_1_minus_d(p)
        dominant = float(sub["relative_abundance"].iloc[0]) if len(sub) else 0.0
        top3 = float(sub["relative_abundance"].head(3).sum()) if len(sub) else 0.0
        top10 = float(sub["relative_abundance"].head(10).sum()) if len(sub) else 0.0
        tail = float(sub["relative_abundance"].iloc[1:].sum()) if len(sub) > 1 else 0.0
        rows.append({
            "sample_id": keys[0],
            "marker": keys[1],
            "group_name": keys[2],
            "sample_type": keys[3],
            "species_scope": keys[4],
            "target_species_inferred": make_prefix_species(keys[2]),
            "threshold_rel": args.min_relative_abundance,
            "count_min": args.count_min,
            "total_reads_retained": int(sub["count"].sum()),
            "n_unique_sequences": richness,
            "shannon": H,
            "simpson_1_minus_d": S,
            "pielou_evenness": pielou_evenness(H, richness),
            "dominant_rel_abundance": dominant,
            "top3_rel_abundance": top3,
            "top10_rel_abundance": top10,
            "tail_rel_abundance": tail,
        })

    metrics = pd.DataFrame(rows).sort_values(["marker","group_name","sample_id"])
    metrics.to_csv(outdir / "haplotype_complexity_intra_mix.tsv", sep="\t", index=False)

    by_group = metrics.groupby(["marker","group_name","target_species_inferred"], as_index=False).agg(
        n_samples=("sample_id","nunique"),
        mean_n_unique_sequences=("n_unique_sequences","mean"),
        mean_shannon=("shannon","mean"),
        mean_simpson_1_minus_d=("simpson_1_minus_d","mean"),
        mean_pielou_evenness=("pielou_evenness","mean"),
        mean_dominant_rel_abundance=("dominant_rel_abundance","mean"),
        mean_tail_rel_abundance=("tail_rel_abundance","mean"),
    )
    by_group.to_csv(outdir / "haplotype_complexity_group_summary.tsv", sep="\t", index=False)

    print(f"Wrote {outdir / 'haplotype_complexity_intra_mix.tsv'}")
    print(f"Wrote {outdir / 'haplotype_complexity_group_summary.tsv'}")

if __name__ == "__main__":
    main()
