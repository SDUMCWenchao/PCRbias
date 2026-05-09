#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Select case-study candidates for Chapter 3.")
    ap.add_argument("--species-table", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv(args.species_table, sep="\t")

    # main threshold only
    main_df = sp[sp["threshold_label"] == "rel_ge_0.005"].copy()
    if len(main_df) == 0:
        main_df = sp.copy()

    high = main_df.sort_values("log2_bias_ratio", ascending=False).head(20)
    low = main_df.sort_values("log2_bias_ratio", ascending=True).head(20)

    # marker-discordant species
    pivot = main_df.pivot_table(index=["sample_id","group_name","species_label"], columns="marker", values="log2_bias_ratio")
    pivot = pivot.reset_index()
    if {"12S","16S"}.issubset(set(pivot.columns)):
        pivot["marker_gap_abs"] = (pivot["12S"] - pivot["16S"]).abs()
        discordant = pivot.sort_values("marker_gap_abs", ascending=False).head(20)
    else:
        discordant = pivot.head(0)

    high.to_csv(outdir / "case_candidates_high_bias.tsv", sep="\t", index=False)
    low.to_csv(outdir / "case_candidates_low_bias.tsv", sep="\t", index=False)
    discordant.to_csv(outdir / "case_candidates_marker_discordant.tsv", sep="\t", index=False)

    print(f"Wrote {outdir / 'case_candidates_high_bias.tsv'}")
    print(f"Wrote {outdir / 'case_candidates_low_bias.tsv'}")
    print(f"Wrote {outdir / 'case_candidates_marker_discordant.tsv'}")

if __name__ == "__main__":
    main()
