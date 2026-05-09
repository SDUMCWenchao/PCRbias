#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Merge sequence annotation table into master abundance.")
    ap.add_argument("--abundance", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ab = pd.read_csv(args.abundance, sep="\t")
    ann = pd.read_csv(args.annotation, sep="\t", dtype=str).fillna("")

    keep_cols = [c for c in ["sequence_id","species_label","annotation_status","is_target_candidate","notes"] if c in ann.columns]
    ann = ann[keep_cols].drop_duplicates("sequence_id")

    merged = ab.merge(ann, on="sequence_id", how="left")
    merged["species_label"] = merged["species_label"].replace("", "UNANNOTATED").fillna("UNANNOTATED")
    merged["annotation_status"] = merged["annotation_status"].replace("", "unannotated").fillna("unannotated")

    merged.to_csv(outdir / "master_long_abundance_annotated.tsv", sep="\t", index=False)

    seq_status = (
        merged.groupby(["sequence_id","species_label","annotation_status"], as_index=False)
        .agg(total_count=("count","sum"), n_samples=("sample_id","nunique"))
        .sort_values("total_count", ascending=False)
    )
    seq_status.to_csv(outdir / "annotation_coverage_summary.tsv", sep="\t", index=False)

    print(f"Wrote {outdir / 'master_long_abundance_annotated.tsv'}")
    print(f"Wrote {outdir / 'annotation_coverage_summary.tsv'}")

if __name__ == "__main__":
    main()
