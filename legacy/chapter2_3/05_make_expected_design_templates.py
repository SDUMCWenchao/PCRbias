#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

PREFIX_TO_SPECIES = {
    "Ea": "Equus_asinus",
    "Bt": "Bos_taurus",
    "Ss": "Sus_scrofa",
}

def infer_species_from_group(group_name: str) -> str:
    for prefix, species in PREFIX_TO_SPECIES.items():
        if str(group_name).startswith(prefix):
            return species
    return "REPLACE_WITH_TARGET_SPECIES"

def main():
    ap = argparse.ArgumentParser(description="Create expected design templates for Chapter 2 analyses.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-inter-targets", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata, sep="\t", dtype=str).fillna("")
    core = meta[meta["is_core_analysis"].str.lower() == "yes"].copy()

    inter = core[core["sample_type"] == "inter_mix"].copy()
    intra = core[core["sample_type"] == "intra_mix"].copy()

    inter_rows = []
    for _, row in inter.iterrows():
        for i in range(1, args.n_inter_targets + 1):
            inter_rows.append({
                "sample_id": row["sample_id"],
                "marker": row["marker"],
                "group_name": row["group_name"],
                "target_label": f"TARGET_{i:02d}",
                "expected_prop": round(1.0 / args.n_inter_targets, 6),
                "notes": "Edit target_label to the exact species_label used in your sequence annotation table",
            })
    inter_df = pd.DataFrame(inter_rows)

    intra_rows = []
    for _, row in intra.iterrows():
        intra_rows.append({
            "sample_id": row["sample_id"],
            "marker": row["marker"],
            "group_name": row["group_name"],
            "target_label": infer_species_from_group(row["group_name"]),
            "expected_prop": 1.0,
            "notes": "Within-species mixed sample: target species should account for the intended signal at species level",
        })
    intra_df = pd.DataFrame(intra_rows)

    inter_df.to_csv(outdir / "expected_inter_targets_template.tsv", sep="\t", index=False)
    intra_df.to_csv(outdir / "expected_intra_targets.tsv", sep="\t", index=False)

    sample_design = core[["sample_id","marker","group_name","sample_type","species_scope","is_core_analysis"]].copy()
    sample_design.to_csv(outdir / "chapter2_sample_design.tsv", sep="\t", index=False)

    print(f"Wrote {outdir / 'expected_inter_targets_template.tsv'}")
    print(f"Wrote {outdir / 'expected_intra_targets.tsv'}")
    print(f"Wrote {outdir / 'chapter2_sample_design.tsv'}")

if __name__ == "__main__":
    main()
