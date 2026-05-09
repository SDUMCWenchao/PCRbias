#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

DEFAULT_THRESHOLDS = ["rel_ge_0.005", "rel_ge_0.01"]

def compat_cols(df):
    if "marker" not in df.columns:
        if "marker_x" in df.columns:
            df["marker"] = df["marker_x"]
        elif "marker_y" in df.columns:
            df["marker"] = df["marker_y"]
    if "group_name" not in df.columns:
        if "group_name_x" in df.columns:
            df["group_name"] = df["group_name_x"]
        elif "group_name_y" in df.columns:
            df["group_name"] = df["group_name_y"]
    if "species_label" not in df.columns:
        if "species_label_x" in df.columns:
            df["species_label"] = df["species_label_x"]
        elif "species_label_y" in df.columns:
            df["species_label"] = df["species_label_y"]
    return df

def weighted_feature_aggregate(df, feature_cols, weight_col):
    out_rows = []
    gb_cols = ["threshold_label","sample_id","marker","group_name","species_label"]
    for keys, sub in df.groupby(gb_cols, dropna=False):
        row = {
            "threshold_label": keys[0],
            "sample_id": keys[1],
            "marker": keys[2],
            "group_name": keys[3],
            "species_label": keys[4],
            "n_sequences_in_species": sub["sequence_id"].nunique(),
            "species_total_count_retained": sub["count"].sum(),
        }
        w = sub[weight_col].astype(float).values
        wsum = w.sum()
        for col in feature_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").values
            if np.isnan(vals).all() or wsum == 0:
                row[col] = np.nan
            else:
                mask = ~np.isnan(vals)
                row[col] = np.average(vals[mask], weights=w[mask]) if mask.any() else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def main():
    ap = argparse.ArgumentParser(description="Build Chapter 3 species-level and haplotype-level analysis tables.")
    ap.add_argument("--renorm-threshold-dir", required=True)
    ap.add_argument("--bias-dir", required=True, help="chapter2/bias_multi_threshold_renorm")
    ap.add_argument("--feature-files", nargs="+", required=True, help="TSV feature files keyed by sequence_id")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold-labels", nargs="*", default=DEFAULT_THRESHOLDS)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # merge feature tables
    feat = None
    for fp in args.feature_files:
        x = pd.read_csv(fp, sep="\t", dtype=str).fillna("")
        keep = [c for c in x.columns if c != "sequence"]
        if feat is None:
            feat = x[keep].copy()
        else:
            feat = feat.merge(x[keep], on=["sequence_id","marker"], how="outer")
    if feat is None:
        raise SystemExit("No feature files loaded.")

    feature_cols = [c for c in feat.columns if c not in ["sequence_id","marker"]]

    species_frames = []
    hap_frames = []
    feature_qc = []

    for label in args.threshold_labels:
        renorm_fp = Path(args.renorm_threshold_dir) / f"{label}.annotated.renorm.tsv"
        bias_fp = Path(args.bias_dir) / label / "inter_species_bias.tsv"
        if not renorm_fp.exists() or not bias_fp.exists():
            print(f"[WARN] missing inputs for {label}")
            continue

        renorm = pd.read_csv(renorm_fp, sep="\t")
        renorm = compat_cols(renorm)
        renorm["threshold_label"] = label

        # join features to sequence rows
        seq_feat = renorm.merge(feat, on=["sequence_id","marker"], how="left")
        seq_feat.to_csv(outdir / f"{label}.sequence_feature_join.tsv", sep="\t", index=False)

        numeric_feature_cols = []
        for c in feature_cols:
            try:
                pd.to_numeric(seq_feat[c], errors="raise")
                numeric_feature_cols.append(c)
            except Exception:
                pass
        feature_qc.append(pd.DataFrame({"threshold_label":[label], "n_numeric_features":[len(numeric_feature_cols)]}))

        # species-level weighted by renormalized abundance within sample-species
        agg = weighted_feature_aggregate(seq_feat, numeric_feature_cols, "relative_abundance_renorm")
        inter_bias = pd.read_csv(bias_fp, sep="\t")
        inter_bias["threshold_label"] = label
        species_level = inter_bias.merge(
            agg,
            on=["threshold_label","sample_id","marker","group_name","species_label"],
            how="left"
        )
        species_frames.append(species_level)

        # haplotype-level only intra_mix core sequences, with dominant/tail flags
        intra = seq_feat[(seq_feat["sample_type"] == "intra_mix") & (seq_feat["is_core_analysis"].astype(str).str.lower() == "yes")].copy()
        if len(intra):
            intra["dominant_rank"] = intra.groupby(["sample_id"])["relative_abundance_renorm"].rank(method="first", ascending=False)
            intra["dominant_flag"] = (intra["dominant_rank"] == 1).astype(int)
            intra["tail_flag"] = (intra["dominant_rank"] > 1).astype(int)
            hap_frames.append(intra)

    if species_frames:
        pd.concat(species_frames, ignore_index=True).to_csv(outdir / "species_level_feature_bias_table.tsv", sep="\t", index=False)
        print(f"Wrote {outdir / 'species_level_feature_bias_table.tsv'}")
    if hap_frames:
        pd.concat(hap_frames, ignore_index=True).to_csv(outdir / "haplotype_level_feature_bias_table.tsv", sep="\t", index=False)
        print(f"Wrote {outdir / 'haplotype_level_feature_bias_table.tsv'}")
    if feature_qc:
        pd.concat(feature_qc, ignore_index=True).to_csv(outdir / "chapter3_feature_merge_qc.tsv", sep="\t", index=False)
        print(f"Wrote {outdir / 'chapter3_feature_merge_qc.tsv'}")

if __name__ == "__main__":
    main()
