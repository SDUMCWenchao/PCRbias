#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def rmse(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan

def main():
    ap = argparse.ArgumentParser(description="Compute Chapter 2 bias and target/non-target summaries from annotated abundance.")
    ap.add_argument("--annotated-abundance", required=True)
    ap.add_argument("--expected-inter", required=True)
    ap.add_argument("--expected-intra", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ab = pd.read_csv(args.annotated_abundance, sep="\t")
    inter = pd.read_csv(args.expected_inter, sep="\t", dtype={"sample_id":str, "target_label":str})
    intra = pd.read_csv(args.expected_intra, sep="\t", dtype={"sample_id":str, "target_label":str})

    core = ab[ab["is_core_analysis"].astype(str).str.lower() == "yes"].copy()

    # Inter-mix species-level bias
    inter_core = core[core["sample_type"] == "inter_mix"].copy()
    observed = (
        inter_core.groupby(["sample_id","marker","group_name","species_label"], as_index=False)
        .agg(observed_prop=("relative_abundance","sum"))
    )
    expected = inter.rename(columns={"target_label":"species_label"})
    inter_bias = expected.merge(observed, on=["sample_id","marker","group_name","species_label"], how="left")
    inter_bias["observed_prop"] = inter_bias["observed_prop"].fillna(0.0)
    inter_bias["bias_ratio"] = np.where(inter_bias["expected_prop"] > 0, inter_bias["observed_prop"] / inter_bias["expected_prop"], np.nan)
    inter_bias["log2_bias_ratio"] = np.log2(inter_bias["bias_ratio"].replace(0, np.nan))
    inter_bias["abs_deviation"] = (inter_bias["observed_prop"] - inter_bias["expected_prop"]).abs()
    inter_bias["sq_error"] = np.square(inter_bias["observed_prop"] - inter_bias["expected_prop"])
    inter_bias.to_csv(outdir / "inter_species_bias.tsv", sep="\t", index=False)

    inter_overall = (
        inter_bias.groupby(["sample_id","marker","group_name"], as_index=False)
        .agg(
            mean_abs_deviation=("abs_deviation","mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            n_targets=("species_label","nunique"),
        )
    )
    inter_overall.to_csv(outdir / "inter_sample_overall_bias.tsv", sep="\t", index=False)

    # Target vs nontarget summary for inter-mix
    inter_target_set = expected[["sample_id","species_label"]].drop_duplicates().assign(is_expected_target=1)
    inter_obs2 = observed.merge(inter_target_set, on=["sample_id","species_label"], how="left")
    inter_obs2["is_expected_target"] = inter_obs2["is_expected_target"].fillna(0)
    inter_target_summary = (
        inter_obs2.groupby(["sample_id","marker","group_name","is_expected_target"], as_index=False)
        .agg(prop=("observed_prop","sum"))
    )
    inter_target_summary["target_class"] = inter_target_summary["is_expected_target"].map({1:"target",0:"nontarget"})
    inter_target_summary.to_csv(outdir / "inter_target_nontarget_summary.tsv", sep="\t", index=False)

    # Intra-mix target vs nontarget at species-label level
    intra_core = core[core["sample_type"] == "intra_mix"].copy()
    intra_obs = (
        intra_core.groupby(["sample_id","marker","group_name","species_label"], as_index=False)
        .agg(observed_prop=("relative_abundance","sum"))
    )
    intra_expected = intra.rename(columns={"target_label":"species_label"})[["sample_id","marker","group_name","species_label","expected_prop"]].drop_duplicates()
    intra_merge = intra_obs.merge(intra_expected.assign(is_expected_target=1), on=["sample_id","marker","group_name","species_label"], how="left")
    intra_merge["is_expected_target"] = intra_merge["is_expected_target"].fillna(0)
    intra_target_summary = (
        intra_merge.groupby(["sample_id","marker","group_name","is_expected_target"], as_index=False)
        .agg(prop=("observed_prop","sum"))
    )
    intra_target_summary["target_class"] = intra_target_summary["is_expected_target"].map({1:"target",0:"nontarget"})
    intra_target_summary.to_csv(outdir / "intra_target_nontarget_summary.tsv", sep="\t", index=False)

    # Combined sample-level target burden
    combined = pd.concat([
        inter_target_summary.assign(sample_type="inter_mix"),
        intra_target_summary.assign(sample_type="intra_mix"),
    ], ignore_index=True)
    combined.to_csv(outdir / "target_nontarget_sample_level.tsv", sep="\t", index=False)

    print(f"Wrote {outdir / 'inter_species_bias.tsv'}")
    print(f"Wrote {outdir / 'inter_sample_overall_bias.tsv'}")
    print(f"Wrote {outdir / 'target_nontarget_sample_level.tsv'}")

if __name__ == "__main__":
    main()
