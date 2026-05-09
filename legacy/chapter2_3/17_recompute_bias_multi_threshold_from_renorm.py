#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = ["rel_ge_0.001", "rel_ge_0.005", "rel_ge_0.01"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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

def compute_inter_bias(df: pd.DataFrame, expected: pd.DataFrame):
    df = normalize_columns(df)

    required = ["sample_id", "marker", "group_name", "species_label", "relative_abundance_renorm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"inter_df missing required columns: {missing}")

    observed = (
        df.groupby(["sample_id","marker","group_name","species_label"], as_index=False)
        .agg(observed_prop=("relative_abundance_renorm","sum"))
    )

    exp = expected.rename(columns={"target_label":"species_label"})
    merged = exp.merge(observed, on=["sample_id","marker","group_name","species_label"], how="left")
    merged["observed_prop"] = merged["observed_prop"].fillna(0.0)
    merged["bias_ratio"] = np.where(merged["expected_prop"] > 0, merged["observed_prop"] / merged["expected_prop"], np.nan)
    merged["log2_bias_ratio"] = np.log2(merged["bias_ratio"].replace(0, np.nan))
    merged["abs_deviation"] = (merged["observed_prop"] - merged["expected_prop"]).abs()
    merged["sq_error"] = np.square(merged["observed_prop"] - merged["expected_prop"])

    overall = (
        merged.groupby(["sample_id","marker","group_name"], as_index=False)
        .agg(
            mean_abs_deviation=("abs_deviation","mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            n_targets=("species_label","nunique"),
        )
    )

    target_set = exp[["sample_id","species_label"]].drop_duplicates().assign(is_expected_target=1)
    observed2 = observed.merge(target_set, on=["sample_id","species_label"], how="left")
    observed2["is_expected_target"] = observed2["is_expected_target"].fillna(0)

    target_summary = (
        observed2.groupby(["sample_id","marker","group_name","is_expected_target"], as_index=False)
        .agg(prop=("observed_prop","sum"))
    )
    target_summary["target_class"] = target_summary["is_expected_target"].map({1:"target",0:"nontarget"})

    sample_sums = (
        observed2.groupby(["sample_id","marker","group_name"], as_index=False)
        .agg(total_prop_after_renorm=("observed_prop","sum"))
    )

    return merged, overall, target_summary, sample_sums

def compute_intra_target_nontarget(df: pd.DataFrame, expected_intra: pd.DataFrame):
    df = normalize_columns(df)

    required = ["sample_id", "marker", "group_name", "species_label", "relative_abundance_renorm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"intra_df missing required columns: {missing}")

    observed = (
        df.groupby(["sample_id","marker","group_name","species_label"], as_index=False)
        .agg(observed_prop=("relative_abundance_renorm","sum"))
    )

    exp = expected_intra.rename(columns={"target_label":"species_label"})[
        ["sample_id","marker","group_name","species_label","expected_prop"]
    ].drop_duplicates()

    merged = observed.merge(exp.assign(is_expected_target=1), on=["sample_id","marker","group_name","species_label"], how="left")
    merged["is_expected_target"] = merged["is_expected_target"].fillna(0)

    out = (
        merged.groupby(["sample_id","marker","group_name","is_expected_target"], as_index=False)
        .agg(prop=("observed_prop","sum"))
    )
    out["target_class"] = out["is_expected_target"].map({1:"target",0:"nontarget"})

    sample_sums = (
        observed.groupby(["sample_id","marker","group_name"], as_index=False)
        .agg(total_prop_after_renorm=("observed_prop","sum"))
    )

    return out, sample_sums

def main():
    ap = argparse.ArgumentParser(description="Recompute bias/target-nontarget from renormalized threshold tables.")
    ap.add_argument("--renorm-threshold-dir", required=True, help="Directory containing rel_ge_*.annotated.renorm.tsv")
    ap.add_argument("--expected-inter", required=True)
    ap.add_argument("--expected-intra", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold-labels", nargs="*", default=DEFAULT_THRESHOLDS)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    expected_inter = pd.read_csv(args.expected_inter, sep="\t", dtype=str)
    expected_intra = pd.read_csv(args.expected_intra, sep="\t", dtype=str)
    expected_inter["expected_prop"] = expected_inter["expected_prop"].astype(float)
    expected_intra["expected_prop"] = expected_intra["expected_prop"].astype(float)

    overall_frames = []
    check_frames = []

    for label in args.threshold_labels:
        fp = Path(args.renorm_threshold_dir) / f"{label}.annotated.renorm.tsv"
        if not fp.exists():
            print(f"[WARN] missing {fp}")
            continue

        tdir = outdir / label
        tdir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(fp, sep="\t")
        df = normalize_columns(df)

        inter_df = df[df["sample_type"] == "inter_mix"].copy()
        intra_df = df[df["sample_type"] == "intra_mix"].copy()

        inter_bias, inter_overall, inter_tn, inter_sums = compute_inter_bias(inter_df, expected_inter)
        intra_tn, intra_sums = compute_intra_target_nontarget(intra_df, expected_intra)

        inter_bias["threshold_label"] = label
        inter_overall["threshold_label"] = label
        inter_tn["threshold_label"] = label
        intra_tn["threshold_label"] = label
        inter_sums["threshold_label"] = label
        intra_sums["threshold_label"] = label

        inter_bias.to_csv(tdir / "inter_species_bias.tsv", sep="\t", index=False)
        inter_overall.to_csv(tdir / "inter_sample_overall_bias.tsv", sep="\t", index=False)
        inter_tn.to_csv(tdir / "inter_target_nontarget_summary.tsv", sep="\t", index=False)
        intra_tn.to_csv(tdir / "intra_target_nontarget_summary.tsv", sep="\t", index=False)

        combined = pd.concat(
            [inter_tn.assign(sample_type="inter_mix"), intra_tn.assign(sample_type="intra_mix")],
            ignore_index=True
        )
        combined.to_csv(tdir / "target_nontarget_sample_level.tsv", sep="\t", index=False)

        # sanity checks: after renorm, observed species sums should approach 1 within each sample
        checks = pd.concat([
            inter_sums.assign(scope="inter_mix"),
            intra_sums.assign(scope="intra_mix")
        ], ignore_index=True)
        checks.to_csv(tdir / "sample_total_prop_checks.tsv", sep="\t", index=False)

        overall_frames.append(inter_overall)
        check_frames.append(checks)

        print(f"Wrote threshold outputs: {tdir}")

    if overall_frames:
        pd.concat(overall_frames, ignore_index=True).to_csv(outdir / "all_thresholds_inter_sample_overall_bias.tsv", sep="\t", index=False)
        print(f"Wrote {outdir / 'all_thresholds_inter_sample_overall_bias.tsv'}")

    if check_frames:
        pd.concat(check_frames, ignore_index=True).to_csv(outdir / "all_thresholds_sample_total_prop_checks.tsv", sep="\t", index=False)
        print(f"Wrote {outdir / 'all_thresholds_sample_total_prop_checks.tsv'}")

if __name__ == "__main__":
    main()
