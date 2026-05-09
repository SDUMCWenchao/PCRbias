#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_tsv, sep="\t")

    # overwrite old degenerate CI columns with new stratified CI columns
    # (keep both; but make old names usable for downstream scripts)
    if "spearman_ci_low" in df.columns:
        df["test_spearman_ci_low"]  = df["spearman_ci_low"]
        df["test_spearman_ci_high"] = df["spearman_ci_high"]
        df["test_spearman_boot_sd"] = df["spearman_boot_sd"]

    if "rmse_ci_low" in df.columns:
        df["test_rmse_ci_low"]  = df["rmse_ci_low"]
        df["test_rmse_ci_high"] = df["rmse_ci_high"]
        df["test_rmse_boot_sd"] = df["rmse_boot_sd"]

    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"[DONE] wrote compat master -> {args.out_tsv}")

if __name__ == "__main__":
    main()
