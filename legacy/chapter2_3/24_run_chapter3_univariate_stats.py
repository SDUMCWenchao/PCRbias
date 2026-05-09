#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

CORE_GROUPS = {
    "mismatch": ["mismatch_fwd_total","mismatch_rev_total","mismatch_total","mismatch_weighted_score","mismatch_3prime_total"],
    "global": ["len_global","gc_global","at_skew_global","gc_skew_global","entropy_global","lz_global","homopolymer_max_global"],
    "regional": [],
    "rnafold": ["mfe_global","mfe_head","mfe_mid1","mfe_mid2","mfe_mid3","mfe_tail"],
    "motif": [],
}

def build_feature_groups(cols):
    groups = {k:list(v) for k,v in CORE_GROUPS.items()}
    groups["regional"] = [c for c in cols if any(c.startswith(prefix) for prefix in ["gc_head","gc_mid","gc_tail","entropy_head","entropy_mid","entropy_tail","lz_head","lz_mid","lz_tail","homopolymer_max_head","homopolymer_max_mid","homopolymer_max_tail"])]
    groups["motif"] = [c for c in cols if c.startswith("motif_")]
    return groups

def std_beta(y, x):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = ~(x.isna() | y.isna())
    if mask.sum() < 6:
        return np.nan, np.nan, np.nan
    x = x[mask]
    y = y[mask]
    xz = (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1)
    yz = (y - y.mean()) / (y.std(ddof=0) if y.std(ddof=0) != 0 else 1)
    X = sm.add_constant(xz)
    model = sm.OLS(yz, X).fit()
    return model.params.iloc[1], model.pvalues.iloc[1], model.rsquared

def run_one(df, response, strata_name):
    rows = []
    feat_cols = [c for c in df.columns if c not in [
        "threshold_label","sample_id","marker","group_name","species_label",
        "expected_prop","observed_prop","bias_ratio","log2_bias_ratio","abs_deviation","sq_error",
        "n_sequences_in_species","species_total_count_retained",
        "sequence_id","sample_type","species_scope","is_core_analysis","count","relative_abundance","retained_total_count",
        "relative_abundance_renorm","retained_signal_fraction_from_original","annotation_status","best_hit_accession","best_hit_description",
        "best_hit_species","pident","qcovs","notes","dominant_rank","dominant_flag","tail_flag"
    ]]
    for feat in feat_cols:
        x = pd.to_numeric(df[feat], errors="coerce")
        y = pd.to_numeric(df[response], errors="coerce")
        mask = ~(x.isna() | y.isna())
        if mask.sum() < 6:
            continue
        rho, p = spearmanr(x[mask], y[mask])
        beta, p_beta, r2 = std_beta(y, x)
        rows.append({
            "strata": strata_name,
            "response": response,
            "feature": feat,
            "n": int(mask.sum()),
            "spearman_rho": rho,
            "spearman_p": p,
            "std_beta": beta,
            "beta_p": p_beta,
            "r2": r2,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["spearman_fdr"] = multipletests(out["spearman_p"].fillna(1.0), method="fdr_bh")[1]
        out["beta_fdr"] = multipletests(out["beta_p"].fillna(1.0), method="fdr_bh")[1]
    return out

def main():
    ap = argparse.ArgumentParser(description="Run Chapter 3 univariate biological statistics.")
    ap.add_argument("--species-table", required=True)
    ap.add_argument("--haplotype-table", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv(args.species_table, sep="\t")
    hp = pd.read_csv(args.haplotype_table, sep="\t")

    species_results = []
    for threshold in sorted(sp["threshold_label"].dropna().unique()):
        for marker in sorted(sp["marker"].dropna().unique()):
            sub = sp[(sp["threshold_label"] == threshold) & (sp["marker"] == marker)].copy()
            if len(sub) < 6:
                continue
            species_results.append(run_one(sub, "log2_bias_ratio", f"species|{threshold}|{marker}|log2_bias_ratio"))
            species_results.append(run_one(sub, "abs_deviation", f"species|{threshold}|{marker}|abs_deviation"))

    if species_results:
        species_stats = pd.concat(species_results, ignore_index=True)
        species_stats.to_csv(outdir / "chapter3_species_univariate_stats.tsv", sep="\t", index=False)
        print(f"Wrote {outdir / 'chapter3_species_univariate_stats.tsv'}")
    else:
        species_stats = pd.DataFrame()

    hap_results = []
    if len(hp):
        # use abundance distortion proxy at haplotype level
        hp["tail_vs_dom_proxy"] = np.where(hp["dominant_flag"] == 1, hp["relative_abundance_renorm"], -hp["relative_abundance_renorm"])
        for threshold in sorted(hp["threshold_label"].dropna().unique()):
            for marker in sorted(hp["marker"].dropna().unique()):
                sub = hp[(hp["threshold_label"] == threshold) & (hp["marker"] == marker)].copy()
                if len(sub) < 6:
                    continue
                hap_results.append(run_one(sub, "relative_abundance_renorm", f"haplotype|{threshold}|{marker}|relative_abundance_renorm"))
                hap_results.append(run_one(sub, "tail_vs_dom_proxy", f"haplotype|{threshold}|{marker}|tail_vs_dom_proxy"))

    if hap_results:
        hap_stats = pd.concat(hap_results, ignore_index=True)
        hap_stats.to_csv(outdir / "chapter3_haplotype_univariate_stats.tsv", sep="\t", index=False)
        print(f"Wrote {outdir / 'chapter3_haplotype_univariate_stats.tsv'}")
    else:
        hap_stats = pd.DataFrame()

    # concise top summary
    summary_rows = []
    for name, df in [("species", species_stats), ("haplotype", hap_stats)]:
        if len(df) == 0:
            continue
        feat_groups = build_feature_groups(df["feature"].tolist())
        for group_name, cols in feat_groups.items():
            sub = df[df["feature"].isin(cols)].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values(["beta_fdr","spearman_fdr", "std_beta"], ascending=[True, True, False])
            best = sub.head(20)
            tmp = best[["strata","feature","spearman_rho","spearman_fdr","std_beta","beta_fdr","r2"]].copy()
            tmp["data_level"] = name
            tmp["feature_group"] = group_name
            summary_rows.append(tmp)

    if summary_rows:
        pd.concat(summary_rows, ignore_index=True).to_csv(outdir / "chapter3_top_signal_summary.tsv", sep="\t", index=False)
        print(f"Wrote {outdir / 'chapter3_top_signal_summary.tsv'}")

if __name__ == "__main__":
    main()
