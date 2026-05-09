#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, gzip, json, re
from pathlib import Path
import pandas as pd

def read_tsv_any(p: Path):
    if p.suffix == ".gz":
        return pd.read_csv(p, sep="\t", compression="gzip")
    return pd.read_csv(p, sep="\t")

def maybe_read_tsv_any(p: Path):
    if not p.exists():
        return None
    try:
        return read_tsv_any(p)
    except Exception:
        return None

def parse_tag_level(tag: str):
    # top1p, top0p5p, top0p1p
    m = re.match(r"top([0-9]+)p([0-9]+)?p?$", tag)
    if tag == "top1p":
        return 0.01
    if tag == "top0p5p":
        return 0.005
    if tag == "top0p1p":
        return 0.001
    return None

def parse_variant_k(variant: str):
    m = re.search(r"_k([0-9]+)$", variant)
    return int(m.group(1)) if m else None

def feature_group(name: str):
    # 仅用于后续画图分层：kmer / primer / other
    if name.startswith("kmer_") or name.startswith("km_") or name.startswith("feat_kmer_"):
        return "kmer"
    if name.startswith("feat_pr_") or name.startswith("primer_"):
        return "primer"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=".")
    ap.add_argument("--models_root", default="analysis_results/06_Models_v3_topbias")
    ap.add_argument("--ci_tsv", default="analysis_results/06_Models_v3_topbias/_bootstrap_ci/test_ci_v2.tsv")
    ap.add_argument("--out_dir", default="analysis_results/07_PlotTables_v3_topbias")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing tables")
    args = ap.parse_args()

    proj = Path(args.project_dir).resolve()
    models_root = (proj / args.models_root).resolve()
    ci_path = (proj / args.ci_tsv).resolve()
    out_dir = (proj / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ci_path.exists():
        raise FileNotFoundError(f"missing CI table: {ci_path}")

    ci = pd.read_csv(ci_path, sep="\t")
    # 基础主表（后面画图的“唯一真相”建议直接用它）
    ci["topbias_frac"] = ci["tag"].map(parse_tag_level)
    ci["k"] = ci["variant"].map(parse_variant_k)
    ci.to_csv(out_dir / "01_master_metrics_ci.tsv", sep="\t", index=False)

    # best by group（每个 tag/species/locus/model 选一个 best variant）
    score_col = "test_spearman" if "test_spearman" in ci.columns else "test_r2"
    tmp = ci.copy()
    # tie-breaker：spearman优先，其次rmse更小，其次r2更大
    tmp["_rmse"] = pd.to_numeric(tmp["test_rmse"], errors="coerce")
    tmp["_r2"]   = pd.to_numeric(tmp["test_r2"], errors="coerce")
    tmp["_score"]= pd.to_numeric(tmp[score_col], errors="coerce")

    tmp = tmp.sort_values(
        by=["tag","species","locus","model","_score","_rmse","_r2"],
        ascending=[True,True,True,True,False,True,False],
        kind="mergesort"
    )
    best = tmp.groupby(["tag","species","locus","model"], as_index=False).head(1).drop(columns=["_rmse","_r2","_score"])
    best.to_csv(out_dir / "02_best_by_group.tsv", sep="\t", index=False)

    # variant effects（相对 no_kmer 的变化；同组内对比）
    base = ci[ci["variant"]=="no_kmer"][["tag","species","locus","model","test_r2","test_rmse","test_spearman","test_mae"]].copy()
    base = base.rename(columns={
        "test_r2":"base_r2","test_rmse":"base_rmse","test_spearman":"base_spearman","test_mae":"base_mae"
    })
    eff = ci.merge(base, on=["tag","species","locus","model"], how="left")
    for a,b in [("test_r2","base_r2"),("test_spearman","base_spearman"),("test_rmse","base_rmse"),("test_mae","base_mae")]:
        eff[f"delta_{a}"] = pd.to_numeric(eff[a], errors="coerce") - pd.to_numeric(eff[b], errors="coerce")
    eff.to_csv(out_dir / "03_variant_effects_vs_nokmer.tsv", sep="\t", index=False)

    # 采集 SHAP v2：global + region（rf/xgb）
    shap_global_rows = []
    shap_region_rows = []
    fi_rows = []

    # 采集 seqcnn IG lenaware：region + pos_from_end
    ig_region_rows = []
    ig_pos_end_rows = []

    def add_model_tables(r):
        tag, sp, lc, model, variant = r["tag"], r["species"], r["locus"], r["model"], r["variant"]
        mdir = models_root / tag / sp / lc / model / variant

        # feature_importance（RF/XGB都有时就收集）
        fi = mdir / "feature_importance.tsv"
        dfi = maybe_read_tsv_any(fi)
        if dfi is not None and "feature" in dfi.columns:
            dfi["tag"]=tag; dfi["species"]=sp; dfi["locus"]=lc; dfi["model"]=model; dfi["variant"]=variant
            dfi["topbias_frac"]=parse_tag_level(tag)
            dfi["k"]=parse_variant_k(variant)
            dfi["feature_group"]=dfi["feature"].map(feature_group)
            fi_rows.append(dfi)

        # SHAP v2
        if model in ("rf","xgb"):
            sg = mdir / "shap_tables_v2" / "shap_global_full_test.tsv.gz"
            sr = mdir / "shap_tables_v2" / "shap_region_full_test.tsv"
            dsg = maybe_read_tsv_any(sg)
            dsr = maybe_read_tsv_any(sr)
            if dsg is not None:
                dsg["tag"]=tag; dsg["species"]=sp; dsg["locus"]=lc; dsg["model"]=model; dsg["variant"]=variant
                dsg["topbias_frac"]=parse_tag_level(tag)
                dsg["k"]=parse_variant_k(variant)
                dsg["feature_group"]=dsg["feature"].map(feature_group)
                shap_global_rows.append(dsg)
            if dsr is not None:
                dsr["tag"]=tag; dsr["species"]=sp; dsr["locus"]=lc; dsr["model"]=model; dsr["variant"]=variant
                dsr["topbias_frac"]=parse_tag_level(tag)
                dsr["k"]=parse_variant_k(variant)
                shap_region_rows.append(dsr)

        # seqcnn IG lenaware
        if model == "seqcnn":
            idir = mdir / "attr_tables_lenaware"
            ir = idir / "attr_region_test.tsv"
            ip = idir / "attr_pos_from_end_test.tsv.gz"
            dir_ = maybe_read_tsv_any(ir)
            dip = maybe_read_tsv_any(ip)
            if dir_ is not None:
                dir_["tag"]=tag; dir_["species"]=sp; dir_["locus"]=lc; dir_["model"]=model; dir_["variant"]=variant
                dir_["topbias_frac"]=parse_tag_level(tag)
                dir_["k"]=parse_variant_k(variant)
                ig_region_rows.append(dir_)
            if dip is not None:
                dip["tag"]=tag; dip["species"]=sp; dip["locus"]=lc; dip["model"]=model; dip["variant"]=variant
                dip["topbias_frac"]=parse_tag_level(tag)
                dip["k"]=parse_variant_k(variant)
                ig_pos_end_rows.append(dip)

    # 全量扫描（不做小批量试跑，不跳过）
    for _, r in ci.iterrows():
        add_model_tables(r)

    def write_cat(rows, outname):
        if not rows:
            return
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(out_dir / outname, sep="\t", index=False)

    write_cat(shap_global_rows, "04_shap_global_full_long.tsv")
    write_cat(shap_region_rows, "05_shap_region_full_long.tsv")
    write_cat(ig_region_rows,   "06_ig_region_long.tsv")
    write_cat(ig_pos_end_rows,  "07_ig_pos_from_end_long.tsv")
    write_cat(fi_rows,          "08_feature_importance_long.tsv")

    print("[DONE] tables ->", out_dir)
    print(" - 01_master_metrics_ci.tsv")
    print(" - 02_best_by_group.tsv")
    print(" - 03_variant_effects_vs_nokmer.tsv")
    print(" - 04_shap_global_full_long.tsv")
    print(" - 05_shap_region_full_long.tsv")
    print(" - 06_ig_region_long.tsv")
    print(" - 07_ig_pos_from_end_long.tsv")
    print(" - 08_feature_importance_long.tsv")

if __name__ == "__main__":
    main()
