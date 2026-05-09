#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import gzip
import re
import tarfile
from pathlib import Path
from collections import defaultdict

TAG_SET = {"top1p", "top0p5p", "top0p1p"}
SPECIES_SET = {"donkey", "pig", "cattle", "10mix"}
LOCUS_SET = {"12S", "16S"}
MODEL_SET = {"rf", "xgb", "seqcnn", "cnn1d"}

PREFERRED_MASTER_METRICS = [
    "01_master_metrics_ci_strat_by_log2fc.tsv",
    "master_metrics_ci_strat_by_log2fc.tsv",
    "01_master_metrics_ci.tsv",
    "master_metrics_ci.tsv",
]
PREFERRED_TEST_CI = [
    "test_ci_strat_by_log2fc.tsv",
    "test_ci.tsv",
]

def open_text(p: Path):
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", newline="")
    return open(p, "rt", encoding="utf-8", newline="")

def write_tsv(path: Path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())

def find_first_existing(dir_: Path, names):
    for n in names:
        p = dir_ / n
        if p.exists():
            return p
    return None

def parse_context_from_path(p: Path):
    parts = list(p.parts)
    tag = next((x for x in parts if x in TAG_SET), "")
    species = next((x for x in parts if x in SPECIES_SET), "")
    locus = next((x for x in parts if x in LOCUS_SET), "")
    model = next((x for x in parts if x in MODEL_SET), "")
    variant = ""
    if model:
        try:
            i = parts.index(model)
            if i + 1 < len(parts):
                variant = parts[i + 1]
        except ValueError:
            pass
    return tag, species, locus, model, variant

def infer_feature_cols(header):
    # very tolerant: pick best-guess column names
    def pick(cands):
        for c in cands:
            if c in header:
                return c
        # case-insensitive fallback
        low = {h.lower(): h for h in header}
        for c in cands:
            if c.lower() in low:
                return low[c.lower()]
        return ""

    feat = pick(["feature", "feat", "name", "feature_name"])
    mean_abs = pick(["mean_abs", "mean_abs_shap", "abs_mean", "mean_abs_value", "mean(|shap|)", "mean_abs_shap_value"])
    mean = pick(["mean", "mean_shap", "mean_value", "avg_shap"])
    return feat, mean_abs, mean

def parse_region_k(feature: str):
    # region: head/tail/mid0..mid9 anywhere
    mreg = re.search(r"(head|tail|mid[0-9]+)", feature)
    region = mreg.group(1) if mreg else ""
    # k: try "k7" or "_k7_" etc
    mk = re.search(r"(?:^|[^0-9a-zA-Z])k([1-8])(?:[^0-9a-zA-Z]|$)", feature)
    k = mk.group(1) if mk else ""
    # primer flag
    is_primer = 1 if (feature.startswith("feat_pr_") or "feat_pr_" in feature) else 0
    return region, k, is_primer

def load_tsv_rows(path: Path):
    with open_text(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        rows = list(r)
        return rows, r.fieldnames or []

def topn_rows(rows, key, n):
    if n is None or n <= 0:
        return rows
    def fval(x):
        try:
            return float(x.get(key, "nan"))
        except Exception:
            return float("-inf")
    rows2 = sorted(rows, key=fval, reverse=True)
    return rows2[:n]

def scan_shap_tables(models_root: Path):
    # find shap_tables dirs and pick a "global" summary file
    candidates = []
    for d in models_root.rglob("shap_tables"):
        if not d.is_dir():
            continue
        # prefer these names
        preferred = []
        for nm in ["global_mean_abs.tsv", "shap_global_mean_abs.tsv", "global_summary.tsv", "global.tsv"]:
            p = d / nm
            if p.exists():
                preferred.append(p)
        if preferred:
            candidates.append(preferred[0])
            continue
        # otherwise any tsv with "global" or "mean_abs"
        ts = sorted(d.glob("*.tsv"))
        pick = None
        for p in ts:
            if "global" in p.name or "mean_abs" in p.name:
                pick = p
                break
        if pick:
            candidates.append(pick)
    return sorted(set(candidates))

def scan_ig_tables(models_root: Path):
    out = []
    for d in models_root.rglob("attr_tables"):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.tsv")):
            if "attr_" in p.name:
                out.append(p)
    return out

def build_perf_slim(master_metrics_path: Path):
    rows, header = load_tsv_rows(master_metrics_path)
    if not rows:
        return [], []

    # choose columns we care about, but keep robust fallback
    # try to detect common names
    def has(col):
        return col in header

    base_cols = []
    for c in ["tag", "species", "locus", "model", "variant", "split", "n", "n_test", "n_val", "n_train"]:
        if has(c):
            base_cols.append(c)

    metric_cols = []
    for c in header:
        lc = c.lower()
        if any(x in lc for x in ["spearman", "rmse", "mae", "sign", "auc", "r2"]):
            metric_cols.append(c)
        if any(x in lc for x in ["ci_", "ci", "p2p5", "p97p5", "low", "high"]) and c not in metric_cols:
            metric_cols.append(c)

    cols = base_cols + [c for c in metric_cols if c not in base_cols]
    # if we failed to detect, just keep all
    if not cols:
        cols = header

    slim = [{k: r.get(k, "") for k in cols} for r in rows]
    return slim, cols

def build_best_by_group(master_metrics_path: Path, out_path: Path):
    rows, header = load_tsv_rows(master_metrics_path)
    if not rows:
        return

    # find a "test spearman" column
    # common patterns: spearman_test, test_spearman, spearman (with split=test)
    spearman_col = ""
    if "spearman_test" in header:
        spearman_col = "spearman_test"
    elif "test_spearman" in header:
        spearman_col = "test_spearman"
    elif "spearman" in header and "split" in header:
        spearman_col = "spearman"
    else:
        # fallback search
        for c in header:
            if "spearman" in c.lower() and "test" in c.lower():
                spearman_col = c
                break

    key_cols = [c for c in ["tag", "species", "locus", "model", "variant", "split"] if c in header]
    if not spearman_col or not {"tag", "species", "locus", "model", "variant"}.issubset(set(header)):
        return

    best = {}
    for r in rows:
        if "split" in header and r.get("split", "") not in ("test", ""):
            continue
        k = (r.get("tag",""), r.get("species",""), r.get("locus",""), r.get("model",""))
        try:
            v = float(r.get(spearman_col, "nan"))
        except Exception:
            continue
        if k not in best or v > best[k][0]:
            best[k] = (v, r)

    out_rows = []
    for (tag, sp, loc, model), (v, r) in sorted(best.items()):
        out = {
            "tag": tag, "species": sp, "locus": loc, "model": model,
            "best_variant": r.get("variant",""),
            "best_spearman_test": r.get(spearman_col, ""),
        }
        # carry a few more common metrics if exist
        for c in ["rmse_test", "mae_test", "sign_acc_test", "signacc_test", "n_test"]:
            if c in header:
                out[c] = r.get(c, "")
        out_rows.append(out)

    cols = list(out_rows[0].keys()) if out_rows else []
    if out_rows:
        write_tsv(out_path, out_rows, cols)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Export minimal tables for writing thesis Results (v3_topbias).")
    ap.add_argument("--project_dir", default=".", help="Project root (default: .)")
    ap.add_argument("--models_root", default=None, help="Default: <project>/analysis_results/06_Models_v3_topbias")
    ap.add_argument("--plot_tables_dir", default=None, help="Default: <project>/analysis_results/07_PlotTables_v3_topbias")
    ap.add_argument("--out_dir", default=None, help="Default: <project>/analysis_results/08_ResultsPack_v3_topbias")
    ap.add_argument("--shap_top", type=int, default=500, help="Keep top N SHAP features per model-run (default 500). Use -1 to keep all.")
    ap.add_argument("--make_tar", action="store_true", help="Also create a tar.gz bundle")
    args = ap.parse_args()

    project = Path(args.project_dir).resolve()
    models_root = Path(args.models_root).resolve() if args.models_root else (project / "analysis_results/06_Models_v3_topbias")
    plot_tables = Path(args.plot_tables_dir).resolve() if args.plot_tables_dir else (project / "analysis_results/07_PlotTables_v3_topbias")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (project / "analysis_results/08_ResultsPack_v3_topbias")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) copy meta/core overview tables if exist
    core_names = [
        "pair_manifest.tsv",
        "true_counts_by_group.tsv",
        "true_counts_all_variants.tsv",
        "dataset_overview.tsv",
        "A1_group_overview.tsv",
        "A3A4_y_hist.tsv",
        "A3A4_y_quantiles_and_thresholds.tsv",
        "F5_permutation_null.tsv.gz",
        "F5_permutation_null.tsv",
    ]
    copied = []
    for nm in core_names:
        p = plot_tables / nm
        if p.exists():
            dst = out_dir / "meta" / nm
            copy_file(p, dst)
            copied.append(dst)

    # 1) performance tables (master metrics + test CI)
    mm = find_first_existing(plot_tables, PREFERRED_MASTER_METRICS)
    if mm and mm.exists():
        copy_file(mm, out_dir / "perf" / mm.name)
        slim, cols = build_perf_slim(mm)
        if slim:
            write_tsv(out_dir / "perf" / "master_metrics_slim.tsv", slim, cols)
        build_best_by_group(mm, out_dir / "perf" / "best_variant_by_group.tsv")

    tci = find_first_existing(plot_tables, PREFERRED_TEST_CI)
    if tci and tci.exists():
        copy_file(tci, out_dir / "perf" / tci.name)

    # 2) SHAP (RF/XGB) global top features + region/k budgets
    shap_files = scan_shap_tables(models_root)
    shap_long = []
    region_budget = []
    k_budget = []
    primer_budget = []

    for sf in shap_files:
        tag, sp, loc, model, var = parse_context_from_path(sf)
        if not (tag and sp and loc and model and var):
            continue
        rows, header = load_tsv_rows(sf)
        if not rows:
            continue
        feat_col, mean_abs_col, mean_col = infer_feature_cols(header)
        if not feat_col or not mean_abs_col:
            continue

        # top N per run
        keep_n = None if args.shap_top is None or args.shap_top < 0 else args.shap_top
        rows = topn_rows(rows, mean_abs_col, keep_n)

        # budgets
        reg_sum = defaultdict(float)
        k_sum = defaultdict(float)
        pr_sum = defaultdict(float)

        for r in rows:
            feat = r.get(feat_col, "")
            mabs = r.get(mean_abs_col, "")
            m = r.get(mean_col, "") if mean_col else ""
            region, k, is_primer = parse_region_k(feat)
            try:
                mabs_f = float(mabs)
            except Exception:
                continue
            reg_sum[region or "NA"] += mabs_f
            k_sum[k or "NA"] += mabs_f
            pr_sum["primer" if is_primer else "nonprimer"] += mabs_f

            shap_long.append({
                "tag": tag, "species": sp, "locus": loc,
                "model": model, "variant": var,
                "feature": feat,
                "region": region or "",
                "k": k or "",
                "is_primer": str(is_primer),
                "mean_abs": str(mabs_f),
                "mean": m if m != "" else "",
                "src": str(sf),
            })

        # normalize budgets (fractions)
        def add_budget(dest, dim_name, dsum):
            tot = sum(dsum.values()) or 1.0
            for key, val in sorted(dsum.items(), key=lambda x: -x[1]):
                dest.append({
                    "tag": tag, "species": sp, "locus": loc,
                    "model": model, "variant": var,
                    dim_name: key,
                    "sum_mean_abs": f"{val:.8g}",
                    "frac": f"{val/tot:.8g}",
                })

        add_budget(region_budget, "region", reg_sum)
        add_budget(k_budget, "k", k_sum)
        add_budget(primer_budget, "class", pr_sum)

    if shap_long:
        write_tsv(out_dir / "shap" / "shap_global_top_long.tsv", shap_long, list(shap_long[0].keys()))
    if region_budget:
        write_tsv(out_dir / "shap" / "shap_region_budget.tsv", region_budget, list(region_budget[0].keys()))
    if k_budget:
        write_tsv(out_dir / "shap" / "shap_k_budget.tsv", k_budget, list(k_budget[0].keys()))
    if primer_budget:
        write_tsv(out_dir / "shap" / "shap_primer_budget.tsv", primer_budget, list(primer_budget[0].keys()))

    # 3) seqCNN IG attr tables (combine)
    ig_files = scan_ig_tables(models_root)
    ig_region_long = []
    ig_pos_long = []
    ig_other = []

    for p in ig_files:
        tag, sp, loc, model, var = parse_context_from_path(p)
        if model != "seqcnn":
            continue
        rows, header = load_tsv_rows(p)
        if not rows:
            continue
        # attach metadata to every row, keep original columns too
        for r in rows:
            r2 = dict(r)
            r2.update({"tag": tag, "species": sp, "locus": loc, "variant": var, "src": str(p)})
            if "region" in header or "Region" in header or "seg" in " ".join(header).lower():
                ig_region_long.append(r2)
            elif any("pos" in h.lower() for h in header) or "from_end" in p.name:
                ig_pos_long.append(r2)
            else:
                ig_other.append(r2)

    if ig_region_long:
        write_tsv(out_dir / "ig" / "ig_region_long.tsv", ig_region_long, list(ig_region_long[0].keys()))
    if ig_pos_long:
        write_tsv(out_dir / "ig" / "ig_pos_long.tsv", ig_pos_long, list(ig_pos_long[0].keys()))
    if ig_other:
        write_tsv(out_dir / "ig" / "ig_other_long.tsv", ig_other, list(ig_other[0].keys()))

    # 4) make tar.gz
    if args.make_tar:
        tar_path = out_dir.parent / f"{out_dir.name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(out_dir, arcname=out_dir.name)
        print(f"[DONE] bundle -> {tar_path}")

    print(f"[DONE] exported -> {out_dir}")
    print(f"[INFO] copied meta={len(copied)} shap_files={len(shap_files)} ig_files={len(ig_files)}")

if __name__ == "__main__":
    main()
