#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

DEFAULT_THRESHOLD_LABELS = ["rel_ge_0.001", "rel_ge_0.005", "rel_ge_0.01"]

def renorm_one(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 兼容旧表头
    if "marker" not in out.columns:
        if "marker_x" in out.columns:
            out["marker"] = out["marker_x"]
        elif "marker_y" in out.columns:
            out["marker"] = out["marker_y"]

    if "group_name" not in out.columns:
        if "group_name_x" in out.columns:
            out["group_name"] = out["group_name_x"]
        elif "group_name_y" in out.columns:
            out["group_name"] = out["group_name_y"]

    required = ["sample_id", "count"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0)
    out["retained_total_count"] = out.groupby("sample_id")["count"].transform("sum")
    out["relative_abundance_renorm"] = out["count"] / out["retained_total_count"].replace(0, pd.NA)
    out["relative_abundance_renorm"] = out["relative_abundance_renorm"].fillna(0.0)

    # 额外输出一个“该样本在当前阈值下保留了原始多少信号”的指标
    if "relative_abundance" in out.columns:
        out["retained_signal_fraction_from_original"] = (
            out.groupby("sample_id")["relative_abundance"].transform("sum")
        )
    else:
        out["retained_signal_fraction_from_original"] = pd.NA

    return out

def sample_checks(df: pd.DataFrame, threshold_label: str) -> pd.DataFrame:
    rows = []
    for sample_id, sub in df.groupby("sample_id", dropna=False):
        rows.append({
            "threshold_label": threshold_label,
            "sample_id": sample_id,
            "n_rows": len(sub),
            "retained_total_count": int(sub["count"].sum()),
            "renorm_sum": float(sub["relative_abundance_renorm"].sum()),
            "retained_signal_fraction_from_original": float(sub["retained_signal_fraction_from_original"].iloc[0]) if "retained_signal_fraction_from_original" in sub.columns and pd.notna(sub["retained_signal_fraction_from_original"].iloc[0]) else None,
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Renormalize threshold-filtered annotated tables within each sample.")
    ap.add_argument("--annotated-threshold-dir", required=True, help="Directory containing rel_ge_*.annotated.tsv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold-labels", nargs="*", default=DEFAULT_THRESHOLD_LABELS)
    args = ap.parse_args()

    in_dir = Path(args.annotated_threshold_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    checks = []
    for label in args.threshold_labels:
        fp = in_dir / f"{label}.annotated.tsv"
        if not fp.exists():
            print(f"[WARN] missing {fp}")
            continue

        df = pd.read_csv(fp, sep="\t")
        out = renorm_one(df)
        out.to_csv(outdir / f"{label}.annotated.renorm.tsv", sep="\t", index=False)
        checks.append(sample_checks(out, label))
        print(f"Wrote {outdir / f'{label}.annotated.renorm.tsv'}")

    if checks:
        pd.concat(checks, ignore_index=True).to_csv(outdir / "renorm_sample_checks.tsv", sep="\t", index=False)
        print(f"Wrote {outdir / 'renorm_sample_checks.tsv'}")

if __name__ == "__main__":
    main()
