#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05a_plot_joint_reports.py

Parse markdown joint ranking report (produced by 04h_joint_rank_stratify.py)
and generate:
- per-group TSV table
- topN bar plot by joint_stable
- scatter plot: mean_abs_corr vs mean_abs_delta (size ~ joint_strength)
- feature type counts summary plot

Fixes:
- Correctly parse table rows (previous versions often inverted the condition)
- Robust to CRLF (\r\n)
- Robust to occasional non-table lines inside table blocks
"""

import argparse
import re
import math
from pathlib import Path
import pandas as pd

# matplotlib only (no seaborn)
import matplotlib.pyplot as plt


RE_GROUP = re.compile(r"^##\s+Group:\s*(.+?)\s*$")
RE_HEADER = re.compile(
    r"^\|rank\|feature\|joint_stable\|mean_abs_corr\|mean_abs_delta\|median_delta\|frac_pos_corr\|frac_pos_delta\|\s*$"
)
RE_SEPARATOR = re.compile(r"^\|\s*-+.*$")


def slugify(s: str) -> str:
    s = s.strip().replace("=", "_").replace(":", "_").replace("/", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:160]


def feature_type(name: str) -> str:
    # explicit kmer features like k7_tail30_AGGCCAT
    if re.match(r"^k\d+_", name):
        return "kmer"
    if name.startswith("feat_pr_"):
        return "primer"
    if name.startswith("feat_"):
        return "seq_global"
    return "other"


def parse_report_md(md_path: Path) -> dict[str, pd.DataFrame]:
    """
    Return {group_name: DataFrame} for all tables in report.md
    """
    groups: dict[str, list[dict]] = {}
    cur_group = None
    in_table = False

    def ensure_group(g):
        if g not in groups:
            groups[g] = []

    with md_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")

            m = RE_GROUP.match(line)
            if m:
                cur_group = m.group(1).strip()
                in_table = False
                ensure_group(cur_group)
                continue

            if cur_group is None:
                continue

            if RE_HEADER.match(line):
                in_table = True
                continue

            if not in_table:
                continue

            # skip separator lines like |---:|---|...
            if RE_SEPARATOR.match(line):
                continue

            # Accept only lines starting with '|', but don't terminate the table on non-'|' lines;
            # simply ignore them (robust to occasional "..." lines).
            if not line.startswith("|"):
                continue

            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) != 8:
                continue
            if not parts[0].isdigit():
                continue

            groups[cur_group].append(
                dict(
                    rank=int(parts[0]),
                    feature=parts[1],
                    joint_stable=float(parts[2]) if parts[2] else math.nan,
                    mean_abs_corr=float(parts[3]) if parts[3] else math.nan,
                    mean_abs_delta=float(parts[4]) if parts[4] else math.nan,
                    median_delta=float(parts[5]) if parts[5] else math.nan,
                    frac_pos_corr=float(parts[6]) if parts[6] else math.nan,
                    frac_pos_delta=float(parts[7]) if parts[7] else math.nan,
                )
            )

    out: dict[str, pd.DataFrame] = {}
    for g, rows in groups.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
        df["joint_strength"] = df["mean_abs_corr"].abs() * df["mean_abs_delta"].abs()
        df["ftype"] = df["feature"].map(feature_type)
        out[g] = df
    return out


def plot_topn_bar(df: pd.DataFrame, out_png: Path, title: str, topn: int = 30):
    d = df.sort_values("rank").head(topn).copy()
    # reverse for nicer bar plot (largest on top)
    d = d.iloc[::-1]

    plt.figure(figsize=(12, max(6, 0.35 * len(d))))
    plt.barh(d["feature"], d["joint_stable"])
    plt.title(title)
    plt.xlabel("joint_stable")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_scatter(df: pd.DataFrame, out_png: Path, title: str, topn_label: int = 25):
    d = df.copy()
    size = (d["joint_strength"].fillna(0).clip(lower=0) * 200.0) + 10.0

    plt.figure(figsize=(9, 7))
    plt.scatter(d["mean_abs_corr"], d["mean_abs_delta"], s=size, alpha=0.6)
    plt.title(title)
    plt.xlabel("mean_abs_corr")
    plt.ylabel("mean_abs_delta")

    # label a few top features
    lab = df.sort_values("rank").head(topn_label)
    for _, r in lab.iterrows():
        plt.text(r["mean_abs_corr"], r["mean_abs_delta"], r["feature"], fontsize=7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_md", required=True, help="04h output report.md")
    ap.add_argument("--out_dir", required=True, help="output dir")
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--only_groups", default="", help="comma-separated group names to plot (optional)")
    args = ap.parse_args()

    report_md = Path(args.report_md)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = parse_report_md(report_md)
    if not groups:
        print(f"[ERROR] No tables parsed from: {report_md}")
        print("Expected markdown tables with header:")
        print("|rank|feature|joint_stable|mean_abs_corr|mean_abs_delta|median_delta|frac_pos_corr|frac_pos_delta|")
        raise SystemExit(2)

    wanted = None
    if args.only_groups.strip():
        wanted = {g.strip() for g in args.only_groups.split(",") if g.strip()}

    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # combined summary
    all_rows = []

    for g, df in groups.items():
        if wanted and g not in wanted:
            continue

        slug = slugify(g)
        df_out = tables_dir / f"{slug}.tsv"
        df.to_csv(df_out, sep="\t", index=False)

        plot_topn_bar(
            df,
            plots_dir / f"{slug}.top{args.topn}.joint_stable.png",
            title=f"{report_md.name} :: {g} :: Top{args.topn} joint_stable",
            topn=args.topn,
        )
        plot_scatter(
            df,
            plots_dir / f"{slug}.scatter.corr_vs_delta.png",
            title=f"{report_md.name} :: {g} :: corr vs delta",
        )

        all_rows.append(df.assign(group=g))

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(out_dir / "summary.all_groups.tsv", sep="\t", index=False)

    # feature-type counts
    ct = all_df.groupby(["group", "ftype"]).size().reset_index(name="n")
    ct.to_csv(out_dir / "feature_type_counts.tsv", sep="\t", index=False)

    # plot stacked counts (simple grouped bar)
    pivot = ct.pivot(index="group", columns="ftype", values="n").fillna(0)
    pivot = pivot.sort_index()

    plt.figure(figsize=(12, max(5, 0.4 * len(pivot))))
    pivot.plot(kind="barh", stacked=True, ax=plt.gca())
    plt.title(f"{report_md.name} :: feature type counts (parsed rows)")
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_type_counts.stacked.png", dpi=200)
    plt.close()

    print(f"[DONE] Parsed groups={len(groups)}  -> {out_dir}")


if __name__ == "__main__":
    main()
