#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4H (Joint ranking; stratify by locus):
Combine:
  (A) pair_feature_corr.tsv.gz  : feature ? log2fc correlation per pair
  (B) pair_feature_shift*.tsv.gz: feature shift between PCR yes/no per pair

Output:
  meta_corr__<group>.tsv
  meta_shift__<group>.tsv
  meta_joint__<group>.tsv
  report.md

Important:
- Supports --out_dir to avoid overwriting results from different feature sets.
- Robustly reads various shift schemas:
    Prefer delta_yes_minus_no; else use mean_yes - mean_no.
    effect size column may be effect_d (optional).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional


def safe_name(s: str) -> str:
    """Make a filesystem-safe token."""
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def sign_consistency(frac_pos: float) -> float:
    """0..1 ; 1 means all same sign, 0 means half-half"""
    return 2.0 * abs(frac_pos - 0.5)


class CorrAgg:
    __slots__ = ("vals",)

    def __init__(self) -> None:
        self.vals: List[float] = []

    def add(self, c: float) -> None:
        if math.isfinite(c):
            self.vals.append(c)

    def summary(self) -> Dict[str, Any]:
        n = len(self.vals)
        if n == 0:
            return {}
        abs_vals = [abs(x) for x in self.vals]
        mean_abs = sum(abs_vals) / n
        med_abs = statistics.median(abs_vals)
        frac_pos = sum(1 for x in self.vals if x > 0) / n
        sc = sign_consistency(frac_pos)
        stability = mean_abs * sc
        return {
            "n_pairs_corr": n,
            "mean_abs_corr": mean_abs,
            "median_abs_corr": med_abs,
            "frac_pos_corr": frac_pos,
            "sign_cons_corr": sc,
            "stability_corr": stability,
        }


class ShiftAgg:
    __slots__ = ("deltas", "effects")

    def __init__(self) -> None:
        self.deltas: List[float] = []
        self.effects: List[float] = []

    def add(self, delta: float, effect_d: Optional[float]) -> None:
        if math.isfinite(delta):
            self.deltas.append(delta)
        if effect_d is not None and math.isfinite(effect_d):
            self.effects.append(effect_d)

    def summary(self) -> Dict[str, Any]:
        n = len(self.deltas)
        if n == 0:
            return {}
        abs_d = [abs(x) for x in self.deltas]
        mean_abs_delta = sum(abs_d) / n
        med_delta = statistics.median(self.deltas)
        frac_pos = sum(1 for x in self.deltas if x > 0) / n
        sc = sign_consistency(frac_pos)
        med_abs_eff = statistics.median([abs(x) for x in self.effects]) if self.effects else float("nan")
        return {
            "n_pairs_shift": n,
            "median_delta": med_delta,
            "mean_abs_delta": mean_abs_delta,
            "frac_pos_delta": frac_pos,
            "sign_cons_delta": sc,
            "median_abs_effect_d": med_abs_eff,
        }


def read_pairs_map(pairs_fp: Path) -> Dict[str, Dict[str, str]]:
    m: Dict[str, Dict[str, str]] = {}
    with pairs_fp.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            pid = row.get("pair_id")
            if not pid:
                continue
            m[pid] = {
                "species": row.get("species", "NA"),
                "n_individuals": row.get("n_individuals", "NA"),
                "locus": row.get("locus", "NA"),
                "yes_file_id": row.get("yes_file_id", "NA"),
                "no_file_id": row.get("no_file_id", "NA"),
            }
    if not m:
        raise ValueError(f"pairs.tsv empty or unreadable: {pairs_fp}")
    return m


def group_keys(pair_meta: Dict[str, str]) -> List[Tuple[str, str]]:
    locus = pair_meta.get("locus", "NA")
    return [("ALL", "ALL"), ("locus", locus)]


def open_tsv_maybe_gz(fp: Path):
    if str(fp).endswith(".gz"):
        return gzip.open(fp, "rt", encoding="utf-8", newline="")
    return fp.open("r", encoding="utf-8", newline="")


def write_tsv(fp: Path, header: List[str], rows: List[List[Any]]) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.upper() == "NA" or s.upper() == "NAN":
        return None
    try:
        v = float(s)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def infer_delta_from_row(row: Dict[str, str]) -> Optional[float]:
    # prefer explicit delta
    for k in ("delta_yes_minus_no", "delta", "delta_yes_no"):
        v = parse_float(row.get(k))
        if v is not None:
            return v
    # else infer from mean_yes/mean_no
    my = parse_float(row.get("mean_yes"))
    mn = parse_float(row.get("mean_no"))
    if my is not None and mn is not None:
        return my - mn
    return None


def infer_effect_from_row(row: Dict[str, str]) -> Optional[float]:
    for k in ("effect_d", "cohen_d", "d"):
        v = parse_float(row.get(k))
        if v is not None:
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")

    ap.add_argument("--pairs_tsv", default=None,
                    help="default: analysis_results/03_DataWeaver/pairs.tsv")

    ap.add_argument("--pair_corr", default=None,
                    help="default: analysis_results/04_Stats/pair_feature_corr.tsv.gz")
    ap.add_argument("--pair_shift", default=None,
                    help="default: analysis_results/04_Stats/pair_feature_shift_top200.tsv.gz")

    ap.add_argument("--out_dir", default=None,
                    help="Output directory (relative to project_dir or absolute). "
                         "Example: analysis_results/04_Stats_NoKmer/joint")

    ap.add_argument("--top_k_report", type=int, default=30)
    args = ap.parse_args()

    project = Path(args.project_dir)

    pairs_fp = Path(args.pairs_tsv) if args.pairs_tsv else (project / "analysis_results" / "03_DataWeaver" / "pairs.tsv")
    corr_fp = Path(args.pair_corr) if args.pair_corr else (project / "analysis_results" / "04_Stats" / "pair_feature_corr.tsv.gz")
    shift_fp = Path(args.pair_shift) if args.pair_shift else (project / "analysis_results" / "04_Stats" / "pair_feature_shift_top200.tsv.gz")

    if not pairs_fp.exists():
        raise FileNotFoundError(f"Missing pairs.tsv: {pairs_fp}")
    if not corr_fp.exists():
        raise FileNotFoundError(f"Missing corr file: {corr_fp}")
    if not shift_fp.exists():
        raise FileNotFoundError(f"Missing shift file: {shift_fp}")

    # output dir (critical to avoid overwriting)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = project / out_dir
    else:
        out_dir = project / "analysis_results" / "04_Stats" / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_map = read_pairs_map(pairs_fp)

    # ---- aggregate corr by group + feature ----
    corr_aggs: Dict[Tuple[str, str, str], CorrAgg] = {}
    with open_tsv_maybe_gz(corr_fp) as f:
        r = csv.DictReader(f, delimiter="\t")
        # expected columns: pair_id, feature, pearson_corr (others ok)
        for row in r:
            pid = row.get("pair_id")
            feat = row.get("feature")
            c = parse_float(row.get("pearson_corr"))
            if not pid or not feat or c is None:
                continue
            pm = pair_map.get(pid)
            if pm is None:
                continue
            for gt, gv in group_keys(pm):
                key = (gt, gv, feat)
                if key not in corr_aggs:
                    corr_aggs[key] = CorrAgg()
                corr_aggs[key].add(c)

    corr_sum: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for k, agg in corr_aggs.items():
        corr_sum[k] = agg.summary()

    # ---- aggregate shift by group + feature ----
    shift_aggs: Dict[Tuple[str, str, str], ShiftAgg] = {}
    with open_tsv_maybe_gz(shift_fp) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            pid = row.get("pair_id")
            feat = row.get("feature")
            if not pid or not feat:
                continue
            delta = infer_delta_from_row(row)
            if delta is None:
                continue
            eff = infer_effect_from_row(row)

            pm = pair_map.get(pid)
            if pm is None:
                continue
            for gt, gv in group_keys(pm):
                key = (gt, gv, feat)
                if key not in shift_aggs:
                    shift_aggs[key] = ShiftAgg()
                shift_aggs[key].add(delta, eff)

    shift_sum: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for k, agg in shift_aggs.items():
        shift_sum[k] = agg.summary()

    # ---- groups present ----
    groups = sorted(set((gt, gv) for (gt, gv, _) in list(corr_sum.keys()) + list(shift_sum.keys())))

    # ---- write meta corr/shift per group ----
    def dump_corr(gt: str, gv: str) -> None:
        rows: List[List[Any]] = []
        for (gtt, gvv, feat), s in corr_sum.items():
            if gtt != gt or gvv != gv:
                continue
            rows.append([
                feat,
                s["n_pairs_corr"],
                f"{s['mean_abs_corr']:.6g}",
                f"{s['median_abs_corr']:.6g}",
                f"{s['frac_pos_corr']:.6g}",
                f"{s['sign_cons_corr']:.6g}",
                f"{s['stability_corr']:.6g}",
            ])
        rows.sort(key=lambda x: float(x[6]), reverse=True)
        fp = out_dir / f"meta_corr__{safe_name(gt)}_{safe_name(gv)}.tsv"
        write_tsv(fp,
                  ["feature", "n_pairs_corr", "mean_abs_corr", "median_abs_corr", "frac_pos_corr", "sign_cons_corr", "stability_corr"],
                  rows)

    def dump_shift(gt: str, gv: str) -> None:
        rows: List[List[Any]] = []
        for (gtt, gvv, feat), s in shift_sum.items():
            if gtt != gt or gvv != gv:
                continue
            rows.append([
                feat,
                s["n_pairs_shift"],
                f"{s['median_delta']:.6g}",
                f"{s['mean_abs_delta']:.6g}",
                f"{s['frac_pos_delta']:.6g}",
                f"{s['sign_cons_delta']:.6g}",
                (f"{float(s['median_abs_effect_d']):.6g}" if s["median_abs_effect_d"] == s["median_abs_effect_d"] else "NA"),
            ])
        rows.sort(key=lambda x: float(x[3]), reverse=True)
        fp = out_dir / f"meta_shift__{safe_name(gt)}_{safe_name(gv)}.tsv"
        write_tsv(fp,
                  ["feature", "n_pairs_shift", "median_delta", "mean_abs_delta", "frac_pos_delta", "sign_cons_delta", "median_abs_effect_d"],
                  rows)

    for gt, gv in groups:
        dump_corr(gt, gv)
        dump_shift(gt, gv)

    # ---- joint merge per group ----
    def dump_joint(gt: str, gv: str) -> List[List[Any]]:
        feats = set()
        for (gtt, gvv, feat) in corr_sum.keys():
            if gtt == gt and gvv == gv:
                feats.add(feat)
        for (gtt, gvv, feat) in shift_sum.keys():
            if gtt == gt and gvv == gv:
                feats.add(feat)

        rows: List[List[Any]] = []
        for feat in feats:
            c = corr_sum.get((gt, gv, feat), {})
            s = shift_sum.get((gt, gv, feat), {})
            if not c or not s:
                continue  # require both sides

            mean_abs_corr = float(c["mean_abs_corr"])
            sc_corr = float(c["sign_cons_corr"])
            mean_abs_delta = float(s["mean_abs_delta"])
            sc_delta = float(s["sign_cons_delta"])

            joint_strength = mean_abs_corr * mean_abs_delta
            joint_stable = (mean_abs_corr * sc_corr) * (mean_abs_delta * sc_delta)

            rows.append([
                feat,
                c["n_pairs_corr"],
                f"{mean_abs_corr:.6g}",
                f"{float(c['median_abs_corr']):.6g}",
                f"{float(c['frac_pos_corr']):.6g}",
                f"{sc_corr:.6g}",
                f"{float(c['stability_corr']):.6g}",
                s["n_pairs_shift"],
                f"{float(s['median_delta']):.6g}",
                f"{mean_abs_delta:.6g}",
                f"{float(s['frac_pos_delta']):.6g}",
                f"{sc_delta:.6g}",
                (f"{float(s['median_abs_effect_d']):.6g}" if s["median_abs_effect_d"] == s["median_abs_effect_d"] else "NA"),
                f"{joint_strength:.6g}",
                f"{joint_stable:.6g}",
            ])

        rows.sort(key=lambda x: float(x[-1]), reverse=True)  # joint_stable
        fp = out_dir / f"meta_joint__{safe_name(gt)}_{safe_name(gv)}.tsv"
        write_tsv(fp,
                  ["feature",
                   "n_pairs_corr", "mean_abs_corr", "median_abs_corr", "frac_pos_corr", "sign_cons_corr", "stability_corr",
                   "n_pairs_shift", "median_delta", "mean_abs_delta", "frac_pos_delta", "sign_cons_delta", "median_abs_effect_d",
                   "joint_strength", "joint_stable"],
                  rows)
        return rows

    joint_rows_by_group: Dict[Tuple[str, str], List[List[Any]]] = {}
    for gt, gv in groups:
        joint_rows_by_group[(gt, gv)] = dump_joint(gt, gv)

    # ---- report.md ----
    rep_fp = out_dir / "report.md"
    with rep_fp.open("w", encoding="utf-8") as fo:
        fo.write("# Joint feature ranking report\n\n")
        fo.write(f"- pairs.tsv: {pairs_fp}\n")
        fo.write(f"- corr input: {corr_fp}\n")
        fo.write(f"- shift input: {shift_fp}\n\n")
        fo.write("Scoring:\n")
        fo.write("- joint_strength = mean_abs_corr * mean_abs_delta\n")
        fo.write("- joint_stable   = (mean_abs_corr*sign_cons_corr) * (mean_abs_delta*sign_cons_delta)\n\n")

        for gt, gv in groups:
            rows = joint_rows_by_group.get((gt, gv), [])
            fo.write(f"## Group: {gt} = {gv}\n\n")
            fo.write("|rank|feature|joint_stable|mean_abs_corr|mean_abs_delta|median_delta|frac_pos_corr|frac_pos_delta|\n")
            fo.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
            for i, r in enumerate(rows[:args.top_k_report], start=1):
                fo.write(f"|{i}|{r[0]}|{r[-1]}|{r[2]}|{r[9]}|{r[8]}|{r[4]}|{r[10]}|\n")
            fo.write("\n")

    print(f"[DONE] outputs -> {out_dir}")
    print(f"[DONE] report  -> {rep_fp}")


if __name__ == "__main__":
    main()
