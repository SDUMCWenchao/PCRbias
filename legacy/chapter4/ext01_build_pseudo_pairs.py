#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import pandas as pd

def parse_cycle_order(sample_name: str):
    m = re.fullmatch(r"PCR0*([0-9]+)", sample_name)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--pair_mode", default="vs_baseline", choices=["vs_baseline","max_vs_min"], help="默认：所有实验 vs baseline")
    ap.add_argument("--baseline", default="auto", help="auto 或指定样本名（如 PCR1）")
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    mani = out_root / "00_manifest"
    ds_path = mani / "datasets.tsv"
    if not ds_path.exists():
        raise FileNotFoundError(f"missing {ds_path}, run ext00 first")

    ds_df = pd.read_csv(ds_path, sep="\t")
    pair_dir = out_root / "01_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in ds_df.iterrows():
        ds = r["dataset"]
        ab = Path(r["abundance_csv"])
        # 只读表头
        cols = list(pd.read_csv(ab, nrows=1).columns)
        exp_cols = [c for c in cols if c != "seq_id"]
        if len(exp_cols) < 2:
            continue

        # cycle_order 仅用于排序/标记
        cyc = {c: parse_cycle_order(c) for c in exp_cols}

        def pick_baseline():
            if args.baseline != "auto":
                if args.baseline in exp_cols:
                    return args.baseline
                # 如果指定了但不在列里，就退化为 auto
            # auto：优先最小 cycle_order；否则取第一列
            with_cyc = [c for c in exp_cols if cyc[c] is not None]
            if with_cyc:
                return sorted(with_cyc, key=lambda x: cyc[x])[0]
            return exp_cols[0]

        base = pick_baseline()

        # yes 列
        others = [c for c in exp_cols if c != base]
        # 排序：有 cycle_order 的按 cycle_order；否则保持原顺序
        with_cyc = [c for c in others if cyc[c] is not None]
        without = [c for c in others if cyc[c] is None]
        with_cyc = sorted(with_cyc, key=lambda x: cyc[x])
        yes_list = with_cyc + without

        if args.pair_mode == "max_vs_min":
            # 只取最大 vs 最小（如果有 cycle_order）
            if with_cyc:
                mx = sorted(with_cyc, key=lambda x: cyc[x])[-1]
                yes_list = [mx]
            else:
                yes_list = [yes_list[-1]]

        for yes in yes_list:
            rows.append({
                "pair_id": f"{ds}::{yes}_vs_{base}",
                "dataset": ds,
                "dataset_type": r["dataset_type"],
                "yes_sample": yes,
                "no_sample": base,
                "cycle_yes_order": ("" if cyc[yes] is None else cyc[yes]),
                "cycle_no_order": ("" if cyc[base] is None else cyc[base]),
                "cycle_delta_order": ("" if (cyc[yes] is None or cyc[base] is None) else (cyc[yes]-cyc[base])),
            })

    out = pd.DataFrame(rows)
    out_path = pair_dir / "pairs.tsv.gz"
    out.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print("[DONE]", out_path)
    print("[NOTE] PCR1..PCR6 的 cycle_order 已写入；真实 cycle 数值未知可后续在 00_manifest/cycle_map.tsv 填写再扩展。")

if __name__ == "__main__":
    main()
