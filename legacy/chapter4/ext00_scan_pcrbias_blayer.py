#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import pandas as pd

def list_datasets(pcrbias_root: Path, include_validation: bool):
    base = pcrbias_root / "analysis" / "data"
    ext_root = base / "external_datasets"
    int_root = base / "internal_datasets"

    datasets = []
    # external_datasets 下全部
    for d in sorted(ext_root.glob("*")):
        if d.is_dir():
            datasets.append(("external", d.name, d))

    # internal_datasets：默认只要 GCall/GCfix；可选把 validation_* 也纳入
    for name in ["GCall", "GCfix"]:
        d = int_root / name
        if d.is_dir():
            datasets.append(("internal", name, d))

    if include_validation:
        for d in sorted(int_root.glob("validation_*")):
            if d.is_dir():
                datasets.append(("internal_validation", d.name, d))

    return datasets

def parse_cycle_order(sample_name: str):
    # PCR1..PCR6 / PCR01..PCR10 之类
    m = re.fullmatch(r"PCR0*([0-9]+)", sample_name)
    if m:
        return int(m.group(1))
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcrbias_root", required=True, help=".../PCR-bias-1.0.3/PCR-bias-1.0.3")
    ap.add_argument("--out_root", required=True, help=".../2511_PCR_Bias/external_test")
    ap.add_argument("--include_validation", action="store_true", help="include internal validation_* datasets")
    args = ap.parse_args()

    pcrbias_root = Path(args.pcrbias_root).resolve()
    out_root = Path(args.out_root).resolve()
    mani_dir = out_root / "00_manifest"
    mani_dir.mkdir(parents=True, exist_ok=True)

    ds_rows = []
    sample_rows = []

    for ds_type, ds_name, ds_dir in list_datasets(pcrbias_root, args.include_validation):
        ab = ds_dir / "abundance_by_experiment.csv"
        fa = ds_dir / "design_files.fasta"
        pa = ds_dir / "params.csv"
        sp = ds_dir / "seqprops.csv"

        if not ab.exists() or not fa.exists():
            continue

        # 只读表头 + 行数（尽量轻）
        df0 = pd.read_csv(ab, nrows=5)
        cols = list(df0.columns)
        exp_cols = [c for c in cols if c != "seq_id"]

        n_rows = sum(1 for _ in open(ab, "r", encoding="utf-8")) - 1
        ds_rows.append({
            "dataset": ds_name,
            "dataset_type": ds_type,
            "dataset_dir": str(ds_dir),
            "abundance_csv": str(ab),
            "design_fasta": str(fa),
            "has_params": int(pa.exists()),
            "has_seqprops": int(sp.exists()),
            "n_seqs_abundance": int(n_rows),
            "n_experiments": int(len(exp_cols)),
            "experiment_cols": ",".join(exp_cols),
        })

        for c in exp_cols:
            cyc = parse_cycle_order(c)
            sample_rows.append({
                "dataset": ds_name,
                "dataset_type": ds_type,
                "sample": c,
                "cycle_order": ("" if cyc is None else cyc),
                # cycle_n：真实循环数（未知先空；你后续可填写）
                "cycle_n": "",
            })

    ds_df = pd.DataFrame(ds_rows)
    sm_df = pd.DataFrame(sample_rows)

    ds_path = mani_dir / "datasets.tsv"
    sm_path = mani_dir / "samples.tsv"
    cm_path = mani_dir / "cycle_map.tsv"

    ds_df.to_csv(ds_path, sep="\t", index=False)
    sm_df.to_csv(sm_path, sep="\t", index=False)
    # cycle_map 与 samples 相同，单独拎出来方便你后面手工补真实循环数
    sm_df.to_csv(cm_path, sep="\t", index=False)

    print("[DONE]", ds_path)
    print("[DONE]", sm_path)
    print("[NOTE] PCR1..PCR6 已解析为 cycle_order（序位）；cycle_n 真实循环数目前为空，后续可改 00_manifest/cycle_map.tsv")

if __name__ == "__main__":
    main()
