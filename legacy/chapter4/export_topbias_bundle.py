#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import gzip
import json
from pathlib import Path

import pandas as pd
import numpy as np

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False

def peek_tsv_gz(path: Path, n=5):
    try:
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f, sep="\t", nrows=n)
        return {"columns": list(df.columns), "head": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot_tables_dir", required=True, help="解压后的 07_PlotTables_v3_topbias 目录")
    ap.add_argument("--analysis_root", required=True, help="analysis_results 目录的上级路径（含 analysis_results/05_ModelInputs...）")
    ap.add_argument("--outdir", required=True, help="导出目录")
    args = ap.parse_args()

    plot_dir = Path(args.plot_tables_dir)
    analysis_root = Path(args.analysis_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    overview = pd.read_csv(plot_dir / "dataset_overview.tsv", sep="\t")
    metrics = pd.read_csv(plot_dir / "01_master_metrics_ci_strat_by_log2fc_COMPAT.tsv", sep="\t")

    inventory = []
    # 1) 按 overview 导出 y 与 meta
    for _, r in overview.iterrows():
        for key in ["y_file", "meta_file"]:
            rel = Path(r[key])
            src = analysis_root / rel
            dst = outdir / rel
            ok = safe_copy(src, dst)
            inventory.append({"source": str(src), "dest": str(dst), "ok": ok, "kind": key,
                              "tag": r["tag"], "species": r["species"], "locus": r["locus"],
                              "variant": r["variant"], "split": r["split"]})

    # 2) 按 metrics 导出 pred_test
    for _, r in metrics.iterrows():
        rel = Path(r["pred_path"])
        src = analysis_root / rel
        dst = outdir / rel
        ok = safe_copy(src, dst)
        inventory.append({"source": str(src), "dest": str(dst), "ok": ok, "kind": "pred_path",
                          "tag": r["tag"], "species": r["species"], "locus": r["locus"],
                          "variant": r["variant"], "model": r["model"]})

    inv_path = outdir / "EXPORT_inventory.tsv"
    pd.DataFrame(inventory).to_csv(inv_path, sep="\t", index=False)

    # 3) 提取一个 meta 的列名样例，方便我直接写“方法”
    sample_meta = None
    for item in inventory:
        if item["ok"] and item["kind"] == "meta_file":
            p = Path(item["dest"])
            if p.suffixes[-2:] == [".tsv", ".gz"]:
                sample_meta = p
                break

    snapshot = {
        "analysis_root": str(analysis_root),
        "plot_tables_dir": str(plot_dir),
        "sample_meta_peek": peek_tsv_gz(sample_meta) if sample_meta else {"note": "no meta_file exported"},
    }
    with open(outdir / "METHOD_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"[OK] Export finished: {outdir}")
    print(f"[OK] Inventory: {inv_path}")

if __name__ == "__main__":
    main()
