#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import pandas as pd

def safe_load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def count_features(ds_dir: Path):
    fn = ds_dir / "feature_names.txt"
    if not fn.exists():
        return None
    return sum(1 for _ in fn.read_text(encoding="utf-8").splitlines() if _.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--models_root", default="analysis_results/06_Models_v3_topbias")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--out_tsv", default="analysis_results/07_Tables/topbias_models_metrics.tsv")
    args = ap.parse_args()

    project = Path(args.project_dir)
    mroot = project / args.models_root
    iroot = project / args.inputs_root
    out_tsv = project / args.out_tsv
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # expect structure: <tag>/<sp>/<locus>/{rf|xgb|seqcnn}/<variant>/metrics.json
    for tag_dir in sorted(mroot.glob("*")):
        if not tag_dir.is_dir():
            continue
        tag = tag_dir.name
        for sp_dir in sorted(tag_dir.glob("*")):
            if not sp_dir.is_dir():
                continue
            sp = sp_dir.name
            for lc_dir in sorted(sp_dir.glob("*")):
                if not lc_dir.is_dir():
                    continue
                lc = lc_dir.name
                for model_type in ["rf", "xgb", "seqcnn"]:
                    mt_dir = lc_dir / model_type
                    if not mt_dir.exists():
                        continue
                    for v_dir in sorted(mt_dir.glob("*")):
                        if not v_dir.is_dir():
                            continue
                        variant = v_dir.name
                        mj = v_dir / "metrics.json"
                        if not mj.exists():
                            continue
                        m = safe_load_json(mj)
                        if m is None:
                            continue

                        ds_dir = iroot / tag / sp / lc / variant
                        nfeat = count_features(ds_dir) if ds_dir.exists() else None

                        row = {
                            "tag": tag, "species": sp, "locus": lc,
                            "model": model_type, "variant": variant,
                            "nfeat": nfeat,
                            "val_r2": m.get("val", {}).get("r2"),
                            "val_rmse": m.get("val", {}).get("rmse"),
                            "val_spearman": m.get("val", {}).get("spearman"),
                            "val_sign_acc": m.get("val", {}).get("sign_acc"),
                            "val_n": m.get("val", {}).get("n"),
                            "test_r2": m.get("test", {}).get("r2"),
                            "test_rmse": m.get("test", {}).get("rmse"),
                            "test_spearman": m.get("test", {}).get("spearman"),
                            "test_sign_acc": m.get("test", {}).get("sign_acc"),
                            "test_n": m.get("test", {}).get("n"),
                            "test_pair_spear_mean": m.get("test_pair", {}).get("pair_spearman_mean"),
                            "test_pair_spear_n": m.get("test_pair", {}).get("pair_spearman_n"),
                        }
                        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["tag","species","locus","model","variant"]).reset_index(drop=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[DONE] wrote {out_tsv} rows={len(df)}")

if __name__ == "__main__":
    main()
