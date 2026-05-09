#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
from pathlib import Path
import pandas as pd

def parse_metrics(p: Path):
    try:
        m = json.loads(p.read_text())
    except Exception as e:
        return {"parse_error": str(e)}
    out = {"parse_error": ""}
    for split in ["train", "val", "test"]:
        # 支持 {"train":{"n":..,"r2":..}} 或 {"train_r2":..,"n_train":..} 两种风格
        if isinstance(m.get(split), dict):
            d = m[split]
            out[f"n_{split}"] = d.get("n", None)
            for k in ["spearman","pearson","r2","rmse","mae"]:
                out[f"{split}_{k}"] = d.get(k, None)
        else:
            out[f"n_{split}"] = m.get(f"n_{split}", m.get(f"{split}_n", None))
            for k in ["spearman","pearson","r2","rmse","mae"]:
                out[f"{split}_{k}"] = m.get(f"{split}_{k}", None)
    return out

def find_layout(models_root: Path):
    # layout A: models_root/tag/compare/model/variant/metrics.json
    # layout B: models_root/compare/model/variant/metrics.json  (无 tag 层)
    tags = [d.name for d in models_root.iterdir() if d.is_dir() and d.name.startswith("top")]
    if tags:
        return "tag_first"
    return "no_tag"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", required=True, help="e.g. external_test/analysis_results/06_Models_external_topbias_v2_resplit_v1_resplit_v1")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--tag", default="", help="only used when models_root has no tag layer")
    args = ap.parse_args()

    root = Path(args.models_root)
    if not root.exists():
        raise FileNotFoundError(f"[BAD] not exists: {root}")

    layout = find_layout(root)
    rows = []

    if layout == "tag_first":
        tag_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("top")])
        for tag_dir in tag_dirs:
            tag = tag_dir.name
            for metrics in tag_dir.rglob("metrics.json"):
                # .../tag/compare/model/variant/metrics.json
                parts = metrics.relative_to(tag_dir).parts
                if len(parts) < 4:
                    continue
                compare, model, variant = parts[0], parts[1], parts[2]
                base = metrics.parent
                row = {
                    "tag": tag, "compare_id": compare, "model": model, "variant": variant,
                    "metrics_json": str(metrics),
                }
                row.update(parse_metrics(metrics))

                # 文件存在性 + 大小（快速判断 val/test 是否只有表头）
                def sz(p): return os.path.getsize(p) if os.path.exists(p) else 0
                row.update({
                    "pred_train_size": sz(base / "pred_train.tsv"),
                    "pred_val_size":   sz(base / "pred_val.tsv"),
                    "pred_test_size":  sz(base / "pred_test.tsv"),
                    "has_model_joblib": (base/"model.joblib").exists(),
                    "has_model_json":   (base/"model.json").exists(),
                    "has_model_pt":     (base/"model.pt").exists(),
                    "has_importance":   (base/"feature_importance.tsv").exists(),
                })
                rows.append(row)
    else:
        tag = args.tag.strip() or "NA"
        for metrics in root.rglob("metrics.json"):
            # .../compare/model/variant/metrics.json
            rel = metrics.relative_to(root).parts
            if len(rel) < 4:
                continue
            compare, model, variant = rel[0], rel[1], rel[2]
            base = metrics.parent
            row = {
                "tag": tag, "compare_id": compare, "model": model, "variant": variant,
                "metrics_json": str(metrics),
            }
            row.update(parse_metrics(metrics))
            def sz(p): return os.path.getsize(p) if os.path.exists(p) else 0
            row.update({
                "pred_train_size": sz(base / "pred_train.tsv"),
                "pred_val_size":   sz(base / "pred_val.tsv"),
                "pred_test_size":  sz(base / "pred_test.tsv"),
                "has_model_joblib": (base/"model.joblib").exists(),
                "has_model_json":   (base/"model.json").exists(),
                "has_model_pt":     (base/"model.pt").exists(),
                "has_importance":   (base/"feature_importance.tsv").exists(),
            })
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["tag","compare_id","model","variant"])
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"[DONE] audit rows={len(df)} -> {args.out_tsv}")

if __name__ == "__main__":
    main()
