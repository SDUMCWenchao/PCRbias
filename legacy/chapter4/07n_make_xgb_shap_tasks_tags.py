#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--models_root", default="analysis_results/06_Models_v3_topbias")
    ap.add_argument("--tags", default="top1p")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_tasks", required=True)
    args = ap.parse_args()

    project = Path(args.project_dir)
    iroot = project / args.inputs_root
    mroot = project / args.models_root
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    out_tasks = project / args.out_tasks
    out_tasks.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_tasks.open("w", encoding="utf-8") as w:
        for tag in tags:
            tag_dir = mroot / tag
            if not tag_dir.exists():
                continue
            for sp_dir in sorted(tag_dir.glob("*")):
                if not sp_dir.is_dir(): continue
                sp = sp_dir.name
                for lc_dir in sorted(sp_dir.glob("*")):
                    if not lc_dir.is_dir(): continue
                    lc = lc_dir.name
                    xdir = lc_dir / "xgb"
                    if not xdir.exists(): continue
                    for v_dir in sorted(xdir.glob("*")):
                        if not v_dir.is_dir(): continue
                        v = v_dir.name
                        ds = iroot / tag / sp / lc / v
                        if not ds.exists():
                            continue
                        w.write(f"{tag}\t{sp}\t{lc}\t{v}\t{args.split}\t{ds}\t{v_dir}\n")
                        n += 1

    print(f"[DONE] xgb shap tasks={n} -> {out_tasks}")

if __name__ == "__main__":
    main()
