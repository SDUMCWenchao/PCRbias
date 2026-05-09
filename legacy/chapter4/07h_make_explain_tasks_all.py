#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--inputs_root", default="analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--models_root", default="analysis_results/06_Models_v3_topbias")
    ap.add_argument("--model", choices=["rf", "seqcnn"], required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--out_tasks", required=True)
    args = ap.parse_args()

    project = Path(args.project_dir)
    iroot = project / args.inputs_root
    mroot = project / args.models_root
    out_tasks = project / args.out_tasks
    out_tasks.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_tasks.open("w", encoding="utf-8") as w:
        # models_root/<tag>/<species>/<locus>/<model>/<variant>
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
                    model_dir = lc_dir / args.model
                    if not model_dir.exists():
                        continue
                    for v_dir in sorted(model_dir.glob("*")):
                        if not v_dir.is_dir():
                            continue
                        v = v_dir.name
                        ds = iroot / tag / sp / lc / v
                        if not ds.exists():
                            continue
                        w.write(f"{tag}\t{sp}\t{lc}\t{v}\t{args.split}\t{ds}\t{v_dir}\n")
                        n += 1

    print(f"[DONE] model={args.model} tasks={n} -> {out_tasks}")

if __name__ == "__main__":
    main()
