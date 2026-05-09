#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

RE_GROUP = re.compile(r"^##\s+Group:\s*(.+?)\s*$")


def parse_top_features(report_md: Path, group_name: str, topn: int):
    in_group = False
    in_table = False
    feats = []

    with report_md.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")

            m = RE_GROUP.match(line)
            if m:
                in_group = (m.group(1).strip() == group_name)
                in_table = False
                continue

            if not in_group:
                continue

            if line.startswith("|rank|feature|joint_stable|"):
                in_table = True
                continue
            if not in_table:
                continue
            if line.startswith("|---"):
                continue
            if not line.startswith("|"):
                continue

            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) != 8:
                continue
            if not parts[0].isdigit():
                continue

            feats.append(parts[1])
            if len(feats) >= topn:
                break

    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_md", required=True)
    ap.add_argument("--group", default="ALL = ALL")
    ap.add_argument("--topn", type=int, default=3000)
    ap.add_argument("--mode", choices=["no_kmer", "kmer_only", "all"], required=True)
    ap.add_argument("--out_txt", required=True)
    args = ap.parse_args()

    feats = parse_top_features(Path(args.report_md), args.group, args.topn)

    if args.mode == "no_kmer":
        feats = [f for f in feats if not re.match(r"^k\d+_", f)]
    elif args.mode == "kmer_only":
        feats = [f for f in feats if re.match(r"^k\d+_", f)]
    else:
        pass

    out = Path(args.out_txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(feats) + "\n", encoding="utf-8")
    print(f"[DONE] mode={args.mode} n={len(feats)} -> {out}")


if __name__ == "__main__":
    main()
