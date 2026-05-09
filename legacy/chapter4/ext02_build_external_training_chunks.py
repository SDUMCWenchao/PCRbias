#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, gzip, math
from pathlib import Path
import numpy as np
import pandas as pd

def fasta_iter(path: Path):
    name = None
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name is not None:
            yield name, "".join(seq)

def safe_log2(x):
    return np.log(x) / np.log(2.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcrbias_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--y_clip", type=float, default=6.0)
    ap.add_argument("--rows_per_chunk", type=int, default=200000)
    args = ap.parse_args()

    pcrbias_root = Path(args.pcrbias_root).resolve()
    out_root = Path(args.out_root).resolve()

    mani = out_root / "00_manifest"
    ds_df = pd.read_csv(mani / "datasets.tsv", sep="\t")
    pairs = pd.read_csv(out_root / "01_pairs" / "pairs.tsv.gz", sep="\t", compression="gzip")

    # 输出目录（模仿你现有结构）
    seq_dir = out_root / "analysis_results" / "01_Sequences"
    tw_dir  = out_root / "analysis_results" / "03_DataWeaver"
    chunk_dir = tw_dir / "training_chunks"
    seq_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # 1) 写全量 fasta（header 用 Seq_ID=dataset:seq_id，避免冲突）
    fasta_out = seq_dir / "ALL_SEQUENCES.external.fasta"
    map_out   = seq_dir / "seq_map.external.tsv.gz"
    with open(fasta_out, "w", encoding="utf-8") as fo, gzip.open(map_out, "wt", encoding="utf-8") as mo:
        mo.write("Seq_ID\tdataset\tseq_id\torig_fasta_id\tlen\n")
        for _, r in ds_df.iterrows():
            ds = r["dataset"]
            fa = Path(r["design_fasta"])
            for fid, s in fasta_iter(fa):
                # fid 形如 000000；对应 abundance seq_id=0
                try:
                    sid = int(fid)
                except Exception:
                    # 退化：直接用原id
                    sid = fid
                seq_id = f"{ds}:{sid}"
                fo.write(f">{seq_id}\n{s}\n")
                mo.write(f"{seq_id}\t{ds}\t{sid}\t{fid}\t{len(s)}\n")

    print("[DONE] fasta ->", fasta_out)
    print("[DONE] map   ->", map_out)

    # 2) 逐数据集构造 pair rows，并按 rows_per_chunk 切 chunk_XXXX.train.tsv.gz
    chunk_idx = 0
    in_chunk = 0
    chunk_fh = None

    def open_new_chunk():
        nonlocal chunk_idx, in_chunk, chunk_fh
        if chunk_fh is not None:
            chunk_fh.close()
        p = chunk_dir / f"chunk_{chunk_idx:04d}.train.tsv.gz"
        chunk_fh = gzip.open(p, "wt", encoding="utf-8")
        header = [
            "pair_id","dataset","yes_sample","no_sample",
            "Seq_ID","seq_id",
            "count_yes","count_no","total_yes","total_no",
            "rel_yes","rel_no","log2fc",
            "cycle_yes_order","cycle_no_order","cycle_delta_order"
        ]
        chunk_fh.write("\t".join(header) + "\n")
        in_chunk = 0
        chunk_idx += 1
        return p

    cur_chunk_path = open_new_chunk()

    # quick lookup ds->dir
    ds2ab = {r["dataset"]: Path(r["abundance_csv"]) for _, r in ds_df.iterrows()}

    for ds, g in pairs.groupby("dataset", sort=False):
        ab = ds2ab.get(ds)
        if ab is None or not ab.exists():
            continue
        # 只读需要的列：seq_id + 所有相关实验列
        need_cols = ["seq_id"] + sorted(set(g["yes_sample"].tolist() + g["no_sample"].tolist()))
        df = pd.read_csv(ab, usecols=need_cols)
        # totals per column
        totals = {c: float(df[c].sum()) for c in need_cols if c != "seq_id"}

        seq_id_arr = df["seq_id"].astype(int).values

        for _, pr in g.iterrows():
            yes = pr["yes_sample"]; no = pr["no_sample"]
            cy = pr.get("cycle_yes_order",""); cn = pr.get("cycle_no_order",""); cd = pr.get("cycle_delta_order","")

            c_yes = df[yes].astype(float).values
            c_no  = df[no].astype(float).values
            t_yes = totals[yes] if totals[yes] > 0 else 1.0
            t_no  = totals[no]  if totals[no]  > 0 else 1.0

            rel_yes = c_yes / t_yes
            rel_no  = c_no  / t_no
            y = safe_log2((rel_yes + args.eps) / (rel_no + args.eps))
            if args.y_clip is not None and args.y_clip > 0:
                y = np.clip(y, -args.y_clip, args.y_clip)

            # 写出（逐行，避免巨型中间表）
            for i in range(len(seq_id_arr)):
                if in_chunk >= args.rows_per_chunk:
                    cur_chunk_path = open_new_chunk()

                sid = int(seq_id_arr[i])
                Seq_ID = f"{ds}:{sid}"
                row = [
                    pr["pair_id"], ds, yes, no,
                    Seq_ID, str(sid),
                    str(int(c_yes[i])), str(int(c_no[i])),
                    str(int(t_yes)), str(int(t_no)),
                    f"{rel_yes[i]:.12g}", f"{rel_no[i]:.12g}", f"{y[i]:.6g}",
                    str(cy), str(cn), str(cd)
                ]
                chunk_fh.write("\t".join(row) + "\n")
                in_chunk += 1

        print(f"[INFO] built pairs for {ds}: pairs={len(g)} rows={len(df)*len(g)}")

    if chunk_fh is not None:
        chunk_fh.close()

    # 写索引
    idx_path = tw_dir / "training_chunks.index.tsv"
    with open(idx_path, "w", encoding="utf-8") as f:
        f.write("chunk_file\n")
        for p in sorted(chunk_dir.glob("chunk_*.train.tsv.gz")):
            f.write(str(p) + "\n")
    print("[DONE] chunks ->", chunk_dir)
    print("[DONE] index  ->", idx_path)

if __name__ == "__main__":
    main()
