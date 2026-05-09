#!/usr/bin/env python3
from pathlib import Path
import argparse
import multiprocessing as mp
import subprocess
import tempfile
import pandas as pd
import re
import shutil
import json

REGIONS = ["global","head","mid1","mid2","mid3","tail"]
RNAFOLD_BIN = None

def init_worker(rnafold_bin):
    global RNAFOLD_BIN
    RNAFOLD_BIN = rnafold_bin

def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def run_rnafold_chunk(args):
    region, chunk_idx, chunk = args
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fa") as tmp:
        tmp_path = Path(tmp.name)
        for sid, seq in chunk:
            tmp.write(f">{sid}\n{seq}\n")
    cmd = [RNAFOLD_BIN, "--noPS", str(tmp_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    tmp_path.unlink(missing_ok=True)

    results = []
    lines = [x.rstrip("\n") for x in proc.stdout.splitlines() if x.strip()]
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        sid = lines[i][1:].strip()
        struct_line = lines[i+2].strip() if i+2 < len(lines) else ""
        m = re.search(r"\(([-0-9\.]+)\)", struct_line)
        mfe = float(m.group(1)) if m else None
        results.append((sid, mfe))
        i += 3
    return region, chunk_idx, results

def detect_rnafold(user_bin):
    if user_bin:
        p = Path(user_bin)
        if p.exists():
            return str(p.resolve())
        raise SystemExit(f"--rnafold-bin not found: {user_bin}")
    auto = shutil.which("RNAfold")
    if auto:
        return auto
    raise SystemExit("RNAfold not found. Activate the correct environment, or pass --rnafold-bin /absolute/path/to/RNAfold")

def main():
    ap = argparse.ArgumentParser(description="Parallel RNAfold feature extraction with chunk-level checkpoints.")
    ap.add_argument("--sequence-regions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threads", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--rnafold-bin", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "chunk_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rnafold_bin = detect_rnafold(args.rnafold_bin)
    print(f"[INFO] Using RNAfold binary: {rnafold_bin}")

    regions = pd.read_csv(args.sequence_regions, sep="\t", dtype=str).fillna("")

    jobs = []
    manifest_rows = []
    for region in REGIONS:
        seq_pairs = []
        for _, r in regions.iterrows():
            sid = r["sequence_id"]
            seq = r["sequence"] if region == "global" else r[region]
            if str(seq).strip():
                seq_pairs.append((sid, seq))
        for chunk_idx, chunk in enumerate(chunk_list(seq_pairs, args.chunk_size), start=1):
            ckpt_file = ckpt_dir / f"{region}.chunk_{chunk_idx:05d}.tsv"
            manifest_rows.append({
                "region": region,
                "chunk_idx": chunk_idx,
                "n_sequences": len(chunk),
                "checkpoint_file": str(ckpt_file),
                "done": int(ckpt_file.exists() and ckpt_file.stat().st_size > 0)
            })
            if args.resume and ckpt_file.exists() and ckpt_file.stat().st_size > 0:
                continue
            jobs.append((region, chunk_idx, chunk))

    pd.DataFrame(manifest_rows).to_csv(outdir / "rnafold_chunk_manifest.tsv", sep="\t", index=False)

    if jobs:
        print(f"[INFO] RNAfold chunks to run: {len(jobs)}")
        with mp.Pool(processes=args.threads, initializer=init_worker, initargs=(rnafold_bin,)) as pool:
            for region, chunk_idx, results in pool.imap_unordered(run_rnafold_chunk, jobs, chunksize=1):
                ckpt_file = ckpt_dir / f"{region}.chunk_{chunk_idx:05d}.tsv"
                pd.DataFrame(results, columns=["sequence_id", f"mfe_{region}"]).to_csv(ckpt_file, sep="\t", index=False)
                print(f"[INFO] wrote {ckpt_file}")
    else:
        print("[INFO] No pending RNAfold chunks. Using checkpointed results.")

    merged = regions[["sequence_id","marker"]].drop_duplicates().copy()
    for region in REGIONS:
        reg_files = sorted(ckpt_dir.glob(f"{region}.chunk_*.tsv"))
        if not reg_files:
            raise SystemExit(f"No checkpoint files found for region {region}")
        dfs = [pd.read_csv(f, sep="\t") for f in reg_files]
        reg = pd.concat(dfs, ignore_index=True).drop_duplicates("sequence_id", keep="last")
        merged = merged.merge(reg, on="sequence_id", how="left")

    merged.to_csv(outdir / "rnafold_features.tsv", sep="\t", index=False)
    (outdir / "rnafold_run_meta.json").write_text(json.dumps({
        "rnafold_bin": rnafold_bin,
        "threads": args.threads,
        "chunk_size": args.chunk_size,
        "resume": args.resume
    }, indent=2), encoding="utf-8")

    print(f"Wrote {outdir / 'rnafold_features.tsv'}")
    print(f"Wrote {outdir / 'rnafold_run_meta.json'}")

if __name__ == "__main__":
    main()
