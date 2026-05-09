#!/usr/bin/env python3
from pathlib import Path
import argparse, multiprocessing as mp, subprocess, tempfile, pandas as pd, re

REGIONS = ["global","head","mid1","mid2","mid3","tail"]

def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def run_rnafold_chunk(args):
    region, chunk = args
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".fa") as tmp:
        tmp_path = Path(tmp.name)
        for sid, seq in chunk:
            tmp.write(f">{sid}\n{seq}\n")
    cmd = ["RNAfold", "--noPS", str(tmp_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    tmp_path.unlink(missing_ok=True)
    results = {}
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
        results[sid] = mfe
        i += 3
    return region, results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence-regions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threads", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--chunk-size", type=int, default=400)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    regions = pd.read_csv(args.sequence_regions, sep="\t", dtype=str).fillna("")

    jobs = []
    for region in REGIONS:
        seq_pairs = []
        for _, r in regions.iterrows():
            sid = r["sequence_id"]
            seq = r["sequence"] if region == "global" else r[region]
            if str(seq).strip():
                seq_pairs.append((sid, seq))
        for chunk in chunk_list(seq_pairs, args.chunk_size):
            jobs.append((region, chunk))

    region_results = {r:{} for r in REGIONS}
    with mp.Pool(processes=args.threads) as pool:
        for region, result in pool.imap_unordered(run_rnafold_chunk, jobs, chunksize=1):
            region_results[region].update(result)

    rows = []
    for _, r in regions.iterrows():
        sid = r["sequence_id"]
        row = {"sequence_id": sid, "marker": r["marker"]}
        for region in REGIONS:
            row[f"mfe_{region}"] = region_results.get(region, {}).get(sid, None)
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "rnafold_features.tsv", sep="\t", index=False)
    print(outdir / "rnafold_features.tsv")

if __name__ == "__main__":
    main()
