#!/usr/bin/env python3
from pathlib import Path
import argparse, multiprocessing as mp
from collections import Counter, defaultdict
import pandas as pd

REGION_SET = ["global","head","tail","mid1","mid2","mid3"]
SELECTED = {}

def kmers(seq, k):
    seq = str(seq).upper()
    for i in range(0, max(0, len(seq)-k+1)):
        yield seq[i:i+k]

def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def count_chunk(records, region, k):
    counter = Counter()
    for r in records:
        seq = r["sequence"] if region == "global" else r[region]
        counter.update(kmers(seq, k))
    return counter

def process_region_k_job(args):
    records, region, k = args
    return region, k, count_chunk(records, region, k)

def init_selected(sel):
    global SELECTED
    SELECTED = sel

def process_feature_row(r):
    row = {"sequence_id": r["sequence_id"], "marker": r["marker"]}
    seq_by_region = {region: (r["sequence"] if region == "global" else r[region]).upper() for region in REGION_SET}
    for region, motifs in SELECTED.items():
        seq = seq_by_region[region]
        by_k = defaultdict(Counter)
        for k in sorted(set(len(m) for m in motifs)):
            by_k[k] = Counter(kmers(seq, k))
        for motif in motifs:
            count = by_k[len(motif)].get(motif, 0)
            safe = motif.replace(";", "_")
            row[f"motif_{safe}_count_{region}"] = count
            row[f"motif_{safe}_present_{region}"] = int(count > 0)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence-regions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k-values", nargs="*", type=int, default=[4,5])
    ap.add_argument("--top-per-region", type=int, default=20)
    ap.add_argument("--threads", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--count-chunk-size", type=int, default=1000)
    ap.add_argument("--feature-chunksize", type=int, default=50)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.sequence_regions, sep="\t", dtype=str).fillna("")
    records = df.to_dict("records")

    jobs = []
    for region in REGION_SET:
        for k in args.k_values:
            for chunk in chunk_list(records, args.count_chunk_size):
                jobs.append((chunk, region, k))

    merged_counts = defaultdict(Counter)
    with mp.Pool(processes=args.threads) as pool:
        for region, k, counter in pool.imap_unordered(process_region_k_job, jobs, chunksize=1):
            merged_counts[(region, k)].update(counter)

    selected_rows = []
    selected = defaultdict(list)
    for (region, k), counter in merged_counts.items():
        for motif, count in counter.most_common(args.top_per_region):
            selected_rows.append({"region": region, "k": k, "motif": motif, "global_occurrence_count": count})
            selected[region].append(motif)

    pd.DataFrame(selected_rows).drop_duplicates(["region","k","motif"]).to_csv(outdir / "motif_candidate_catalog.tsv", sep="\t", index=False)

    with mp.Pool(processes=args.threads, initializer=init_selected, initargs=(dict(selected),)) as pool:
        rows = list(pool.imap(process_feature_row, records, chunksize=args.feature_chunksize))

    pd.DataFrame(rows).to_csv(outdir / "motif_candidate_features.tsv", sep="\t", index=False)
    print(outdir / "motif_candidate_catalog.tsv")
    print(outdir / "motif_candidate_features.tsv")

if __name__ == "__main__":
    main()
