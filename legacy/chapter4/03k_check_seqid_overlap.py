#!/usr/bin/env python3
import gzip, csv, sqlite3
from pathlib import Path

PROJECT="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
weaver=Path(PROJECT)/"analysis_results/03_DataWeaver"
feat=Path(PROJECT)/"analysis_results/02_Features"

# 你把这里改成你实际的 sparse kmer 目录（如果不是这个）
kmer_dir_candidates=[
    feat/"kmer_sparse_chunks",
    feat/"kmer_sparse",
    feat/"kmer_chunks",
]
kmer_dir=None
for c in kmer_dir_candidates:
    if c.exists():
        kmer_dir=c
        break
if kmer_dir is None:
    raise SystemExit("Cannot find kmer sparse dir. Please set it in script.")

chunk = sorted(kmer_dir.glob("chunk_*.kmer.tsv.gz"))[0]
N=2000  # 抽样多少条 Seq_ID 来测 overlap

seq_ids=[]
with gzip.open(chunk, "rt", encoding="utf-8") as f:
    r=csv.DictReader(f, delimiter="\t")
    col = "Seq_ID" if "Seq_ID" in r.fieldnames else "seq_id"
    for i,row in enumerate(r):
        seq_ids.append(row[col])
        if len(seq_ids)>=N: break

totals_tsv=weaver/"sample_totals.tsv"
counts_dir=weaver/"sample_counts"

samples=[]
with totals_tsv.open("r", encoding="utf-8") as f:
    rr=csv.DictReader(f, delimiter="\t")
    for row in rr:
        samples.append(row["file_id"])

def fetch_present(dbp: Path, ids):
    conn=sqlite3.connect(str(dbp))
    cur=conn.cursor()
    present=0
    # 分批避免 SQL 参数限制
    step=800
    for i in range(0, len(ids), step):
        batch=ids[i:i+step]
        q="SELECT COUNT(*) FROM counts WHERE seq_id IN (" + ",".join(["?"]*len(batch)) + ")"
        present += cur.execute(q, batch).fetchone()[0]
    conn.close()
    return present

print(f"[INFO] probe chunk: {chunk}")
print(f"[INFO] probe Seq_IDs: {len(seq_ids)}")

for sid in samples:
    dbp=counts_dir/f"{sid}.sqlite"
    if not dbp.exists():
        print(sid, "MISSING_DB")
        continue
    p=fetch_present(dbp, seq_ids)
    print(f"{sid}\tpresent_in_{len(seq_ids)}\t{p}")
