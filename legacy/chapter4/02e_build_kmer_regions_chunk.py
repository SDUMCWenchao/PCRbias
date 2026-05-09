#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, gzip, json, re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional

# ---------- JSON (fast if orjson available) ----------
try:
    import orjson  # type: ignore
    def dumps(obj): return orjson.dumps(obj).decode("utf-8")
except Exception:
    def dumps(obj): return json.dumps(obj, separators=(",", ":"))

# ---------- FASTA reader ----------
def read_fasta(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    name = None
    buf = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf).upper()
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if name is not None:
            yield name, "".join(buf).upper()

# ---------- IUPAC primer regex ----------
IUPAC = {
    "A":"A","C":"C","G":"G","T":"T",
    "R":"[AG]","Y":"[CT]","S":"[GC]","W":"[AT]",
    "K":"[GT]","M":"[AC]",
    "B":"[CGT]","D":"[AGT]","H":"[ACT]","V":"[ACG]",
    "N":"[ACGT]"
}
RC = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")

def revcomp_iupac(seq: str) -> str:
    return seq.upper().translate(RC)[::-1]

def compile_iupac_full(seq: str) -> re.Pattern:
    pat = "".join(IUPAC.get(ch, ch) for ch in seq.upper())
    return re.compile("^" + pat + "$")

def build_primer_matchers() -> Tuple[Dict[str, Tuple[int,re.Pattern]], Dict[str, Tuple[int,re.Pattern]]]:
    # your primers (IUPAC allowed)
    primers = {
        "12S_F": "GGGATTAGATACCCCACTATGCYTA",
        "12S_R": "GAGGGTGACGGGCGGTGT",
        "16S_F": "ACCAAAAACATCACCTCYAGCAT",
        "16S_R": "AATAGGATTGCGCTGTTATCCCTA",
    }
    head = {
        "12S_F": (len(primers["12S_F"]), compile_iupac_full(primers["12S_F"])),
        "16S_F": (len(primers["16S_F"]), compile_iupac_full(primers["16S_F"])),
    }
    # tail: match reverse primer's REVCOMP at sequence end
    tail = {
        "12S_R": (len(primers["12S_R"]), compile_iupac_full(revcomp_iupac(primers["12S_R"]))),
        "16S_R": (len(primers["16S_R"]), compile_iupac_full(revcomp_iupac(primers["16S_R"]))),
    }
    return head, tail

def clip_primers(seq: str,
                 head_matchers: Dict[str, Tuple[int,re.Pattern]],
                 tail_matchers: Dict[str, Tuple[int,re.Pattern]]) -> Tuple[str, Optional[str], Optional[str], int, int]:
    """
    Strict clipping:
      - head: only if primer matches EXACTLY at position 0 (IUPAC-aware)
      - tail: only if revcomp(reverse primer) matches EXACTLY at the end
    If multiple head primers match (rare), clip the longer one.
    """
    clip_h = 0
    hit_h = None
    # head
    for name, (plen, rgx) in head_matchers.items():
        if len(seq) >= plen and rgx.match(seq[:plen]):
            if plen > clip_h:
                clip_h = plen
                hit_h = name
    if clip_h > 0:
        seq = seq[clip_h:]

    clip_t = 0
    hit_t = None
    # tail
    for name, (plen, rgx) in tail_matchers.items():
        if len(seq) >= plen and rgx.match(seq[-plen:]):
            if plen > clip_t:
                clip_t = plen
                hit_t = name
    if clip_t > 0:
        seq = seq[:-clip_t]

    return seq, hit_h, hit_t, clip_h, clip_t

# ---------- k-mer scanning (rolling 2-bit) ----------
BASE2 = {"A":0, "C":1, "G":2, "T":3}

def load_union_vocab_as_int(kmer_union_tsv: Path, min_k: int, max_k: int):
    vocab_int = defaultdict(set)     # k -> {int_code}
    int2str   = defaultdict(dict)    # k -> {int_code: kmer_str}
    with kmer_union_tsv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            k = int(row["k"])
            if k < min_k or k > max_k:
                continue
            mer = row["kmer"].upper()
            code = 0
            ok = True
            for ch in mer:
                v = BASE2.get(ch)
                if v is None:
                    ok = False
                    break
                code = (code << 2) | v
            if not ok:
                continue
            vocab_int[k].add(code)
            int2str[k][code] = mer
    ks = sorted(vocab_int.keys())
    if not ks:
        raise ValueError("Union vocab is empty in selected k range.")
    return vocab_int, int2str, ks

def region_for_start(start: int, k: int, L: int, head_win: int, tail_win: int, mid_bins: int) -> Optional[str]:
    """
    Strict regions (boundary-crossing kmers are dropped):
      head{head_win}: start+k <= head_win
      tail{tail_win}: start >= L-tail_win
      mid0..mid{mid_bins-1}: kmer fully inside [head_win, L-tail_win)
    """
    if L <= 0:
        return None
    head_win = max(0, head_win)
    tail_win = max(0, tail_win)

    tail_start = L - tail_win
    if tail_start < 0:
        tail_start = 0

    # head
    if start + k <= head_win:
        return f"head{head_win}"

    # tail
    if start >= tail_start:
        return f"tail{tail_win}"

    # mid strict
    if start < head_win:
        return None
    if start + k > tail_start:
        return None

    mid_len = tail_start - head_win
    npos_mid = mid_len - k + 1
    if npos_mid <= 0:
        return None
    j = start - head_win
    if j < 0 or j >= npos_mid:
        return None
    b = int(j * mid_bins / npos_mid)
    if b >= mid_bins:
        b = mid_bins - 1
    return f"mid{b}"

def scan_seq_kmers(seq: str, k: int, vocab: set, int2str_k: Dict[int, str],
                   head_win: int, tail_win: int, mid_bins: int,
                   x_mode: str, feats: Dict[str, int]) -> None:
    L = len(seq)
    if L < k:
        return
    mask = (1 << (2*k)) - 1
    code = 0
    run = 0

    for i, ch in enumerate(seq):
        v = BASE2.get(ch)
        if v is None:
            code = 0
            run = 0
            continue
        code = ((code << 2) | v) & mask
        run += 1
        if run < k:
            continue
        start = i - k + 1

        reg = region_for_start(start, k, L, head_win, tail_win, mid_bins)
        if reg is None:
            continue
        if code not in vocab:
            continue
        mer = int2str_k.get(code)
        if mer is None:
            continue

        key = f"k{k}_{reg}_{mer}"
        if x_mode == "presence":
            feats[key] = 1
        else:
            feats[key] = feats.get(key, 0) + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_fasta", required=True)
    ap.add_argument("--kmer_union_tsv", required=True)
    ap.add_argument("--out_tsv_gz", required=True)

    ap.add_argument("--min_k", type=int, default=6)
    ap.add_argument("--max_k", type=int, default=8)

    ap.add_argument("--head_win", type=int, default=30)
    ap.add_argument("--tail_win", type=int, default=30)
    ap.add_argument("--mid_bins", type=int, default=3)

    ap.add_argument("--x_mode", choices=["count", "presence"], default="count")

    ap.add_argument("--clip_primers", action="store_true", default=True)
    ap.add_argument("--no_clip_primers", dest="clip_primers", action="store_false")
    args = ap.parse_args()

    vocab_int, int2str, ks = load_union_vocab_as_int(Path(args.kmer_union_tsv), args.min_k, args.max_k)
    head_matchers, tail_matchers = build_primer_matchers()

    out_fp = Path(args.out_tsv_gz)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    # stats
    n = 0
    clip_h_cnt = defaultdict(int)
    clip_t_cnt = defaultdict(int)

    with gzip.open(out_fp, "wt", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        # keep backward-compatible columns + add trim info
        w.writerow(["Seq_ID", "seq_len", "clip_head", "clip_tail", "kmer_json"])

        for sid, seq0 in read_fasta(Path(args.chunk_fasta)):
            seq = seq0
            hit_h = None
            hit_t = None

            if args.clip_primers:
                seq, hit_h, hit_t, _, _ = clip_primers(seq, head_matchers, tail_matchers)
                if hit_h:
                    clip_h_cnt[hit_h] += 1
                if hit_t:
                    clip_t_cnt[hit_t] += 1

            feats: Dict[str, int] = {}
            for k in ks:
                scan_seq_kmers(
                    seq=seq,
                    k=k,
                    vocab=vocab_int[k],
                    int2str_k=int2str[k],
                    head_win=args.head_win,
                    tail_win=args.tail_win,
                    mid_bins=args.mid_bins,
                    x_mode=args.x_mode,
                    feats=feats,
                )

            w.writerow([sid, len(seq), hit_h or "NA", hit_t or "NA", dumps(feats)])
            n += 1

    # print chunk-level trimming stats (goes to slurm log)
    print(f"[DONE] {out_fp} seqs={n} regions=head{args.head_win}, mid0..mid{args.mid_bins-1}, tail{args.tail_win}")
    if args.clip_primers:
        print("[CLIP] head:", dict(clip_h_cnt))
        print("[CLIP] tail:", dict(clip_t_cnt))

if __name__ == "__main__":
    main()
