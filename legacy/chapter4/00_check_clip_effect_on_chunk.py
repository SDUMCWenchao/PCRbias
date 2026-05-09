#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, gzip, csv
from pathlib import Path
from collections import Counter

# 直接复用 02e 的剪引物逻辑（轻量复制）
import re
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

def build_primer_matchers():
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
    tail = {
        "12S_R": (len(primers["12S_R"]), compile_iupac_full(revcomp_iupac(primers["12S_R"]))),
        "16S_R": (len(primers["16S_R"]), compile_iupac_full(revcomp_iupac(primers["16S_R"]))),
    }
    return head, tail

def clip_primers(seq: str, head, tail):
    clip_h = 0; hit_h = None
    for name,(plen,rgx) in head.items():
        if len(seq) >= plen and rgx.match(seq[:plen]):
            if plen > clip_h:
                clip_h = plen; hit_h = name
    if clip_h: seq = seq[clip_h:]
    clip_t = 0; hit_t = None
    for name,(plen,rgx) in tail.items():
        if len(seq) >= plen and rgx.match(seq[-plen:]):
            if plen > clip_t:
                clip_t = plen; hit_t = name
    if clip_t: seq = seq[:-clip_t]
    return seq, hit_h, hit_t

def read_fasta(fp: Path):
    opener = gzip.open if str(fp).endswith(".gz") else open
    name=None; buf=[]
    with opener(fp,"rt",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf).upper()
                name=line[1:].split()[0]; buf=[]
            else:
                buf.append(line)
        if name is not None:
            yield name, "".join(buf).upper()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_fasta", required=True)
    ap.add_argument("--sample_n", type=int, default=20000)
    ap.add_argument("--head_win", type=int, default=30)
    args = ap.parse_args()

    head, tail = build_primer_matchers()

    total = 0
    pre = Counter()
    post = Counter()
    top20_pre = Counter()
    top20_post = Counter()

    for _, seq0 in read_fasta(Path(args.chunk_fasta)):
        total += 1
        if total > args.sample_n:
            break

        h0 = seq0[:args.head_win]
        if len(h0) >= 22 and head["16S_F"][1].match(seq0[:head["16S_F"][0]]): pre["16S_F"] += 1
        if len(h0) >= 25 and head["12S_F"][1].match(seq0[:head["12S_F"][0]]): pre["12S_F"] += 1
        if len(seq0) >= 20: top20_pre[seq0[:20]] += 1

        seq, _, _ = clip_primers(seq0, head, tail)
        if len(seq) >= head["16S_F"][0] and head["16S_F"][1].match(seq[:head["16S_F"][0]]): post["16S_F"] += 1
        if len(seq) >= head["12S_F"][0] and head["12S_F"][1].match(seq[:head["12S_F"][0]]): post["12S_F"] += 1
        if len(seq) >= 20: top20_post[seq[:20]] += 1

    print(f"[INFO] sampled={total}")
    print("[PRE ] head primer exact-at-start:", {k: pre[k]/total for k in ["16S_F","12S_F"]})
    print("[POST] head primer exact-at-start:", {k: post[k]/total for k in ["16S_F","12S_F"]})

    if top20_pre:
        m, c = top20_pre.most_common(1)[0]
        print(f"[PRE ] top20_frac={c/total:.6f} top20={m}")
    if top20_post:
        m, c = top20_post.most_common(1)[0]
        print(f"[POST] top20_frac={c/total:.6f} top20={m}")

if __name__ == "__main__":
    main()
