#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, gzip, re
from pathlib import Path

IUPAC = {
    "A":"A","C":"C","G":"G","T":"T",
    "R":"[AG]","Y":"[CT]","S":"[GC]","W":"[AT]",
    "K":"[GT]","M":"[AC]",
    "B":"[CGT]","D":"[AGT]","H":"[ACT]","V":"[ACG]",
    "N":"[ACGT]"
}
RC = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")

def iupac_to_regex(seq: str) -> re.Pattern:
    seq = seq.upper()
    pat = "".join(IUPAC.get(ch, ch) for ch in seq)
    return re.compile("^" + pat)

def revcomp_iupac(seq: str) -> str:
    return seq.upper().translate(RC)[::-1]

def read_fasta(fp: Path):
    opener = gzip.open if str(fp).endswith(".gz") else open
    name = None
    buf = []
    with opener(fp, "rt", encoding="utf-8") as f:
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--head", type=int, default=30)
    ap.add_argument("--tail", type=int, default=30)
    args = ap.parse_args()

    # 你的引物（含IUPAC）
    primers = {
        "12S_F": "GGGATTAGATACCCCACTATGCYTA",
        "12S_R": "GAGGGTGACGGGCGGTGT",
        "16S_F": "ACCAAAAACATCACCTCYAGCAT",
        "16S_R": "AATAGGATTGCGCTGTTATCCCTA",
    }

    # 头部：检测 F 是否在序列开头窗口出现（从窗口起点开始匹配）
    head_re = {k: iupac_to_regex(v) for k, v in primers.items()}

    # 尾部：检测 R 的反向互补是否出现在序列尾窗口的起点（同样从窗口起点开始匹配）
    tail_re = {k: iupac_to_regex(revcomp_iupac(v)) for k, v in primers.items()}

    total = 0
    hit_head = {k: 0 for k in primers}
    hit_tail = {k: 0 for k in primers}

    # 另外做个“头部20mer是否高度一致”的指纹：如果引物没去掉，头部会高度统一
    from collections import Counter
    c20 = Counter()

    for _, seq in read_fasta(Path(args.fasta)):
        total += 1
        h = seq[:args.head]
        t = seq[-args.tail:] if len(seq) >= args.tail else seq

        if len(h) >= 20:
            c20[h[:20]] += 1

        for k, rgx in head_re.items():
            if rgx.match(h):
                hit_head[k] += 1
        for k, rgx in tail_re.items():
            if rgx.match(t):
                hit_tail[k] += 1

    print(f"[INFO] total_seqs = {total}")
    print(f"[INFO] head_window = {args.head}  tail_window = {args.tail}")

    print("\n[HEAD primer-like matches at window start]")
    for k in primers:
        frac = hit_head[k] / total if total else 0
        print(f"{k}\t{hit_head[k]}\t{frac:.6f}")

    print("\n[TAIL primer-like matches (revcomp) at window start]")
    for k in primers:
        frac = hit_tail[k] / total if total else 0
        print(f"{k}\t{hit_tail[k]}\t{frac:.6f}")

    print("\n[Fingerprint: top head20 frequency]")
    if total and c20:
        top20, cnt = c20.most_common(1)[0]
        print(f"top20_count={cnt}\ttop20_frac={cnt/total:.6f}\ttop20={top20}")
        # 也给前5
        print("top5:")
        for mer, c in c20.most_common(5):
            print(f"{c}\t{c/total:.6f}\t{mer}")

if __name__ == "__main__":
    main()
