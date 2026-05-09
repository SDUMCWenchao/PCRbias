#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2A worker: compute full core features for each Seq_ID in a fasta chunk,
and export:
  1) core feature table: core_chunks/<chunk>.core.tsv.gz  (NO Sequence column)
  2) sparse kmer table:  kmer_sparse_chunks/<chunk>.kmer.tsv.gz (json maps; later filtered)

Features included (v1):
- composition: base fractions, GC/AT, skews, end bases, head/tail GC
- dinucleotide freq: 16 di frequencies + CpG/UpA + CpG odds ratio
- complexity: entropy1, LZ, dust_score, linguistic complexity (k=1..8 summary)
- repeats: homopolymer max A/C/G/T + run_ge4/5/6 + max_direp + max_trirep + tandem_density
- palindromes: pal_count_len>=6/8/10 + pal_maxlen + end_complement_score
- G4: g4_count + G_island_max + G_island_n3
- ZDNA alt: longest alternating R/Y run
- structure: RNAfold MFE + dot-bracket derived stats (pair_frac, stem_max, loop_max, hairpin_count)
- Tm: try Biopython NN; fallback Wallace. Also tm_w20_max and gc_w20_max (W=20)
- primer (12S & 16S): best/end-restricted match, mismatch totals, 3' last5 mismatches,
  positions, is_end flags; local GC; local MFE for end-matches only (to reduce cost)

K-mer handling:
- We DO NOT expand kmer columns here to avoid explosion.
- Output sparse maps for k=1..8 in:
    kmer_all_json, kmer_head_json, kmer_tail_json
  (each is {"k=1":{"A":..}, "k=2":{...}, ...} but stored as a compact json string)
- Also output summary metrics per k: distinct, entropy_kmer, gini, top10_cov (for all/global)

Temp cleanup:
- RNAfold is called with --noPS and a per-task temp dir (removed on exit).

Requires:
- RNAfold in PATH (ViennaRNA)
Optional:
- Biopython (for Tm_NN). If absent, uses Wallace rule.
"""

from __future__ import annotations
import argparse
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# ------------------ FASTA ------------------

def fasta_iter(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (header_without_>, sequence). Assumes one-line sequence or multi-line; joins."""
    header = None
    seq_parts = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts).upper()
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts).upper()

# ------------------ IUPAC ------------------

IUPAC = {
    "A": {"A"}, "C": {"C"}, "G": {"G"}, "T": {"T"},
    "R": {"A","G"}, "Y": {"C","T"}, "S": {"G","C"}, "W": {"A","T"},
    "K": {"G","T"}, "M": {"A","C"},
    "B": {"C","G","T"}, "D": {"A","G","T"}, "H": {"A","C","T"}, "V": {"A","C","G"},
    "N": {"A","C","G","T"},
}
IUPAC_COMP = {
    "A":"T","T":"A","C":"G","G":"C",
    "R":"Y","Y":"R","S":"S","W":"W",
    "K":"M","M":"K",
    "B":"V","V":"B","D":"H","H":"D",
    "N":"N"
}

def revcomp_iupac(primer: str) -> str:
    primer = primer.upper().replace("U","T")
    return "".join(IUPAC_COMP.get(b,"N") for b in primer[::-1])

def mismatch_count_iupac(primer_pat: str, target: str) -> int:
    """Count mismatches aligning primer pattern (IUPAC) to target (same length)."""
    mm = 0
    for p, t in zip(primer_pat, target):
        if t not in IUPAC.get(p, {"A","C","G","T"}):
            mm += 1
    return mm

# ------------------ basic stats ------------------

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def shannon_entropy_from_counts(counts: Dict[str,int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0: 
            continue
        p = c / total
        ent -= p * math.log2(p)
    return ent

def gini_from_counts(counts: List[int]) -> float:
    """Gini of nonnegative counts."""
    if not counts:
        return 0.0
    x = sorted([c for c in counts if c >= 0])
    n = len(x)
    s = sum(x)
    if s == 0:
        return 0.0
    cum = 0
    for i, v in enumerate(x, start=1):
        cum += i * v
    return (2*cum)/(n*s) - (n+1)/n

# ------------------ kmer ------------------

def kmer_counts(seq: str, k: int) -> Counter:
    c = Counter()
    n = len(seq)
    if n < k or k <= 0:
        return c
    for i in range(n - k + 1):
        kmer = seq[i:i+k]
        if "N" in kmer:
            continue
        c[kmer] += 1
    return c

def kmer_entropy(counter: Counter) -> float:
    return shannon_entropy_from_counts(counter)

def topk_coverage(counter: Counter, topk: int = 10) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    s = sum(v for _, v in counter.most_common(topk))
    return s / total

# ------------------ repeats ------------------

def max_homopolymer(seq: str, base: str) -> int:
    base = base.upper()
    best = 0
    cur = 0
    for ch in seq:
        if ch == base:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

def count_homopolymer_ge(seq: str, threshold: int) -> int:
    """Count runs (any base) with length >= threshold."""
    if threshold <= 1:
        return 0
    cnt = 0
    cur_base = None
    cur = 0
    for ch in seq + "X":  # sentinel
        if ch == cur_base:
            cur += 1
        else:
            if cur_base is not None and cur >= threshold:
                cnt += 1
            cur_base = ch
            cur = 1
    return cnt

def max_tandem_repeat(seq: str, unit_len: int) -> Tuple[int, str]:
    """
    Return (max_reps, motif) for exact tandem repeats of given unit length.
    reps count is number of motif repeats (>=2).
    """
    n = len(seq)
    best_reps = 1
    best_motif = ""
    if n < unit_len * 2:
        return 1, ""
    for i in range(0, n - unit_len*2 + 1):
        motif = seq[i:i+unit_len]
        if "N" in motif:
            continue
        reps = 1
        j = i + unit_len
        while j + unit_len <= n and seq[j:j+unit_len] == motif:
            reps += 1
            j += unit_len
        if reps > best_reps:
            best_reps = reps
            best_motif = motif
    return best_reps, best_motif

def tandem_density_approx(seq: str) -> float:
    """
    Approx coverage by:
      - homopolymer runs >=4
      - dinuc repeats >=3 units
      - trinuc repeats >=3 units
    Simple non-overlap approximation.
    """
    n = len(seq)
    if n == 0:
        return 0.0
    covered = [False]*n

    # homopolymer >=4
    i = 0
    while i < n:
        j = i+1
        while j < n and seq[j] == seq[i]:
            j += 1
        if j - i >= 4:
            for k in range(i, j):
                covered[k] = True
        i = j

    # dinuc repeats >=3
    unit = 2
    for i in range(0, n - unit*3 + 1):
        motif = seq[i:i+unit]
        if "N" in motif:
            continue
        reps = 1
        j = i + unit
        while j + unit <= n and seq[j:j+unit] == motif:
            reps += 1
            j += unit
        if reps >= 3:
            for k in range(i, i + reps*unit):
                covered[k] = True

    # trinuc repeats >=3
    unit = 3
    for i in range(0, n - unit*3 + 1):
        motif = seq[i:i+unit]
        if "N" in motif:
            continue
        reps = 1
        j = i + unit
        while j + unit <= n and seq[j:j+unit] == motif:
            reps += 1
            j += unit
        if reps >= 3:
            for k in range(i, i + reps*unit):
                covered[k] = True

    return sum(1 for x in covered if x) / n

# ------------------ palindrome / complement ------------------

def comp_base(b: str) -> str:
    return {"A":"T","T":"A","C":"G","G":"C"}.get(b, "N")

def is_complement(a: str, b: str) -> bool:
    return comp_base(a) == b

def pal_stats(seq: str, minlens=(6,8,10)) -> Tuple[Dict[int,int], int]:
    """
    Count palindromic substrings (reverse-complement palindromes) with length >= thresholds.
    Return counts_by_threshold and max_pal_len.
    Brute expand around centers; OK for short sequences.
    """
    n = len(seq)
    max_len = 0
    counts = {L: 0 for L in minlens}

    # even and odd centers
    def expand(l: int, r: int):
        nonlocal max_len
        while l >= 0 and r < n and is_complement(seq[l], seq[r]):
            pal_len = r - l + 1
            max_len = max(max_len, pal_len)
            for L in minlens:
                if pal_len >= L:
                    counts[L] += 1
            l -= 1
            r += 1

    for c in range(n):
        # odd center (l=c-1,r=c+1) doesn't match complement of itself generally, skip
        # even center
        expand(c, c+1)

    return counts, max_len

def end_complement_score(seq: str, w: int = 30) -> float:
    """
    Complementarity between 5' head and 3' tail windows:
    compare head[i] with tail_revcomp[i] match fraction.
    """
    n = len(seq)
    if n == 0:
        return 0.0
    w = min(w, n)
    head = seq[:w]
    tail = seq[-w:]
    tail_rc = "".join(comp_base(b) for b in tail[::-1])
    matches = sum(1 for a,b in zip(head, tail_rc) if a == b and a in "ACGT")
    denom = sum(1 for a,b in zip(head, tail_rc) if a in "ACGT" and b in "ACGT")
    return safe_div(matches, denom)

# ------------------ G4 / ZDNA ------------------

G4_RE = re.compile(r"(G{3,}[ACGTN]{1,7}){3}G{3,}", re.IGNORECASE)

def g4_count(seq: str) -> int:
    return len(list(G4_RE.finditer(seq)))

def g_island_stats(seq: str) -> Tuple[int,int]:
    """Return (max_G_island_len, count_islands_len>=3)."""
    best = 0
    cnt3 = 0
    cur = 0
    for ch in seq + "X":
        if ch == "G":
            cur += 1
        else:
            if cur > 0:
                best = max(best, cur)
                if cur >= 3:
                    cnt3 += 1
            cur = 0
    return best, cnt3

def zdna_alt_max(seq: str) -> int:
    """
    Longest run of alternating purine/pyrimidine (R/Y).
    R: A,G ; Y: C,T
    """
    def cls(b: str) -> Optional[str]:
        if b in "AG": return "R"
        if b in "CT": return "Y"
        return None

    best = 0
    cur = 0
    prev = None
    for b in seq:
        c = cls(b)
        if c is None:
            cur = 0
            prev = None
            continue
        if prev is None:
            cur = 1
        else:
            cur = cur + 1 if c != prev else 1
        prev = c
        best = max(best, cur)
    return best

# ------------------ DUST (simple) ------------------

def dust_score(seq: str, window: int = 64) -> float:
    """
    Simple DUST-like score based on trinucleotide complexity in a window.
    For short reads, compute on whole seq if len < window.
    This is a lightweight approximation (useful for ranking).
    """
    s = seq
    n = len(s)
    if n < 3:
        return 0.0
    win = min(window, n)
    # score each window, take max
    best = 0.0
    for i in range(0, n - win + 1) if n >= win else [0]:
        sub = s[i:i+win]
        tri = Counter(sub[j:j+3] for j in range(0, len(sub)-2) if "N" not in sub[j:j+3])
        # DUST score ~ sum c*(c-1)/2 normalized
        score = sum(c*(c-1)/2 for c in tri.values())
        best = max(best, score)
    # normalize by window length
    return best / win

def linguistic_complexity(seq: str, kmax: int = 8) -> Dict[int,float]:
    """
    LC(k)= unique_kmers / min(4^k, len-k+1)
    """
    n = len(seq)
    out = {}
    for k in range(1, kmax+1):
        denom = min(4**k, max(0, n-k+1))
        if denom <= 0:
            out[k] = 0.0
            continue
        uniq = len(set(seq[i:i+k] for i in range(n-k+1) if "N" not in seq[i:i+k]))
        out[k] = uniq / denom
    return out

# ------------------ RNAfold ------------------

def run_rnafold(seq: str, tmpdir: Path) -> Tuple[float, str]:
    """
    Run RNAfold --noPS. Return (MFE, structure).
    Returns (nan,"") if fails.
    """
    # RNAfold expects RNA; for DNA we still feed A/C/G/T and it works as RNA model;
    # for relative comparisons this is acceptable.
    seq_in = seq.replace("T", "U")
    cmd = ["RNAfold", "--noPS"]
    try:
        p = subprocess.run(
            cmd,
            input=(seq_in + "\n").encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(tmpdir),
            check=False,
        )
        out = p.stdout.decode(errors="replace").strip().splitlines()
        if len(out) < 2:
            return float("nan"), ""
        # second line: structure + ( -12.30)
        line = out[1].strip()
        # structure is first token
        parts = line.split()
        struct = parts[0]
        mfe = float("nan")
        m = re.search(r"\(([-0-9\.]+)\)", line)
        if m:
            mfe = float(m.group(1))
        return mfe, struct
    except Exception:
        return float("nan"), ""

def structure_stats(struct: str) -> Tuple[float,int,int,int]:
    """
    From dot-bracket:
      pair_frac, stem_max (max consecutive paired positions), loop_max (max consecutive '.'),
      hairpin_count (pairs that directly enclose only dots)
    """
    if not struct:
        return 0.0, 0, 0, 0
    n = len(struct)
    paired = [1 if ch in "()"
              else 0 for ch in struct]
    pair_frac = sum(paired)/n if n else 0.0

    # stem_max: longest run of paired positions
    stem_max = 0
    cur = 0
    for v in paired + [0]:
        if v == 1:
            cur += 1
            stem_max = max(stem_max, cur)
        else:
            cur = 0

    # loop_max: longest run of '.'
    loop_max = 0
    cur = 0
    for ch in struct + "X":
        if ch == ".":
            cur += 1
            loop_max = max(loop_max, cur)
        else:
            cur = 0

    # hairpin_count: count base pairs (i,j) where inside is only dots and no nested pairs
    stack = []
    hairpins = 0
    for idx, ch in enumerate(struct):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            if not stack:
                continue
            i = stack.pop()
            inside = struct[i+1:idx]
            if inside and all(c == "." for c in inside):
                hairpins += 1
    return pair_frac, stem_max, loop_max, hairpins

# ------------------ Tm ------------------

def tm_calc(seq: str) -> Tuple[float,str]:
    """
    Try Biopython NN Tm with common defaults; otherwise Wallace.
    Returns (tm, method).
    """
    s = seq.replace("N", "")
    if len(s) == 0:
        return 0.0, "none"
    try:
        from Bio.SeqUtils import MeltingTemp as mt  # type: ignore
        # common defaults: Na=50mM, dnac1=250nM, dnac2=0, saltcorr=5
        tm = mt.Tm_NN(s, Na=50, dnac1=250, dnac2=0, saltcorr=5)
        return float(tm), "biopython_Tm_NN"
    except Exception:
        # Wallace rule
        a = s.count("A")
        t = s.count("T")
        c = s.count("C")
        g = s.count("G")
        tm = 2*(a+t) + 4*(c+g)
        return float(tm), "wallace"

def tm_wmax(seq: str, w: int = 20) -> float:
    n = len(seq)
    if n < w:
        tm, _ = tm_calc(seq)
        return tm
    best = -1e9
    for i in range(0, n-w+1):
        tm, _ = tm_calc(seq[i:i+w])
        if tm > best:
            best = tm
    return best if best > -1e8 else 0.0

def gc_wmax(seq: str, w: int = 20) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    if n < w:
        gc = (seq.count("G")+seq.count("C"))/n
        return gc
    best = 0.0
    for i in range(0, n-w+1):
        sub = seq[i:i+w]
        gc = (sub.count("G")+sub.count("C"))/w
        if gc > best:
            best = gc
    return best

# ------------------ primer matching ------------------

def best_primer_match(primer_pat: str, target: str) -> Tuple[int,int]:
    """
    Return (best_mismatches, best_pos) scanning all positions.
    If target shorter: returns (len(primer), -1)
    """
    L = len(primer_pat)
    n = len(target)
    if n < L:
        return L, -1
    best_mm = L + 1
    best_pos = -1
    for i in range(0, n-L+1):
        mm = mismatch_count_iupac(primer_pat, target[i:i+L])
        if mm < best_mm:
            best_mm = mm
            best_pos = i
            if best_mm == 0:
                break
    return best_mm, best_pos

def end_primer_match(primer_pat: str, target: str, end: str, slack: int = 10) -> Tuple[int,int,bool]:
    """
    Best match restricted to end region.
    end='5' => positions [0..slack]
    end='3' => positions [n-L-slack .. n-L]
    Returns (best_mm, best_pos, found_flag)
    """
    L = len(primer_pat)
    n = len(target)
    if n < L:
        return L, -1, False
    best_mm = L + 1
    best_pos = -1
    if end == "5":
        start = 0
        stop = min(n-L, slack)
        rng = range(start, stop+1)
    else:
        start = max(0, n-L-slack)
        stop = n-L
        rng = range(start, stop+1)
    for i in rng:
        mm = mismatch_count_iupac(primer_pat, target[i:i+L])
        if mm < best_mm:
            best_mm = mm
            best_pos = i
    return best_mm, best_pos, (best_pos != -1)

def primer_mm_3p5(primer_pat: str, target_seg: str, orientation: str) -> int:
    """
    Count mismatches in last 5 bases of primer (3' end).
    orientation:
      'fwd' => 3' end corresponds to last 5 positions of primer match segment
      'rev_rc' => primer given is reverse primer; we match its reverse-complement to target.
                 3' end of original primer corresponds to first 5 positions of RC match.
    """
    L = len(primer_pat)
    if len(target_seg) != L:
        return L
    if L < 5:
        region = range(0, L)
    else:
        region = range(L-5, L) if orientation == "fwd" else range(0, 5)
    mm = 0
    for idx in region:
        p = primer_pat[idx]
        t = target_seg[idx]
        if t not in IUPAC.get(p, {"A","C","G","T"}):
            mm += 1
    return mm

def local_window(seq: str, pos: int, L: int, win: int = 60) -> str:
    """Extract a window of length win centered on the primer binding segment midpoint."""
    n = len(seq)
    if pos < 0:
        return ""
    mid = pos + L//2
    half = win//2
    a = max(0, mid-half)
    b = min(n, a+win)
    a = max(0, b-win)
    return seq[a:b]

# ------------------ main per-seq ------------------

PRIMERS = {
    "12S": {
        "F": "GGGATTAGATACCCCACTATGCYTA",
        "R": "GAGGGTGACGGGCGGTGT",
    },
    "16S": {
        "F": "ACCAAAAACATCACCTCYAGCAT",
        "R": "AATAGGATTGCGCTGTTATCCCTA",
    }
}

def compute_features(seq_id: str, seq: str, tmpdir: Path, head_tail_w: int = 30) -> Tuple[Dict[str,object], Dict[str,object]]:
    """
    Returns (core_features_dict, kmer_sparse_dict)
    kmer_sparse_dict contains json-able maps for later filtering.
    """
    s = seq.upper().replace("U","T")
    n = len(s)
    A = s.count("A"); C = s.count("C"); G = s.count("G"); T = s.count("T"); Nn = s.count("N")
    gc = safe_div(G+C, n)
    at = safe_div(A+T, n)
    gc_skew = safe_div(G-C, G+C)
    at_skew = safe_div(A-T, A+T)

    head = s[:min(head_tail_w, n)]
    tail = s[-min(head_tail_w, n):] if n else ""
    gc_head = safe_div(head.count("G")+head.count("C"), len(head)) if head else 0.0
    gc_tail = safe_div(tail.count("G")+tail.count("C"), len(tail)) if tail else 0.0

    # dinucleotides
    di = Counter()
    if n >= 2:
        for i in range(n-1):
            d = s[i:i+2]
            if "N" in d:
                continue
            di[d] += 1
    di_total = sum(di.values())
    di_freq = {f"feat_di_{k}": safe_div(v, di_total) for k, v in di.items()}
    # ensure all 16 exist
    for a in "ACGT":
        for b in "ACGT":
            key = f"feat_di_{a}{b}"
            di_freq.setdefault(key, 0.0)

    CpG = di.get("CG", 0)
    UpA = di.get("TA", 0)  # DNA equivalent of UpA
    # CpG odds ratio
    exp_cpg = (safe_div(C, n) * safe_div(G, n) * max(n-1, 0))
    or_cpg = safe_div(CpG, exp_cpg) if exp_cpg else 0.0

    # complexity
    ent1 = shannon_entropy_from_counts({"A":A,"C":C,"G":G,"T":T})
    # LZ via zlib length ratio
    import zlib
    comp = zlib.compress(s.encode("utf-8"))
    lz = safe_div(len(comp), max(1, n))

    dust = dust_score(s)
    ling = linguistic_complexity(s, kmax=8)

    # repeats
    runA = max_homopolymer(s, "A")
    runC = max_homopolymer(s, "C")
    runG = max_homopolymer(s, "G")
    runT = max_homopolymer(s, "T")
    run_ge4 = count_homopolymer_ge(s, 4)
    run_ge5 = count_homopolymer_ge(s, 5)
    run_ge6 = count_homopolymer_ge(s, 6)
    direp_reps, direp_motif = max_tandem_repeat(s, 2)
    trirep_reps, trirep_motif = max_tandem_repeat(s, 3)
    tr_density = tandem_density_approx(s)

    # palindromes + end complement
    pal_counts, pal_maxlen = pal_stats(s, minlens=(6,8,10))
    end_comp = end_complement_score(s, w=head_tail_w)

    # G4/ZDNA
    g4c = g4_count(s)
    g_is_max, g_is_n3 = g_island_stats(s)
    zdna = zdna_alt_max(s)

    # structure (RNAfold)
    mfe, struct = run_rnafold(s, tmpdir)
    pair_frac, stem_max, loop_max, hp_count = structure_stats(struct)

    # Tm
    tm, tm_method = tm_calc(s)
    tmw20 = tm_wmax(s, 20)
    gcw20 = gc_wmax(s, 20)

    # kmer sparse + summaries (k=1..8, all/head/tail)
    kmer_sparse_all = {}
    kmer_sparse_head = {}
    kmer_sparse_tail = {}

    ksum = {}  # summary metrics only (few columns)
    head30 = s[:min(30, n)]
    tail30 = s[-min(30, n):] if n else ""

    for k in range(1, 9):
        c_all = kmer_counts(s, k)
        c_h = kmer_counts(head30, k)
        c_t = kmer_counts(tail30, k)

        # sparse maps (raw counts). store only if non-empty
        if c_all:
            kmer_sparse_all[str(k)] = dict(c_all)
        if c_h:
            kmer_sparse_head[str(k)] = dict(c_h)
        if c_t:
            kmer_sparse_tail[str(k)] = dict(c_t)

        # summary metrics (global only)
        ksum[f"feat_k{k}_distinct"] = len(c_all)
        ksum[f"feat_k{k}_entropy"] = kmer_entropy(c_all) if c_all else 0.0
        ksum[f"feat_k{k}_gini"] = gini_from_counts(list(c_all.values())) if c_all else 0.0
        ksum[f"feat_k{k}_top10cov"] = topk_coverage(c_all, 10) if c_all else 0.0

    # primer features (12S, 16S): compute on this seq regardless of origin (later you可按locus筛用)
    primer_feats = {}
    for locus, pr in PRIMERS.items():
        fwd = pr["F"].upper().replace("U","T")
        rev = pr["R"].upper().replace("U","T")
        rev_rc = revcomp_iupac(rev)

        # best anywhere
        mm_f_best, pos_f_best = best_primer_match(fwd, s)
        mm_r_best, pos_r_best = best_primer_match(rev_rc, s)

        # end-restricted
        mm_f_end, pos_f_end, ok_f_end = end_primer_match(fwd, s, end="5", slack=10)
        mm_r_end, pos_r_end, ok_r_end = end_primer_match(rev_rc, s, end="3", slack=10)

        # 3' mismatches (for end matches primarily)
        mm_f_3p5 = ""
        mm_r_3p5 = ""
        if pos_f_best != -1:
            seg = s[pos_f_best:pos_f_best+len(fwd)]
            mm_f_3p5 = primer_mm_3p5(fwd, seg, "fwd")
        if pos_r_best != -1:
            seg = s[pos_r_best:pos_r_best+len(rev_rc)]
            mm_r_3p5 = primer_mm_3p5(rev_rc, seg, "rev_rc")

        # local GC on end-match segment
        f_gc = ""
        r_gc = ""
        if ok_f_end and pos_f_end != -1:
            seg = s[pos_f_end:pos_f_end+len(fwd)]
            f_gc = safe_div(seg.count("G")+seg.count("C"), len(seg))
        if ok_r_end and pos_r_end != -1:
            seg = s[pos_r_end:pos_r_end+len(rev_rc)]
            r_gc = safe_div(seg.count("G")+seg.count("C"), len(seg))

        # local MFE for end matches only (reduce RNAfold calls)
        f_mfe = ""
        r_mfe = ""
        if ok_f_end and pos_f_end != -1:
            win = local_window(s, pos_f_end, len(fwd), win=60)
            if win:
                f_mfe, _ = run_rnafold(win, tmpdir)
        if ok_r_end and pos_r_end != -1:
            win = local_window(s, pos_r_end, len(rev_rc), win=60)
            if win:
                r_mfe, _ = run_rnafold(win, tmpdir)

        prefix = f"feat_pr_{locus}_"
        primer_feats.update({
            prefix+"f_mm_best": mm_f_best,
            prefix+"f_pos_best": pos_f_best,
            prefix+"f_mm_end": mm_f_end,
            prefix+"f_pos_end": pos_f_end,
            prefix+"f_is_end": int(ok_f_end),
            prefix+"f_mm_3p5_best": mm_f_3p5 if mm_f_3p5 != "" else "",
            prefix+"f_gc_end": f_gc,
            prefix+"f_mfe_endwin60": f_mfe,

            prefix+"r_mm_best": mm_r_best,
            prefix+"r_pos_best": pos_r_best,
            prefix+"r_mm_end": mm_r_end,
            prefix+"r_pos_end": pos_r_end,
            prefix+"r_is_end": int(ok_r_end),
            prefix+"r_mm_3p5_best": mm_r_3p5 if mm_r_3p5 != "" else "",
            prefix+"r_gc_end": r_gc,
            prefix+"r_mfe_endwin60": r_mfe,
        })

    core = {
        "Seq_ID": seq_id,
        "feat_len": n,

        "feat_pA": safe_div(A, n),
        "feat_pC": safe_div(C, n),
        "feat_pG": safe_div(G, n),
        "feat_pT": safe_div(T, n),
        "feat_pN": safe_div(Nn, n),
        "feat_hasN": int(Nn > 0),

        "feat_gc": gc,
        "feat_at": at,
        "feat_gc_skew": gc_skew,
        "feat_at_skew": at_skew,

        "feat_5p_base": s[0] if n else "",
        "feat_3p_base": s[-1] if n else "",
        "feat_gc_head30": gc_head,
        "feat_gc_tail30": gc_tail,

        "feat_CpG": safe_div(CpG, max(1, n-1)),
        "feat_UpA": safe_div(UpA, max(1, n-1)),
        "feat_or_CpG": or_cpg,

        "feat_entropy1": ent1,
        "feat_lz": lz,
        "feat_dust": dust,
        "feat_lingcomp_mean_k1_8": sum(ling.values())/8.0 if ling else 0.0,

        "feat_runA_max": runA,
        "feat_runC_max": runC,
        "feat_runG_max": runG,
        "feat_runT_max": runT,
        "feat_run_ge4": run_ge4,
        "feat_run_ge5": run_ge5,
        "feat_run_ge6": run_ge6,
        "feat_direp_maxreps": direp_reps,
        "feat_trirep_maxreps": trirep_reps,
        "feat_tandem_density": tr_density,

        "feat_pal_n6": pal_counts[6],
        "feat_pal_n8": pal_counts[8],
        "feat_pal_n10": pal_counts[10],
        "feat_pal_maxlen": pal_maxlen,
        "feat_end_complement": end_comp,

        "feat_g4_count": g4c,
        "feat_G_island_max": g_is_max,
        "feat_G_island_n3": g_is_n3,
        "feat_zdna_alt_max": zdna,

        "feat_mfe": mfe,
        "feat_pair_frac": pair_frac,
        "feat_stem_max": stem_max,
        "feat_loop_max": loop_max,
        "feat_hairpin_count": hp_count,

        "feat_tm": tm,
        "feat_tm_method": tm_method,
        "feat_tm_w20_max": tmw20,
        "feat_gc_w20_max": gcw20,
    }

    # add all 16 di frequencies
    core.update(di_freq)
    # add kmer summaries
    core.update(ksum)
    # add linguistic complexity per k (small, 8 cols)
    for k, v in ling.items():
        core[f"feat_lingcomp_k{k}"] = v
    # add primer features
    core.update(primer_feats)

    kmer_sparse = {
        "Seq_ID": seq_id,
        "kmer_all_json": json.dumps(kmer_sparse_all, separators=(",", ":")),
        "kmer_head_json": json.dumps(kmer_sparse_head, separators=(",", ":")),
        "kmer_tail_json": json.dumps(kmer_sparse_tail, separators=(",", ":")),
    }
    return core, kmer_sparse

# ------------------ IO ------------------

def write_tsv_gz(path: Path, rows: List[Dict[str,object]], field_order: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\t".join(field_order) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in field_order) + "\n")

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk_fasta", help="Input fasta chunk")
    ap.add_argument("chunk_name", help="Chunk name prefix (e.g., chunk_0001)")
    ap.add_argument("--project_dir", default=None, help="Project root (default: parent of scripts dir)")
    ap.add_argument("--out_dir", default=None, help="analysis_results/02_Features (default under project)")
    ap.add_argument("--force", action="store_true", help="Overwrite outputs")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = Path(args.project_dir).resolve() if args.project_dir else script_dir.parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (project_dir / "analysis_results" / "02_Features")

    chunk_fasta = Path(args.chunk_fasta).resolve()
    chunk_name = args.chunk_name

    core_out = out_dir / "core_chunks" / f"{chunk_name}.core.tsv.gz"
    kmer_out = out_dir / "kmer_sparse_chunks" / f"{chunk_name}.kmer.tsv.gz"

    if (core_out.exists() or kmer_out.exists()) and (not args.force):
        print(f"[SKIP] outputs exist for {chunk_name}. Use --force to overwrite.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "core_chunks").mkdir(parents=True, exist_ok=True)
    (out_dir / "kmer_sparse_chunks").mkdir(parents=True, exist_ok=True)

    # per-task temp dir, ensure cleanup
    tmp_root = out_dir / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix=f"{chunk_name}_", dir=str(tmp_root)))

    core_rows: List[Dict[str,object]] = []
    kmer_rows: List[Dict[str,object]] = []

    try:
        for sid, seq in fasta_iter(chunk_fasta):
            core, kmer = compute_features(sid, seq, tmpdir)
            core_rows.append(core)
            kmer_rows.append(kmer)

        # stable field order: from first row keys (sorted for reproducibility)
        # Ensure Seq_ID first.
        core_fields = ["Seq_ID"] + sorted([k for k in core_rows[0].keys() if k != "Seq_ID"])
        kmer_fields = ["Seq_ID", "kmer_all_json", "kmer_head_json", "kmer_tail_json"]

        write_tsv_gz(core_out, core_rows, core_fields)
        write_tsv_gz(kmer_out, kmer_rows, kmer_fields)

        print(f"[DONE] {chunk_name}: n={len(core_rows)} -> {core_out.name}, {kmer_out.name}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
