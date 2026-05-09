#!/usr/bin/env python3
from pathlib import Path
import csv, sys
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample_metadata_template.tsv")
OUT.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "sample_id","file_path","marker","group_name","sample_type","species_scope",
    "is_core_analysis","forward_primer_5to3","reverse_primer_5to3","min_len","max_len","notes",
]
rows = [
    ["MD10_12","/path/to/chapter2_3_analysis/01_raw_fastq/MD10_12.fq.gz","12S","MD10","inter_mix","inter","yes","REPLACE_12S_FWD","REPLACE_12S_REV","80","300","ten-species equal-DNA mixture"],
    ["MP10_12","/path/to/chapter2_3_analysis/01_raw_fastq/MP10_12.fq.gz","12S","MP10","inter_mix","inter","yes","REPLACE_12S_FWD","REPLACE_12S_REV","80","300","ten-species equal-PCR-product mixture"],
    ["EaD10_16","/path/to/chapter2_3_analysis/01_raw_fastq/EaD10_16.fq.gz","16S","EaD10","intra_mix","intra","yes","REPLACE_16S_FWD","REPLACE_16S_REV","80","300","Equus asinus 10-individual equal-DNA mixture"],
    ["Ea1_12","/path/to/chapter2_3_analysis/01_raw_fastq/Ea1_12.fq.gz","12S","Ea1","single_ref","single","no","REPLACE_12S_FWD","REPLACE_12S_REV","80","300","single-individual reference only"],
]
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(fields)
    w.writerows(rows)
print(f"Wrote template: {OUT}")
