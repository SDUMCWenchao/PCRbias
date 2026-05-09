#!/usr/bin/env python3
"""Run a tiny synthetic Chapter 4 smoke test without scientific interpretation.

The test creates a temporary project with two artificial FASTQ samples, builds a
manifest, runs single-end preprocessing, and verifies essential outputs.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]


def write_fastq(path: Path, sequences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i, seq in enumerate(sequences, start=1):
            handle.write(f"@{path.stem}_{i}\n{seq}\n+\n{'I' * len(seq)}\n")


def make_project(project_dir: Path) -> None:
    raw_dir = project_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "raw_data_trimmed").mkdir(parents=True, exist_ok=True)

    metadata = project_dir / "samples_meta.tsv"
    metadata.write_text(
        "file_id\tsample_name\tspecies\tn_individuals\tlocus\tpcr\n"
        "H12\tMP10_12\t10_species_mix\t10\t12S\tyes\n"
        "H16\tMP10_16\t10_species_mix\t10\t16S\tno\n",
        encoding="utf-8",
    )
    write_fastq(raw_dir / "H12.fastq", ["ACGTACGTACGT", "ACGTACGTACGT", "ACGTACGTTCGT"])
    write_fastq(raw_dir / "H16.fastq", ["TTTTCCCCAAAA", "TTTTCCCCAAAA", "GGGGAAAACCCC"])


def run_command(command: list[str], root: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(command)
            + f"\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run_chapter4_synthetic_smoke(root: Path = ROOT, workdir: Path | None = None) -> Path:
    """Run the smoke test and return the temporary project directory."""
    if workdir is None:
        temp = tempfile.TemporaryDirectory(prefix="pcrbias_ch4_smoke_")
        project_dir = Path(temp.name)
        cleanup = temp.cleanup
    else:
        project_dir = workdir.resolve()
        if project_dir.exists():
            shutil.rmtree(project_dir)
        cleanup = None

    try:
        make_project(project_dir)
        manifest_dir = project_dir / "analysis_results" / "00_manifest"
        seq_dir = project_dir / "analysis_results" / "01_Sequences"

        run_command(
            [
                sys.executable,
                "legacy/chapter4/00_manifest_builder.py",
                "--project_dir",
                str(project_dir),
                "--meta",
                str(project_dir / "samples_meta.tsv"),
                "--raw_dir",
                str(project_dir / "raw_data"),
                "--trimmed_dir",
                str(project_dir / "raw_data_trimmed"),
                "--out_dir",
                str(manifest_dir),
                "--prefer_raw",
            ],
            root,
        )
        manifest = manifest_dir / "manifest.tsv"
        if not manifest.exists():
            raise AssertionError(f"manifest not created: {manifest}")
        rows = read_tsv_rows(manifest)
        statuses = {row["file_id"]: row["status"] for row in rows}
        if statuses != {"H12": "OK", "H16": "OK"}:
            raise AssertionError(f"unexpected manifest statuses: {statuses}")

        run_command(
            [
                sys.executable,
                "legacy/chapter4/01_preprocess_sequences_se.py",
                "--project_dir",
                str(project_dir),
                "--manifest",
                str(manifest),
                "--out_dir",
                str(seq_dir),
                "--max_reads",
                "100",
                "--validate_fastq",
            ],
            root,
        )

        expected_outputs = [
            seq_dir / "samples_summary.tsv",
            seq_dir / "H12_stats.tsv",
            seq_dir / "H16_stats.tsv",
            seq_dir / "ALL_UNIQUE_SEQUENCES.fasta",
            seq_dir / "ALL_UNIQUE_SEQUENCES.tsv.gz",
            seq_dir / "global_sequences.sqlite",
        ]
        missing = [str(path) for path in expected_outputs if not path.exists()]
        if missing:
            raise AssertionError("missing synthetic smoke outputs: " + ", ".join(missing))

        summary_rows = read_tsv_rows(seq_dir / "samples_summary.tsv")
        total_reads = {row["file_id"]: int(row["total_reads"]) for row in summary_rows}
        if total_reads != {"H12": 3, "H16": 3}:
            raise AssertionError(f"unexpected total_reads: {total_reads}")

        print(f"Synthetic Chapter 4 smoke test passed: {project_dir}")
        return project_dir
    finally:
        if cleanup is not None:
            cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Optional persistent working directory. Existing contents are replaced.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_chapter4_synthetic_smoke(root=args.root.resolve(), workdir=args.workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
