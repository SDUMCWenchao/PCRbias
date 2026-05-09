from __future__ import annotations

from pathlib import Path

from tools.run_synthetic_smoke import run_chapter4_synthetic_smoke


def test_chapter4_synthetic_smoke(tmp_path: Path) -> None:
    project = run_chapter4_synthetic_smoke(workdir=tmp_path / "project")

    assert (project / "analysis_results" / "00_manifest" / "manifest.tsv").exists()
    assert (project / "analysis_results" / "01_Sequences" / "samples_summary.tsv").exists()
