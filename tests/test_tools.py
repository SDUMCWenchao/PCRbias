from __future__ import annotations

from pathlib import Path

from tools import audit_hardcoded_paths, make_inventory


def test_audit_scan_flags_patterns_and_respects_exclusions(tmp_path: Path) -> None:
    (tmp_path / "script.py").write_text("data = '/datapool/project'\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "HARDCODED_PATHS.tsv").write_text("/datapool/allowed\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("/home/ignored\n", encoding="utf-8")

    hits = audit_hardcoded_paths.scan(root=tmp_path)

    assert hits == [(Path("script.py"), 1, "data = '/datapool/project'")]


def test_audit_main_reports_without_failing_in_legacy_mode(tmp_path: Path, capsys) -> None:
    (tmp_path / "notes.txt").write_text("private=/srv/private\n", encoding="utf-8")

    exit_code = audit_hardcoded_paths.main(
        ["--root", str(tmp_path), "--pattern", "/srv/private", "--legacy-mode", "report"]
    )

    assert exit_code == 0
    assert "Reported 1 hits" in capsys.readouterr().out


def test_inventory_helpers_classify_and_skip_cache_dirs(tmp_path: Path) -> None:
    (tmp_path / "workflow.slurm").write_text("#!/bin/bash\n#SBATCH --time=1\n", encoding="utf-8")
    (tmp_path / "README").write_text("one line\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignored.py").write_text("print('ignored')\n", encoding="utf-8")

    rows = make_inventory.iter_inventory_rows(tmp_path)

    assert rows == [(Path("README"), "file", 1), (Path("workflow.slurm"), "slurm", 2)]


def test_write_inventory_uses_tsv_header(tmp_path: Path) -> None:
    output = tmp_path / "inventory.tsv"

    make_inventory.write_inventory([(Path("a.py"), "python", 3)], output)

    assert output.read_text(encoding="utf-8").splitlines() == ["path\ttype\tlines", "a.py\tpython\t3"]
