from __future__ import annotations

from pathlib import Path

from tools import check_repository_consistency


def test_repository_url_check_flags_stale_url(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "README.md").write_text(
        "https://github.com/SDUMCWenchao/PCR_bias\n", encoding="utf-8"
    )
    (tmp_path / "configs" / "config.example.yaml").write_text(
        "repository: https://github.com/SDUMCWenchao/PCRbias\n", encoding="utf-8"
    )
    (tmp_path / "CITATION.cff").write_text(
        "repository-code: https://github.com/SDUMCWenchao/PCRbias\n", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    problems = check_repository_consistency.check_repository_url(tmp_path)

    assert any("stale repository URL" in problem for problem in problems)


def test_run_order_checker_flags_missing_script(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    header = (
        "step_id\tstage\tscript\tstatus\trequired_inputs\tmain_outputs\t"
        "command_example\tresume_safe\n"
    )
    (docs / "RUN_ORDER.chapter2_3.tsv").write_text(
        header + "00\tstage\tmissing.py\tcanonical\tin\tout\tcmd\tno\n",
        encoding="utf-8",
    )
    (docs / "RUN_ORDER.chapter4.tsv").write_text(header, encoding="utf-8")

    problems = check_repository_consistency.check_run_order_files(tmp_path)

    assert problems == [
        "run-order script not found in docs/RUN_ORDER.chapter2_3.tsv: missing.py"
    ]


def test_chapter4_status_allows_existing_non_script_entries(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    chapter4 = tmp_path / "legacy" / "chapter4"
    docs.mkdir()
    chapter4.mkdir(parents=True)
    (chapter4 / "workflow.py").write_text("print('ok')\n", encoding="utf-8")
    (chapter4 / "samples_meta.tsv").write_text("file_id\nS1\n", encoding="utf-8")
    (docs / "CHAPTER4_SCRIPT_STATUS.tsv").write_text(
        "path\tstatus\tnotes\n"
        "legacy/chapter4/workflow.py\tcanonical_or_support\tcovered script\n"
        "legacy/chapter4/samples_meta.tsv\texample_input\tcovered input\n",
        encoding="utf-8",
    )

    problems = check_repository_consistency.check_chapter4_status_coverage(tmp_path)

    assert problems == []


def test_chapter4_status_flags_missing_documented_paths(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "CHAPTER4_SCRIPT_STATUS.tsv").write_text(
        "path\tstatus\tnotes\nlegacy/chapter4/missing.tsv\texample_input\tmissing input\n",
        encoding="utf-8",
    )

    problems = check_repository_consistency.check_chapter4_status_coverage(tmp_path)

    assert problems == ["stale Chapter 4 status entry: legacy/chapter4/missing.tsv"]
