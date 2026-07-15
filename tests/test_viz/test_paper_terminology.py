from __future__ import annotations

from pathlib import Path


def test_current_paper_and_pipeline_use_locked_terminology() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = [
        *sorted((root / "paper").glob("**/*.tex")),
        root / "docs/PIPELINE.md",
    ]
    forbidden = (
        "state consistency",
        "divergence",
        "arbitration",
        "wrong-answer",
        "wrong answer",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8").casefold()
        for term in forbidden:
            assert term not in text, f"{path} contains forbidden paper term {term}"


def test_paper_sources_reference_final_figure_and_table_exports() -> None:
    root = Path(__file__).resolve().parents[2]
    main_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "paper/sections").glob("*.tex"))
    )
    appendix_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "paper/appendix_sections").glob("*.tex"))
    )
    assert main_text.count(r"\includegraphics") == 10
    assert appendix_text.count(r"\includegraphics") == 14
    assert main_text.count(r"\input{tables/generated/") == 3
    assert "fig10_aligned_case" not in main_text
    assert "tab01_main_results" not in main_text
