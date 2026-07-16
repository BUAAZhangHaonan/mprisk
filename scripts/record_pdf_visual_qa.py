from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from mprisk.viz.bundle_figures import FORBIDDEN_PDF_TEXT
from mprisk.viz.runtime_records import (
    append_command_record,
    record_visual_qa,
    utc_now,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record actual vector-PDF visual QA evidence.")
    parser.add_argument("--config", type=Path, default=Path("configs/paper/figure_map.yaml"))
    parser.add_argument("--png-dir", type=Path, required=True)
    parser.add_argument("--run-records", type=Path, required=True)
    parser.add_argument("--qa-key", default="pending_bundle_v1")
    parser.add_argument("--visual-status", choices=("pass", "failure"), required=True)
    parser.add_argument("--notes", required=True)
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    started_at = utc_now()
    try:
        pdfs = _pdf_paths(args.config)
        pngs = [args.png_dir / f"{pdf.stem}-1.png" for pdf in pdfs]
        missing_pngs = [path for path in pngs if not path.is_file()]
        if missing_pngs:
            raise ValueError(f"missing rendered PNGs: {', '.join(map(str, missing_pngs))}")
        embedded_count = sum(_has_embedded_vector_font(pdf) for pdf in pdfs)
        forbidden_matches = _forbidden_matches(pdfs)
        if embedded_count != len(pdfs):
            raise ValueError("one or more PDFs have no embedded vector font")
        if forbidden_matches:
            raise ValueError(f"forbidden PDF text: {', '.join(sorted(forbidden_matches))}")
        record_visual_qa(
            args.run_records,
            qa_key=args.qa_key,
            status=args.visual_status,
            pdf_count=len(pdfs),
            rendered_png_count=len(pngs),
            embedded_font_pdf_count=embedded_count,
            forbidden_match_count=0,
            notes=args.notes,
        )
    except Exception as exc:
        append_command_record(
            args.run_records,
            command_id="record_pdf_visual_qa",
            argv=command,
            status="failure",
            pid=os.getpid(),
            started_at=started_at,
            reason=str(exc),
        )
        raise
    append_command_record(
        args.run_records,
        command_id="record_pdf_visual_qa",
        argv=command,
        status="success",
        pid=os.getpid(),
        started_at=started_at,
    )
    print(args.run_records)
    return 0


def _pdf_paths(config_path: Path) -> list[Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths = [
        Path(str(spec["output"]))
        for group in ("figures", "appendix")
        for spec in (config.get(group) or {}).values()
    ]
    if any(not path.is_file() for path in paths):
        raise ValueError("configured PDF output is missing")
    return paths


def _has_embedded_vector_font(path: Path) -> bool:
    completed = subprocess.run(
        ["pdffonts", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = completed.stdout.splitlines()[2:]
    return bool(rows) and all(" yes " in f" {row} " for row in rows if row.strip())


def _forbidden_matches(paths: list[Path]) -> set[str]:
    matches: set[str] = set()
    for path in paths:
        completed = subprocess.run(
            ["pdftotext", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = completed.stdout.casefold()
        matches.update(term for term in FORBIDDEN_PDF_TEXT if term in text)
    return matches


if __name__ == "__main__":
    raise SystemExit(main())
