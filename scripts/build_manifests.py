from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.data.delivery import build_derived_manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the frozen delivery and build deterministic manifests."
    )
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    result = build_derived_manifests(
        args.repo_root,
        check_media=True,
        verify_archive=True,
    )
    print(
        json.dumps(
            {
                "total_rows": result.validation.total_rows,
                "unique_media_paths": result.validation.unique_media_paths,
                "outputs": {key: str(path) for key, path in result.output_paths.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
