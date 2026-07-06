from __future__ import annotations

import argparse

from curation.backend.db import connect, list_annotations
from curation.scripts.common import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="curation/outputs/curation.sqlite")
    parser.add_argument("--output", default="curation/outputs/human/human_annotations.jsonl")
    args = parser.parse_args()
    conn = connect(args.db)
    try:
        write_jsonl(args.output, list_annotations(conn))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
