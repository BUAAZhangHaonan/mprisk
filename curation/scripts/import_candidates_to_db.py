from __future__ import annotations

import argparse

from curation.backend.db import connect, upsert_sample
from curation.scripts.common import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--db", default="curation/outputs/curation.sqlite")
    args = parser.parse_args()
    conn = connect(args.db)
    try:
        for row in read_jsonl(args.input):
            upsert_sample(conn, row)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
