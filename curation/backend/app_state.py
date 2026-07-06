from __future__ import annotations

import os

from curation.backend.db import connect


def get_conn():
    path = os.environ.get("MPRISK_CURATION_DB", "curation/outputs/curation.sqlite")
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()
