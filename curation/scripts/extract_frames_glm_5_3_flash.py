"""Extract a uniform frame pool per clip for GLM-5.3-flash annotation.

Reads the clip -> vision-media mapping from curation.sqlite (read-only) and
writes JPEG frames plus manifest.jsonl under <data-root>/media/frames96/.
Idempotent: clips whose latest manifest record shows a complete extraction
are skipped on rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

TIMEOUT = 60

# Populated in main() before the worker pool forks; workers read only these.
CURATION_DB = ""
MEDIA_ROOT = Path()
OUT_ROOT = Path()
MANIFEST = Path()
N_TARGET = 96
FRAME_WIDTH = 640
N_WORKERS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract uniform per-clip frame pools for GLM annotation"
    )
    parser.add_argument(
        "--curation-db",
        default=os.environ.get("MPRISK_CURATION_DB"),
        help="curation.sqlite path (env MPRISK_CURATION_DB)",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("MPRISK_CURATION_DATA"),
        help="curation data root; media root and output dir are derived "
        "as <data-root>/media and <data-root>/media/frames96 "
        "(env MPRISK_CURATION_DATA)",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=int(os.environ.get("GLM_TARGET_FRAMES", "96")),
        help="frames to extract per clip (env GLM_TARGET_FRAMES, default 96)",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=int(os.environ.get("GLM_FRAME_WIDTH", "640")),
        help="JPEG output width in pixels (env GLM_FRAME_WIDTH, default 640)",
    )
    parser.add_argument(
        "--conc",
        type=int,
        default=int(os.environ.get("GLM_CONC", "16")),
        help="worker processes (env GLM_CONC, default 16)",
    )
    args = parser.parse_args()
    if not args.curation_db:
        parser.error("--curation-db is required (or set MPRISK_CURATION_DB)")
    if not args.data_root:
        parser.error("--data-root is required (or set MPRISK_CURATION_DATA)")
    return args


def load_clips() -> list[tuple[str, str]]:
    conn = sqlite3.connect(f"file:{CURATION_DB}?mode=ro", uri=True)
    rows = conn.execute("SELECT payload_json FROM samples").fetchall()
    conn.close()
    clips: dict[str, str] = {}
    for (payload,) in rows:
        data = json.loads(payload)
        source_id = data.get("source_id")
        vision = (data.get("media_asset_ids") or {}).get("vision")
        if source_id and vision:
            clips[source_id] = vision
    return sorted(clips.items())


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,r_frame_rate,width,height",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    info = json.loads(out.stdout)
    stream = info["streams"][0]
    duration = float(info.get("format", {}).get("duration") or 0.0)
    num, den = stream.get("r_frame_rate", "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    n_total = int(stream.get("nb_read_frames") or 0)
    if n_total <= 0 and duration > 0 and fps > 0:
        n_total = int(round(duration * fps))
    return {
        "n_total_frames": n_total,
        "fps": round(fps, 3),
        "duration": round(duration, 3),
        "width": stream.get("width"),
        "height": stream.get("height"),
    }


def extract(path: Path, out_dir: Path, indices: list[int]) -> None:
    select = "+".join(f"eq(n\\,{i})" for i in indices)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select='{select}',scale={FRAME_WIDTH}:-2",
        "-vsync",
        "0",
        "-frames:v",
        str(len(indices)),
        "-q:v",
        "2",
        "-qmin",
        "2",
        "-qmax",
        "4",
        str(out_dir / "f%03d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, check=True)


def process(item: tuple[str, str]) -> dict:
    source_id, relpath = item
    rec: dict = {"source_id": source_id}
    try:
        path = MEDIA_ROOT / relpath
        info = probe(path)
        n_total = info["n_total_frames"]
        n = min(N_TARGET, n_total)
        if n <= 0:
            raise RuntimeError(
                f"no frames detected (duration={info['duration']} fps={info['fps']})"
            )
        indices = sorted({round(i * (n_total - 1) / (n - 1)) for i in range(n)})
        out_dir = OUT_ROOT / source_id
        out_dir.mkdir(parents=True, exist_ok=True)
        extract(path, out_dir, indices)
        rec.update(info)
        rec["n_extracted"] = len(list(out_dir.glob("f*.jpg")))
        rec["done"] = 1
    except Exception as exc:  # noqa: BLE001 - any failure is recorded in the manifest
        rec["done"] = 0
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def load_done_ids() -> set[str]:
    done_ids: set[str] = set()
    if not MANIFEST.exists():
        return done_ids
    with MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # keep the latest record per clip
            if row.get("done") and row.get("n_extracted") == min(
                N_TARGET, row.get("n_total_frames") or 0
            ):
                done_ids.add(row["source_id"])
            else:
                done_ids.discard(row["source_id"])
    return done_ids


def main() -> None:
    global CURATION_DB, MEDIA_ROOT, OUT_ROOT, MANIFEST, N_TARGET, FRAME_WIDTH, N_WORKERS
    args = parse_args()
    CURATION_DB = args.curation_db
    MEDIA_ROOT = Path(args.data_root) / "media"
    OUT_ROOT = MEDIA_ROOT / "frames96"
    MANIFEST = OUT_ROOT / "manifest.jsonl"
    N_TARGET = args.target_frames
    FRAME_WIDTH = args.frame_width
    N_WORKERS = args.conc

    if not Path(CURATION_DB).is_file():
        print(f"FATAL: curation db not found: {CURATION_DB}", file=sys.stderr)
        sys.exit(2)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids()
    clips = load_clips()
    todo = [item for item in clips if item[0] not in done_ids]
    print(f"clips={len(clips)} done={len(done_ids)} todo={len(todo)}", flush=True)

    t0 = time.time()
    n_ok = n_fail = 0
    with MANIFEST.open("a", encoding="utf-8") as handle, Pool(N_WORKERS) as pool:
        for i, rec in enumerate(pool.imap_unordered(process, todo, chunksize=4), 1):
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec.get("done"):
                n_ok += 1
            else:
                n_fail += 1
                print(f"FAIL {rec['source_id']}: {rec.get('error')}", flush=True)
            if i % 50 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {i}/{len(todo)} ok={n_ok} fail={n_fail} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    print(
        f"done: ok={n_ok} fail={n_fail} skipped={len(done_ids)} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
