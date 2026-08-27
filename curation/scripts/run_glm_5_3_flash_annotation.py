"""GLM-5.3-flash tri-modal (V/T) annotation over frames96 clip pools.

Model lineage: the pipeline was originally run under the free OpenRouter
alias "stealth/ox-alpha"; when its free period ended the alias was
delisted. The model behind it is z-ai/glm-5.3-flash, which is now the
default (GLM_MODEL).

Writes results to a standalone DB (<data-root>/glm_5_3_flash_annotation.sqlite);
never touches curation.sqlite (opened read-only).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image

# ---------------------------------------------------------------- config knobs (env)
MODEL = os.environ.get("GLM_MODEL", "z-ai/glm-5.3-flash")
ENDPOINTS_URL = f"https://openrouter.ai/api/v1/models/{MODEL}/endpoints"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
PROXY = os.environ.get("GLM_PROXY", "").strip()
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None
MAX_FRAMES = int(os.environ.get("GLM_MAX_FRAMES", "48"))
FRAME_WIDTH = int(os.environ.get("GLM_FRAME_WIDTH", "480"))
WORKERS = int(os.environ.get("GLM_CONC", "4"))

MAX_TOKENS = 4000
TIMEOUT_S = 240
MAX_ATTEMPTS = 5
BACKOFF_S = [30, 60, 120, 240]
JITTER_S = 10
ABORT_AFTER_CONSEC_FAILS = 60
ROUNDS = (1, 2, 3)
PROGRESS_EVERY = 20

PROMPT_V = (
    "你是情感标注员。下面按时间顺序给出同一个短视频均匀抽取的 {n} 帧画面。"
    "请只根据画面中人物的面部表情、眼神、头部姿态和身体动作判断该片段的整体情绪倾向。"
    "严格忽略画面中出现的任何字幕、水印、文字，它们不是判断依据。"
    "情绪三分类：1=积极，0=中性，-1=消极。"
    '输出 JSON：{"label": -1|0|1, "confidence": 0到1, '
    '"evidence": "一句话画面依据"}。不要输出其他内容。'
)
PROMPT_T = (
    "你是情感标注员。下面是一段中文视频里人物所说的台词文本。"
    "请只根据文本语义判断情绪倾向，视频画面与语音一概不考虑。"
    "情绪三分类：1=积极，0=中性，-1=消极。"
    '输出 JSON：{"label": -1|0|1, "confidence": 0到1, '
    '"evidence": "一句话文本依据"}。不要输出其他内容。'
)


# ---------------------------------------------------------------- paths
@dataclass(frozen=True)
class Paths:
    curation_db: str
    data_root: Path
    out_db: str
    frames_root: Path
    manifest: Path
    meta_csv: str | None
    to_adjudicate: Path

    @classmethod
    def resolve(cls, curation_db: str, data_root: str, meta_csv: str | None) -> Paths:
        root = Path(data_root)
        frames_root = root / "media" / "frames96"
        return cls(
            curation_db=curation_db,
            data_root=root,
            out_db=str(root / "glm_5_3_flash_annotation.sqlite"),
            frames_root=frames_root,
            manifest=frames_root / "manifest.jsonl",
            meta_csv=meta_csv,
            to_adjudicate=root / "to_adjudicate.jsonl",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLM-5.3-flash V/T annotation (annotate/aggregate/adjudicate/report)"
    )
    parser.add_argument(
        "--phase", choices=["annotate", "aggregate", "adjudicate", "report"], default="annotate"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--modalities", default="V,T")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--curation-db",
        default=os.environ.get("MPRISK_CURATION_DB"),
        help="curation.sqlite path, opened read-only (env MPRISK_CURATION_DB)",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("MPRISK_CURATION_DATA"),
        help="curation data root; the annotation DB and frames dir are derived "
        "as <data-root>/glm_5_3_flash_annotation.sqlite and "
        "<data-root>/media/frames96 (env MPRISK_CURATION_DATA)",
    )
    parser.add_argument(
        "--meta-csv",
        default=os.environ.get("MPRISK_META_CSV"),
        help="ground-truth meta.csv for the report phase; if unset the truth "
        "comparison is skipped (env MPRISK_META_CSV)",
    )
    parser.add_argument(
        "--api-key-file",
        default=os.environ.get("OPENROUTER_API_KEY_FILE", str(Path.home() / ".openrouter_api_key")),
        help="OpenRouter API key file (env OPENROUTER_API_KEY_FILE)",
    )
    args = parser.parse_args()
    if args.self_test:
        return args
    if not args.curation_db:
        parser.error("--curation-db is required (or set MPRISK_CURATION_DB)")
    if not args.data_root:
        parser.error("--data-root is required (or set MPRISK_CURATION_DATA)")
    return args


# ---------------------------------------------------------------- db
def init_db(out_db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(out_db, timeout=10)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(
        """
CREATE TABLE IF NOT EXISTS glm_runs(
  source_id TEXT NOT NULL, modality TEXT NOT NULL CHECK(modality IN ('V','T')),
  round INTEGER NOT NULL, label INTEGER, confidence REAL, evidence TEXT,
  status TEXT NOT NULL, http_status INTEGER, latency_s REAL,
  prompt_tokens INTEGER, completion_tokens INTEGER, raw_response TEXT,
  model TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(source_id, modality, round));
CREATE TABLE IF NOT EXISTS glm_adjudications(
  id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL, modality TEXT NOT NULL,
  round_labels_json TEXT NOT NULL, final_label INTEGER, confidence REAL,
  rationale TEXT, status TEXT NOT NULL, raw_response TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS glm_final(
  source_id TEXT NOT NULL, modality TEXT NOT NULL, final_label INTEGER,
  method TEXT NOT NULL, agreement REAL, mean_confidence REAL,
  adjudication_id INTEGER, created_at TEXT NOT NULL,
  PRIMARY KEY(source_id, modality));
"""
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------- data loading
def load_clips(curation_db: str) -> dict[str, str]:
    """source_id -> deduped text_content (read-only connection)."""
    conn = sqlite3.connect(f"file:{curation_db}?mode=ro", uri=True)
    rows = conn.execute("SELECT payload_json FROM samples").fetchall()
    conn.close()
    clips: dict[str, str] = {}
    for (payload,) in rows:
        data = json.loads(payload)
        sid = data.get("source_id")
        if sid and sid not in clips:
            clips[sid] = data.get("text_content") or ""
    return clips


def load_manifest(manifest: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["source_id"]] = rec
    return records


def sign_label(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def load_truth(meta_csv: str | None) -> dict[tuple[str, str], dict[str, int]]:
    truth: dict[tuple[str, str], dict[str, int]] = {}
    if not meta_csv:
        return truth
    with open(meta_csv, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        for line in fh:
            parts = line.rstrip("\n").split(",")
            truth[(parts[idx["video_id"]], parts[idx["clip_id"]])] = {
                "label_V": sign_label(float(parts[idx["label_V"]])),
                "label_T": sign_label(float(parts[idx["label_T"]])),
            }
    return truth


# ---------------------------------------------------------------- frames
def pick_frame_indices(n_extracted: int, cap: int, round_no: int) -> list[int]:
    """Indices into the sorted frame list; round1=first N, round2=last N, round3=even stride."""
    n = min(cap, n_extracted)
    if n_extracted <= n:
        return list(range(n_extracted))
    if round_no == 1:
        return list(range(n))
    if round_no == 2:
        return list(range(n_extracted - n, n_extracted))
    step = n_extracted / n
    return sorted({min(int(i * step), n_extracted - 1) for i in range(n)})


def frame_files(frames_root: Path, source_id: str) -> list[Path]:
    d = frames_root / source_id
    if not d.is_dir():
        return []
    return sorted(d.glob("f*.jpg"))


def image_data_url(path: Path) -> str:
    img = Image.open(io.BytesIO(path.read_bytes())).convert("RGB")
    if img.width > FRAME_WIDTH:
        h = max(2, round(img.height * FRAME_WIDTH / img.width))
        h -= h % 2
        img = img.resize((FRAME_WIDTH, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------- parsing
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_json(text: str, evidence_key: str = "evidence") -> tuple[int, float, str]:
    """Return (label, confidence, evidence); raise ValueError on any malformation."""
    if not text:
        raise ValueError("empty response")
    candidate = text.strip()
    m = FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        m = BRACE_RE.search(candidate)
        if not m:
            raise ValueError(f"no JSON object in: {text[:120]!r}") from None
        obj = json.loads(m.group(0))
    label = obj.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or label not in (-1, 0, 1):
        raise ValueError(f"label out of range: {label!r}")
    conf = obj.get("confidence")
    if not isinstance(conf, int | float) or isinstance(conf, bool):
        raise ValueError(f"bad confidence: {conf!r}")
    conf = max(0.0, min(1.0, float(conf)))
    evidence = str(obj.get(evidence_key) or "")
    return label, conf, evidence


# ---------------------------------------------------------------- api
def read_api_key(api_key_file: str) -> str:
    key = Path(api_key_file).read_text().strip()
    if not key:
        print(f"FATAL: empty API key file {api_key_file}", file=sys.stderr)
        sys.exit(2)
    return key


def assert_endpoint_free(api_key: str) -> None:
    last_err = "no endpoints returned"
    for attempt in range(3):
        try:
            resp = requests.get(
                ENDPOINTS_URL,
                proxies=PROXIES,
                timeout=30,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            endpoints = (resp.json().get("data") or {}).get("endpoints") or []
            if not endpoints:
                last_err = "no endpoints returned"
            else:
                pricing = endpoints[0].get("pricing") or {}
                p_prompt = float(pricing.get("prompt") or 1)
                p_completion = float(pricing.get("completion") or 1)
                if p_prompt > 1e-7 or p_completion > 3e-7:
                    print(
                        f"FATAL: {MODEL} too expensive (prompt={p_prompt}, "
                        f"completion={p_completion})",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                print(f"[assert] {MODEL} price ok (prompt={p_prompt} completion={p_completion})")
                return
        except Exception as e:  # noqa: BLE001 - retried below, fatal after 3 attempts
            last_err = str(e)
        if attempt < 2:
            time.sleep(30)
    print(f"FATAL: {last_err} for {MODEL} after 3 attempts", file=sys.stderr)
    sys.exit(2)


def chat(messages: list[dict], *, temperature: float, reasoning_effort: str, api_key: str):
    """POST one chat completion; return (http_status, parsed_body, latency)."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": reasoning_effort},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t0 = time.time()
    resp = requests.post(
        CHAT_URL, json=payload, headers=headers, proxies=PROXIES, timeout=TIMEOUT_S
    )
    latency = time.time() - t0
    body = resp.json() if resp.content else {}
    return resp.status_code, body, latency


# ---------------------------------------------------------------- annotate
def prompt_v_text(n: int) -> str:
    return PROMPT_V.replace("{n}", str(n))


def build_messages(
    paths: Paths, modality: str, source_id: str, text: str, frame_idx: list[int]
) -> list[dict]:
    if modality == "V":
        files = frame_files(paths.frames_root, source_id)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_v_text(len(frame_idx))}]
        for i in frame_idx:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(files[i])}})
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": PROMPT_T + "\n\n台词文本：" + text}]


def is_retryable(status: int | None) -> bool:
    return status is None or status == 429 or status >= 500


def annotate_one(paths: Paths, task: dict, api_key: str) -> dict:
    """Run one (source_id, modality, round) task with retries. Returns DB row dict."""
    sid, modality, rnd = task["source_id"], task["modality"], task["round"]
    row = {
        "source_id": sid,
        "modality": modality,
        "round": rnd,
        "label": None,
        "confidence": None,
        "evidence": None,
        "status": "failed",
        "http_status": None,
        "latency_s": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "raw_response": None,
        "model": MODEL,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    messages = build_messages(paths, modality, sid, task["text"], task["frame_idx"])
    for attempt in range(MAX_ATTEMPTS):
        try:
            status, body, latency = chat(
                messages, temperature=0.7, reasoning_effort="low", api_key=api_key
            )
            row["http_status"] = status
            row["latency_s"] = round(latency, 3)
            if status != 200:
                raise RuntimeError(f"http {status}: {str(body)[:200]}")
            choice = (body.get("choices") or [{}])[0]
            raw = (choice.get("message") or {}).get("content") or ""
            row["raw_response"] = raw
            usage = body.get("usage") or {}
            row["prompt_tokens"] = usage.get("prompt_tokens")
            row["completion_tokens"] = usage.get("completion_tokens")
            label, conf, evidence = parse_model_json(raw)
            row.update(label=label, confidence=conf, evidence=evidence, status="ok")
            return row
        except Exception as exc:  # noqa: BLE001 - any failure consumes one attempt
            row["raw_response"] = row["raw_response"] or f"attempt{attempt + 1}: {exc}"
            http = row["http_status"] if isinstance(exc, RuntimeError) else None
            if not is_retryable(http):
                break
            if attempt < MAX_ATTEMPTS - 1:
                sleep_s = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)] + random.uniform(0, JITTER_S)
                time.sleep(sleep_s)
    return row


def save_run(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO glm_runs(source_id,modality,round,label,confidence,evidence,"
        "status,http_status,latency_s,prompt_tokens,completion_tokens,raw_response,model,created_at)"
        " VALUES(:source_id,:modality,:round,:label,:confidence,:evidence,:status,:http_status,"
        ":latency_s,:prompt_tokens,:completion_tokens,:raw_response,:model,:created_at)",
        row,
    )
    conn.commit()


def build_tasks(
    paths: Paths,
    clips: dict[str, str],
    manifest: dict[str, dict],
    conn: sqlite3.Connection,
    modalities: list[str],
    limit: int | None,
) -> tuple[list[dict], list[str]]:
    done = {
        (sid, m, r)
        for sid, m, r, st in conn.execute(
            "SELECT source_id,modality,round,status FROM glm_runs WHERE status='ok'"
        )
    }
    sids = sorted(clips)
    if limit is not None:
        sids = sids[:limit]
    tasks: list[dict] = []
    skipped: list[str] = []
    for sid in sids:
        files = frame_files(paths.frames_root, sid)
        n_ext = (manifest.get(sid) or {}).get("n_extracted", len(files))
        for modality in modalities:
            for rnd in ROUNDS:
                if (sid, modality, rnd) in done:
                    skipped.append(f"{sid}/{modality}/{rnd}")
                    continue
                if modality == "V" and not files:
                    continue
                frame_idx = pick_frame_indices(n_ext, MAX_FRAMES, rnd) if modality == "V" else []
                tasks.append(
                    {
                        "source_id": sid,
                        "modality": modality,
                        "round": rnd,
                        "text": clips[sid],
                        "frame_idx": frame_idx,
                    }
                )
    return tasks, skipped


def run_annotate(paths: Paths, clips, manifest, conn, modalities, limit, api_key_file: str) -> None:
    tasks, skipped = build_tasks(paths, clips, manifest, conn, modalities, limit)
    total = len(tasks)
    print(
        f"[annotate] tasks={total} skipped(done)={len(skipped)} "
        f"clips={len(clips) if limit is None else min(limit, len(clips))}"
    )
    if not tasks:
        return
    api_key = read_api_key(api_key_file)
    assert_endpoint_free(api_key)
    done_count = fail_count = consec_fail = 0
    t0 = time.time()
    aborted = False
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(annotate_one, paths, t, api_key): t for t in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            save_run(conn, row)
            done_count += 1
            if row["status"] == "ok":
                consec_fail = 0
            else:
                fail_count += 1
                consec_fail += 1
            if done_count % PROGRESS_EVERY == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] done={done_count}/{total} "
                    f"failed={fail_count} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            if consec_fail >= ABORT_AFTER_CONSEC_FAILS:
                print(
                    f"[abort] {ABORT_AFTER_CONSEC_FAILS} consecutive failures, stopping.",
                    file=sys.stderr,
                )
                aborted = True
                pool.shutdown(wait=False, cancel_futures=True)
                break
    print(
        f"[annotate] finished done={done_count}/{total} failed={fail_count} "
        f"aborted={aborted} elapsed={time.time() - t0:.0f}s"
    )


# ---------------------------------------------------------------- aggregate
def aggregate_round(labels: list[tuple[int, float]]) -> dict:
    """labels: [(label, confidence)] for ok rounds only."""
    if len(labels) < 2:
        return {"decision": "needs_more_rounds"}
    counts: dict[int, int] = {}
    for lab, _ in labels:
        counts[lab] = counts.get(lab, 0) + 1
    best_lab, best_n = max(counts.items(), key=lambda kv: kv[1])
    if best_n >= 2:
        mean_conf = sum(c for lab, c in labels if lab == best_lab) / best_n
        if mean_conf >= 0.6:
            return {
                "decision": "majority",
                "label": best_lab,
                "agreement": best_n / len(labels),
                "mean_confidence": mean_conf,
            }
    return {"decision": "adjudicate"}


def run_aggregate(paths: Paths, conn, clips, modalities, limit) -> None:
    adjudicate_list: list[tuple[str, str]] = []
    n_majority = n_needs = 0
    sids = sorted(clips)
    if limit is not None:
        sids = sids[:limit]
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for sid in sids:
        for modality in modalities:
            rows = conn.execute(
                "SELECT round,label,confidence,status FROM glm_runs "
                "WHERE source_id=? AND modality=? ORDER BY round",
                (sid, modality),
            ).fetchall()
            ok = [(lab, conf) for _, lab, conf, st in rows if st == "ok" and lab is not None]
            res = aggregate_round(ok)
            if res["decision"] == "majority":
                conn.execute(
                    "INSERT OR REPLACE INTO glm_final(source_id,modality,final_label,method,"
                    "agreement,mean_confidence,adjudication_id,created_at) "
                    "VALUES(?,?,?,?,?,?,NULL,?)",
                    (
                        sid,
                        modality,
                        res["label"],
                        "majority",
                        res["agreement"],
                        res["mean_confidence"],
                        now,
                    ),
                )
                n_majority += 1
            elif res["decision"] == "adjudicate":
                adjudicate_list.append((sid, modality))
            else:
                n_needs += 1
    conn.commit()
    print(
        f"[aggregate] majority={n_majority} to_adjudicate={len(adjudicate_list)} "
        f"needs_more_rounds={n_needs}"
    )
    with open(paths.to_adjudicate, "w", encoding="utf-8") as fh:
        for sid, m in adjudicate_list:
            fh.write(json.dumps({"source_id": sid, "modality": m}) + "\n")


# ---------------------------------------------------------------- adjudicate
def adjudication_messages(
    paths: Paths, sid: str, modality: str, clips: dict[str, str], rounds: list[dict]
) -> list[dict]:
    """Self-contained arbitration prompt: states the media type, shows the three round
    conclusions, and asks for exactly one output schema ({"label","confidence","rationale"})."""
    rounds_desc = "\n".join(
        f"第{r['round']}轮: label={r['label']}, confidence={r['confidence']:.2f}, "
        f"依据: {r['evidence']}"
        for r in rounds
    )
    if modality == "V":
        files = frame_files(paths.frames_root, sid)
        idx = pick_frame_indices(len(files), MAX_FRAMES, 3)
        text = (
            "你是情感标注仲裁员。同一段短视频的画面情绪标注进行了三轮独立评审，三轮结论存在分歧。"
            "下面先给出三轮结论，再按时间顺序给出该片段均匀抽取的画面帧。"
            "请以画面为准（只看人物的面部表情、眼神、头部姿态和身体动作，"
            "严格忽略字幕、水印、文字），综合三轮意见做出最终裁决。"
            "情绪三分类：1=积极，0=中性，-1=消极。"
            '输出 JSON：{"label": -1|0|1, "confidence": 0到1, '
            '"rationale": "一句话裁决理由"}。不要输出其他内容。'
            f"\n\n三轮结论：\n{rounds_desc}"
            f"\n\n画面帧如下（共 {len(idx)} 帧，按时间顺序）："
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for i in idx:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(files[i])}})
        return [{"role": "user", "content": content}]
    text = (
        "你是情感标注仲裁员。同一段中文视频的台词文本情绪标注进行了三轮独立评审，"
        "三轮结论存在分歧。下面先给出三轮结论，再给出完整台词文本。"
        "请以文本语义为准（不考虑画面与语音），综合三轮意见做出最终裁决。"
        "情绪三分类：1=积极，0=中性，-1=消极。"
        '输出 JSON：{"label": -1|0|1, "confidence": 0到1, '
        '"rationale": "一句话裁决理由"}。不要输出其他内容。'
        f"\n\n三轮结论：\n{rounds_desc}"
        f"\n\n台词文本：{clips[sid]}"
    )
    return [{"role": "user", "content": text}]


def adjudicate_one(
    paths: Paths, sid: str, modality: str, clips: dict[str, str], rounds: list[dict], api_key: str
) -> dict:
    messages = adjudication_messages(paths, sid, modality, clips, rounds)
    rec = {
        "source_id": sid,
        "modality": modality,
        "round_labels_json": json.dumps(
            [
                {"round": r["round"], "label": r["label"], "confidence": r["confidence"]}
                for r in rounds
            ],
            ensure_ascii=False,
        ),
        "final_label": None,
        "confidence": None,
        "rationale": None,
        "status": "failed",
        "raw_response": None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for attempt in range(MAX_ATTEMPTS):
        try:
            status, body, _latency = chat(
                messages, temperature=0.3, reasoning_effort="high", api_key=api_key
            )
            if status != 200:
                raise RuntimeError(f"http {status}")
            raw = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            rec["raw_response"] = raw
            label, conf, rationale = parse_model_json(raw, evidence_key="rationale")
            rec.update(final_label=label, confidence=conf, rationale=rationale, status="ok")
            break
        except Exception as exc:  # noqa: BLE001 - any failure consumes one attempt
            rec["raw_response"] = rec["raw_response"] or str(exc)
            if attempt < MAX_ATTEMPTS - 1:
                sleep_s = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)] + random.uniform(0, JITTER_S)
                time.sleep(sleep_s)
    return rec


def run_adjudicate(paths: Paths, conn, clips, modalities, api_key_file: str) -> None:
    pending: list[tuple[str, str]] = []
    for sid in sorted(clips):
        for modality in modalities:
            if conn.execute(
                "SELECT 1 FROM glm_final WHERE source_id=? AND modality=?", (sid, modality)
            ).fetchone():
                continue
            rows = conn.execute(
                "SELECT round,label,confidence FROM glm_runs "
                "WHERE source_id=? AND modality=? AND status='ok' AND label IS NOT NULL "
                "ORDER BY round",
                (sid, modality),
            ).fetchall()
            if len(rows) < 2:
                continue
            res = aggregate_round([(lab, conf) for _, lab, conf in rows])
            if res["decision"] == "adjudicate":
                pending.append((sid, modality))
    print(f"[adjudicate] pending={len(pending)}")
    if not pending:
        return
    api_key = read_api_key(api_key_file)
    assert_endpoint_free(api_key)
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for sid, modality in pending:
            rows = conn.execute(
                "SELECT round,label,confidence,evidence FROM glm_runs "
                "WHERE source_id=? AND modality=? AND status='ok' AND label IS NOT NULL "
                "ORDER BY round",
                (sid, modality),
            ).fetchall()
            rounds = [
                {"round": r, "label": lab, "confidence": c, "evidence": e or ""}
                for r, lab, c, e in rows
            ]
            futures[pool.submit(adjudicate_one, paths, sid, modality, clips, rounds, api_key)] = (
                sid,
                modality,
            )
        for fut in as_completed(futures):
            sid, modality = futures[fut]
            rec = fut.result()
            cur = conn.execute(
                "INSERT INTO glm_adjudications(source_id,modality,round_labels_json,final_label,"
                "confidence,rationale,status,raw_response,created_at)"
                " VALUES(:source_id,:modality,:round_labels_json,:final_label,:confidence,"
                ":rationale,:status,:raw_response,:created_at)",
                rec,
            )
            adj_id = cur.lastrowid
            if rec["status"] == "ok":
                conn.execute(
                    "INSERT OR REPLACE INTO glm_final(source_id,modality,final_label,method,"
                    "agreement,mean_confidence,adjudication_id,created_at)"
                    " VALUES(?,?,?,?,NULL,NULL,?,?)",
                    (sid, modality, rec["final_label"], "adjudicated", adj_id, rec["created_at"]),
                )
                n_ok += 1
            else:
                n_fail += 1
            conn.commit()
    print(f"[adjudicate] ok={n_ok} failed={n_fail}")


# ---------------------------------------------------------------- report
def run_report(paths: Paths, conn, clips, modalities) -> None:
    truth = load_truth(paths.meta_csv)
    if not paths.meta_csv:
        print("(no --meta-csv / MPRISK_META_CSV configured; skipping truth comparison)")
    for modality in modalities:
        rows = conn.execute(
            "SELECT source_id,status FROM glm_runs WHERE modality=?", (modality,)
        ).fetchall()
        n_ok = sum(1 for _, st in rows if st == "ok")
        n_fail = sum(1 for _, st in rows if st != "ok")
        print(f"\n=== modality {modality} ===")
        print(f"runs ok={n_ok} failed={n_fail}")
        per_clip: dict[str, list[int]] = {}
        for sid, lab, st in conn.execute(
            "SELECT source_id,label,status FROM glm_runs WHERE modality=?", (modality,)
        ):
            if st == "ok" and lab is not None:
                per_clip.setdefault(sid, []).append(lab)
        unanimous = majority2 = total_clips = 0
        for _sid, values in per_clip.items():
            if len(values) < 3:
                continue
            total_clips += 1
            if len(set(values)) == 1:
                unanimous += 1
            if any(values.count(v) >= 2 for v in set(values)):
                majority2 += 1
        if total_clips:
            print(
                f"3-round unanimous={unanimous}/{total_clips} "
                f"({unanimous / total_clips:.1%}) "
                f"2of3-agree={majority2}/{total_clips} ({majority2 / total_clips:.1%})"
            )
        n_adj = conn.execute(
            "SELECT COUNT(*) FROM glm_adjudications WHERE modality=?", (modality,)
        ).fetchone()[0]
        print(f"adjudications={n_adj}")
        final_rows = conn.execute(
            "SELECT source_id,final_label,method FROM glm_final WHERE modality=?", (modality,)
        ).fetchall()
        dist = {-1: 0, 0: 0, 1: 0}
        for _, lab, _m in final_rows:
            if lab in dist:
                dist[lab] += 1
        print(f"final labels: -1={dist[-1]} 0={dist[0]} 1={dist[1]} (total={len(final_rows)})")
        # confusion vs truth
        conf_matrix = {t: {p: 0 for p in (-1, 0, 1)} for t in (-1, 0, 1)}
        joined = agree = 0
        for sid, lab, _m in final_rows:
            if lab is None or "/" not in sid:
                continue
            video_id, clip_id = sid.split("/", 1)
            t = truth.get((video_id, clip_id))
            if not t:
                continue
            joined += 1
            tv = t[f"label_{modality}"]
            conf_matrix[tv][lab] += 1
            if tv == lab:
                agree += 1
        if paths.meta_csv and joined:
            print(f"truth-joined={joined}/{len(final_rows)} agreement={agree / joined:.1%}")
            print("confusion (rows=truth, cols=pred):")
            print("        pred-1  pred0  pred1")
            for tv in (-1, 0, 1):
                print(
                    f"true{tv:+d}   {conf_matrix[tv][-1]:6d} {conf_matrix[tv][0]:6d} "
                    f"{conf_matrix[tv][1]:6d}"
                )
        elif paths.meta_csv:
            print("truth-joined=0 (no meta.csv match)")


# ---------------------------------------------------------------- offline self-test
def self_test() -> None:
    # parse_model_json: good/bad inputs
    assert parse_model_json('{"label": 1, "confidence": 0.9, "evidence": "ok"}') == (1, 0.9, "ok")
    assert parse_model_json(
        '```json\n{"label": -1, "confidence": 0.5, "evidence": "fenced"}\n```'
    ) == (-1, 0.5, "fenced")
    assert parse_model_json(
        '好的，结果如下：\n{"label": 0, "confidence": 1.4, "evidence": "clamped"} 以上。'
    ) == (0, 1.0, "clamped")
    for bad in (
        '{"label": 2, "confidence": 0.5, "evidence": "x"}',
        '{"label": "1", "confidence": 0.5, "evidence": "x"}',
        "no json here",
        "",
    ):
        try:
            parse_model_json(bad)
            raise AssertionError(f"should have failed: {bad!r}")
        except (ValueError, json.JSONDecodeError):
            pass
    # parse_model_json with rationale key (adjudication schema)
    assert parse_model_json(
        '{"label": 0, "confidence": 0.7, "rationale": "r"}', evidence_key="rationale"
    ) == (0, 0.7, "r")
    # frame subsets: 48 frames, cap 24
    assert pick_frame_indices(48, 24, 1) == list(range(24))
    assert pick_frame_indices(48, 24, 2) == list(range(24, 48))
    idx3 = pick_frame_indices(48, 24, 3)
    assert len(idx3) == 24 and len(set(idx3)) == 24 and idx3[0] == 0, idx3
    # 32 frames, cap 24
    assert pick_frame_indices(32, 24, 1) == list(range(24))
    assert pick_frame_indices(32, 24, 2) == list(range(8, 32))
    i3 = pick_frame_indices(32, 24, 3)
    assert len(i3) == 24 and len(set(i3)) == 24 and i3[0] == 0, i3
    # n <= cap: all three rounds identical
    assert pick_frame_indices(10, 48, 1) == pick_frame_indices(10, 48, 2) == list(range(10))
    # aggregate logic
    assert aggregate_round([(1, 0.9), (1, 0.8), (0, 0.7)])["decision"] == "majority"
    assert aggregate_round([(1, 0.9), (1, 0.8), (0, 0.7)])["label"] == 1
    assert aggregate_round([(1, 0.9), (0, 0.8), (-1, 0.7)])["decision"] == "adjudicate"
    assert aggregate_round([(1, 0.5), (1, 0.5), (0, 0.9)])["decision"] == "adjudicate"
    assert aggregate_round([(1, 0.9)])["decision"] == "needs_more_rounds"
    assert aggregate_round([(1, 0.9), (0, 0.8)])["decision"] == "adjudicate"
    print("[self-test] all assertions passed")


# ---------------------------------------------------------------- main
def main() -> None:
    args = parse_args()

    if args.self_test:
        self_test()
        return

    paths = Paths.resolve(args.curation_db, args.data_root, args.meta_csv)
    if not Path(paths.curation_db).is_file():
        print(f"FATAL: curation db not found: {paths.curation_db}", file=sys.stderr)
        sys.exit(2)
    if not paths.frames_root.is_dir():
        print(
            f"FATAL: frames root not found: {paths.frames_root} "
            f"(set --data-root / MPRISK_CURATION_DATA)",
            file=sys.stderr,
        )
        sys.exit(2)

    modalities = [m.strip().upper() for m in args.modalities.split(",") if m.strip()]
    clips = load_clips(paths.curation_db)
    print(f"[load] clips={len(clips)} empty_text={sum(1 for v in clips.values() if not v.strip())}")
    manifest = load_manifest(paths.manifest)
    conn = init_db(paths.out_db)

    if args.phase == "annotate":
        if args.dry_run:
            tasks, skipped = build_tasks(paths, clips, manifest, conn, modalities, 1)
            sid = sorted(clips)[0]
            files = frame_files(paths.frames_root, sid)
            n_ext = (manifest.get(sid) or {}).get("n_extracted", len(files))
            print(
                f"\n[dry-run] sample clip={sid} n_extracted={n_ext} "
                f"files_on_disk={len(files)} cap={MAX_FRAMES}"
            )
            for modality in modalities:
                sample = next((t for t in tasks if t["modality"] == modality), None)
                if modality == "V":
                    if sample is None:
                        print("[dry-run V] no task (no frames?)")
                        continue
                    for rnd in ROUNDS:
                        idx = pick_frame_indices(n_ext, MAX_FRAMES, rnd)
                        print(
                            f"  V round{rnd}: n_frames={len(idx)} first3={idx[:3]} last3={idx[-3:]}"
                        )
                    msgs = build_messages(
                        paths, "V", sid, clips[sid], pick_frame_indices(n_ext, MAX_FRAMES, 1)
                    )
                    kinds = [c["type"] for c in msgs[0]["content"]]
                    print(
                        f"  V message structure: roles=['user'] content types={kinds} "
                        f"(image urls are base64 data URLs, not shown)"
                    )
                else:
                    msgs = build_messages(paths, "T", sid, clips[sid], [])
                    print(f"  T message structure: {json.dumps(msgs, ensure_ascii=False)[:300]}")
            print(
                f"\n[dry-run] would-run tasks={len(tasks)} skipped(done)={len(skipped)} "
                f"total_clips={len(clips)}"
            )
            return
        run_annotate(paths, clips, manifest, conn, modalities, args.limit, args.api_key_file)
    elif args.phase == "aggregate":
        run_aggregate(paths, conn, clips, modalities, args.limit)
    elif args.phase == "adjudicate":
        run_adjudicate(paths, conn, clips, modalities, args.api_key_file)
    elif args.phase == "report":
        run_report(paths, conn, clips, modalities)
    conn.close()


if __name__ == "__main__":
    main()
