from __future__ import annotations

import json
from pathlib import Path

from curation.scripts.audit_local_datasets import audit_datasets, main


def test_audit_writes_json_and_markdown_with_counts_and_modalities(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets"
    output_dir = tmp_path / "reports"

    ch_sims_v2 = dataset_root / "ch_sims_v2"
    (ch_sims_v2 / "videos").mkdir(parents=True)
    (ch_sims_v2 / "audio").mkdir()
    (ch_sims_v2 / "labels.csv").write_text(
        "sample_id,text,vision,audio,label\ns1,hello,0.2,0.3,positive\n",
        encoding="utf-8",
    )
    (ch_sims_v2 / "labels.pkl").write_bytes(b"binary-labels")
    (ch_sims_v2 / "videos" / "clip.mp4").write_bytes(b"video")
    (ch_sims_v2 / "audio" / "clip.wav").write_bytes(b"audio")

    cmu_mosi = dataset_root / "cmu_mosi"
    (cmu_mosi / "frames").mkdir(parents=True)
    (cmu_mosi / "labels.jsonl").write_text(
        '{"id": "m1", "text": "hi", "sentiment": 0.7}\n',
        encoding="utf-8",
    )
    (cmu_mosi / "frames" / "frame.jpg").write_bytes(b"image")

    exit_code = main(["--dataset-root", str(dataset_root), "--output-dir", str(output_dir)])

    assert exit_code == 0
    json_path = output_dir / "dataset_audit.json"
    md_path = output_dir / "DATASET_AUDIT.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    datasets = {item["dataset_key"]: item for item in payload["datasets"]}

    assert set(datasets) == {"ch_sims", "ch_sims_v2", "cmu_mosi", "cmu_mosei"}
    assert datasets["ch_sims_v2"]["exists"] is True
    assert datasets["ch_sims_v2"]["file_count"] == 4
    assert datasets["ch_sims_v2"]["detected_modalities"] == {
        "video": True,
        "audio": True,
        "text": True,
        "image": False,
    }
    assert datasets["ch_sims_v2"]["label_files"] == ["labels.csv", "labels.pkl"]
    assert datasets["ch_sims_v2"]["label_columns_by_file"]["labels.csv"] == [
        "sample_id",
        "text",
        "vision",
        "audio",
        "label",
    ]
    assert datasets["ch_sims_v2"]["label_columns_by_file"]["labels.pkl"] == []
    assert datasets["ch_sims_v2"]["protocol_support"] == {
        "VT_native": True,
        "VA_native": True,
        "IT_derived": False,
    }

    assert datasets["cmu_mosi"]["protocol_support"] == {
        "VT_native": False,
        "VA_native": False,
        "IT_derived": True,
    }
    assert datasets["ch_sims"]["exists"] is False
    assert "pending audit" in " ".join(datasets["ch_sims"]["notes"])
    assert "ch_sims_v2" in md_path.read_text(encoding="utf-8")


def test_ch_sims_protocol_support_requires_local_hints(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets"
    ch_sims = dataset_root / "ch_sims"
    ch_sims.mkdir(parents=True)
    (ch_sims / "metadata.tsv").write_text(
        "clip_id\ttext\tvision_label\taudio_label\tmultimodal_label\n"
        "c1\twords\tpositive\tnegative\tneutral\n",
        encoding="utf-8",
    )

    payload = audit_datasets(dataset_root)
    item = next(dataset for dataset in payload["datasets"] if dataset["dataset_key"] == "ch_sims")

    assert item["detected_modalities"]["text"] is True
    assert item["protocol_support"]["VT_native"] is True
    assert item["protocol_support"]["VA_native"] is True
    assert not any("pending audit" in note for note in item["notes"])
