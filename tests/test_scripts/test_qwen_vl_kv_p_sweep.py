from pathlib import Path

from scripts.run_qwen_vl_kv_p_sweep import _aggregate, _init_db, _read_pool


def test_pool_and_ledger_contract(tmp_path: Path) -> None:
    pool = _read_pool(Path("data/processed/prompt_banks/pregen_risk_v1_agent/pool128.jsonl"))
    assert len(pool) >= 64
    assert len({row["prompt_id"] for row in pool}) == len(pool)
    connection = _init_db(tmp_path / "ledger.sqlite3")
    connection.execute(
        "INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (1, "sample", 0, "complete", 12.0, '{"M1":4,"M2":3,"M12":5}', 1,
         "degenerate_single_prompt_full_prefill", None, None, 0.0),
    )
    connection.execute(
        "INSERT INTO state_metrics VALUES(?,?,?,?)",
        (1, "sample", '{"state":{"S_mean":0.1,"D":0.2,"R":0.3,"R_bootstrap_se":0.0}}', 0.0),
    )
    connection.commit()
    summary = _aggregate(connection, 1)
    assert summary["timing"]["total_median_ms"] == 12.0
    assert summary["state_metrics"]["D_mean"] == 0.2
    connection.close()
