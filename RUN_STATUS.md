# RUN STATUS

Live snapshot: `2026-07-16T05:37:51+08:00`. Long jobs are resumable and remain attached to tmux on host `6403`.

## Active commands

| Stage | Status | tmux | PID | Physical GPU | Command |
|---|---|---|---:|---:|---|
| Main P=8 cache queue | Running (`qwen3_vl_8b`, seed `20260717`) | `mprisk-main-p8-queue` | 1144732 (worker 1144933) | 1 | `CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python scripts/extract_prefill_batch.py --manifest data/processed/manifests/protocol_manifests/vt_primary.jsonl --prompt-set configs/prompts/equiv_sets/vt_main_p8_seed20260717.yaml --protocol vt --model-key qwen3_vl_8b --device cuda:0 --output-root outputs/prefill_cache/qwen3_vl_8b/vt_main_p8_seed20260717 --retry-failed --fail-fast --materialize-every 100` |
| Follow-up P=8 cache queue | Waiting for main queue | `mprisk-followup-p8-queue` | 3631738 | 1 | `CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python scripts/run_prefill_dependency_queue.py --config configs/cache/prefill_followup_p8_queue_v1.yaml` |
| Three-seed downstream queue | Running; waiting at cache/registered-GPU-resource gate | `mprisk-downstream-three-seed` | 1210927 | 1 | `CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python scripts/run_downstream_queue.py --config configs/downstream/tme_three_seed_queue_v1.yaml` |
| Figure export | Success | completed | - | CPU | `python scripts/export_paper_figures.py --config configs/paper/figure_map.yaml` |
| Table export | Success | completed | - | CPU | `python scripts/export_paper_tables.py --config configs/paper/table_map.yaml` |

The main queue runs `qwen3_vl_8b`, `internvl3_5_8b`, then `qwen2_5_omni_7b`. The follow-up queue runs the same three backbones for seeds `20260715` and `20260716`. The downstream producer guard prevents a training process from racing an extractor on GPU 1. After the producers exit, one explicitly registered ComfyUI command may retain at most 512 MiB; unknown commands, a second matching process, per-process overflow, aggregate external-context overflow, and total GPU occupancy at or above 85% are all rejected.

## GPU snapshot

| GPU | Name | Memory used / total (MiB) | Utilization | Use |
|---:|---|---:|---:|---|
| 0 | NVIDIA A100 80GB PCIe | 72,331 / 81,920 (88.3%) | 0% | External process; excluded from mprisk |
| 1 | NVIDIA A100 80GB PCIe | 23,058 / 81,920 (28.1%) | 98% | Active hidden-state extraction; includes the registered 416 MiB ComfyUI context |

## Hidden-state cache

Every run has `P=8` prompts and conditions `M1`, `M2`, and `M12`. `Missing` means all ledger rows not yet completed, including running rows. An em dash means the queue has not materialized that ledger yet.

| Model | Protocol | Prompt seed | Expected | Completed | Running | Pending | Failed | Missing | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `qwen3_vl_8b` | V-T | 20260717 | 60,288 | 12,860 | 1 | 47,427 | 0 | 47,428 | Running |
| `internvl3_5_8b` | V-T | 20260717 | 60,288 | 0 | - | - | - | 60,288 | Queued after Qwen3-VL |
| `qwen2_5_omni_7b` | V-A | 20260717 | 53,808 | 0 | - | - | - | 53,808 | Queued after InternVL |
| `qwen3_vl_8b` | V-T | 20260715 | 60,288 | 0 | - | - | - | 60,288 | Follow-up queued |
| `internvl3_5_8b` | V-T | 20260715 | 60,288 | 0 | - | - | - | 60,288 | Follow-up queued |
| `qwen2_5_omni_7b` | V-A | 20260715 | 53,808 | 0 | - | - | - | 53,808 | Follow-up queued |
| `qwen3_vl_8b` | V-T | 20260716 | 60,288 | 0 | - | - | - | 60,288 | Follow-up queued |
| `internvl3_5_8b` | V-T | 20260716 | 60,288 | 0 | - | - | - | 60,288 | Follow-up queued |
| `qwen2_5_omni_7b` | V-A | 20260716 | 53,808 | 0 | - | - | - | 53,808 | Follow-up queued |

The one-Conflict/one-Aligned smoke gate passed for all three backbones before the full queues started (48/48 condition-prompt tasks per backbone).

## Registered representation split

Source: `data/processed/manifests/splits/representation_v1/representation_split_summary_v1.json`.

| Split | Samples | Aligned | Conflict | Role |
|---|---:|---:|---:|---|
| `relation_train` | 3,354 | 2,943 | 411 | Conflict/Aligned representation training |
| `relation_val` | 376 | 302 | 74 | Convergence and model selection |
| `aligned_calibration` | 290 | 290 | 0 | Independent calibration of kappa and tau only |
| `official_test` | 734 | 634 | 100 | All main-paper metrics and figure inputs |
| **Total** | **4,754** | **4,169** | **585** | Group-disjoint registered split |

- Split key: `representation_split_seed20260716_all_ac_v2`
- Manifest SHA-256: `681114e1d8e9a94a1ac243d44b4194327431eb2bdaf5557de87a0e58cd404475`
- Split-assignment checksum: `bd784c316e2b6ff886c5223b749745606fecf8155c6122b0c0ebe79a415c88b0`

## Downstream experiments

| Experiment | Progress | Status / output |
|---|---:|---|
| Valid cache runs | 0 / 9 | Waiting for complete SQLite ledger, manifest, sidecars, safetensors and checksums |
| Single-Point MLP | 0 / 9 | Queued; ordinary Conflict/Aligned cross-entropy |
| Trajectory MLP | 0 / 9 | Queued; ordinary Conflict/Aligned cross-entropy |
| TME Proxy Anchor | 0 / 9 | Queued; full layer trajectory, spherical condition codes, ordered three-condition relation, Proxy Anchor only |
| All representation runs | 0 / 27 | `outputs/downstream/three_seed_v1/queue_status.json`; PID `1210927`; gate reason `cache_or_registered_gpu_resource_gate` |
| State calibration and S/D/R/State Pattern | 0 / 9 | Runs after each converged TME checkpoint; test outputs use `official_test` only |
| Conflict-retention sensitivity | 0 / 3 models | Runs for the main prompt seed at 10%, 25%, 50%, and 100%; this is Conflict classification, not Misread |
| Three-seed aggregation | 0 / 3 models | Paired by `(model, sample)`; seed mean, sample SD (`ddof=1`), and 95% CI with `t(df=2)` |
| Conflict-only Misread probe | Pending | `outputs/downstream/three_seed_v1/misread_probe/PENDING.json`; no Misread annotations exist, and no pseudo-labels are generated |

## PDF visual QA

All 24 PDFs are openable vector layouts. Ten main PDFs are in `paper/figures/generated/`; fourteen appendix PDFs are in `paper/figures/appendix/`. Layout readiness does not imply that pending result cells contain measurements.

| Artifact group | Status | Notes |
|---|---|---|
| Fig. 1-3 | Ready | Data-independent protocol, representation, and spherical S/D/R diagrams |
| Fig. 4-6 | Openable layout; Pending cache/training/state data | S/D/abs(R) distributions, four State Pattern proportions, and stable-sample D-signed-R geometry |
| Fig. 7 | Openable layout; Pending | Misread relationship panels say `Pending Misread annotations`; real modality-bias panels wait for TME outputs |
| Fig. 8 | Openable layout; Pending | Conflict/Aligned representation panels wait for runs; Misread AUPRC panels remain blank/Pending |
| Fig. 9-10 | Openable layout; Pending cache/training/state data | No illustrative numerical results are substituted |
| Appendix Fig. B1 | Ready | Data-independent architecture detail |
| Appendix A1, B2-B3, C1-C5, D3, E2 | Openable layout; Pending cache/training/state data | Current A/C experiments are not presented as Misread experiments |
| Appendix A2, D1, E1 | Openable layout; Pending Misread annotations | No pseudo-labels or pseudo-curves |

## Generated tables

| Artifact | Status | Path |
|---|---|---|
| Table I, cross-backbone results | Openable layout; Pending real downstream/Misread cells | `paper/tables/generated/tab01_cross_backbone_results.tex` |
| Table II, Conflict-only Misread baselines | Pending Misread annotations | `paper/tables/generated/tab02_conflict_misread_baselines.tex` |
| Table III, downstream quality | Openable layout; Pending real measurements and annotation-dependent cells | `paper/tables/generated/tab03_downstream_quality.tex` |

No `raw_layernorm` smoke result, Conflict/Aligned score, Conflict-retention curve, or State Pattern label is reported as a Misread result.
