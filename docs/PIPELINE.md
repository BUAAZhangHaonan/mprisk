# Pipeline

The pipeline is designed around traceable paper artifacts.

## 1. Data Preparation

Inputs are raw datasets and annotation files. Outputs are normalized sample manifests, sample-type labels, protocol views, and deterministic splits.

Data preparation now starts in the independent `curation/` module:

```text
source data -> initial screening -> LLM screening -> human review -> adjudication -> final manifests
```

Main artifacts:

- `data/processed/manifests/unified_sample_manifest.jsonl`
- `data/processed/manifests/conflict_manifest.jsonl`
- `data/processed/manifests/aligned_manifest.jsonl`
- `data/processed/manifests/protocol_manifests/*.jsonl`
- `data/processed/manifests/delivery_20260714.provenance.json`

Validate the frozen delivery and derive deterministic split/protocol manifests:

```bash
python scripts/build_manifests.py --repo-root .
```

The builder verifies the frozen archive and tracked artifact hashes, the current machine-label
inclusion policy, media existence, variety-text exclusions, and real/generated source boundaries.
It assigns splits only from `split_group_id`, so VT/VA rows from the same source cannot cross
train, validation, and test.

Curation intermediate artifacts:

- `curation/outputs/candidates/*.jsonl`
- `curation/outputs/llm_screening/*.jsonl`
- `curation/outputs/human/*.jsonl`
- `curation/outputs/adjudicated/*.jsonl`
- `curation/outputs/exports/*.jsonl`

## 2. Prompt Banks

Prompt banks define equivalent task formulations for each protocol. The prompt-pool path uses a reviewed raw set of 384 AI candidates, filters it to a deterministic global pool of 128 placeholder-free prompts, and draws three `P = 8` subsets with seeds `20260715`, `20260716`, and `20260717`.

The local generator records the model path, decoding settings, and canonical task. It writes
raw model candidates separately from reviewed candidates; the pool builder consumes
`ai_reviews.jsonl`, not the raw-only file.

```bash
python scripts/generate_prompt_candidates.py \
  --model-path /home/team/lvshuyang/Models/Qwen/Qwen2.5-3B-Instruct \
  --output-dir data/processed/prompt_banks/pregen_risk_v1

python scripts/build_prompt_pool.py \
  --raw-candidates data/processed/prompt_banks/pregen_risk_v1/ai_reviews.jsonl \
  --output-dir data/processed/prompt_banks/pregen_risk_v1 \
  --prompt-set-key pregen_risk_v1 \
  --protocol vt
```

Main artifacts:

- `data/processed/prompt_banks/vt_primary_bank_v1.jsonl`
- `data/processed/prompt_banks/va_aux_bank_v1.jsonl`
- `data/processed/prompt_banks/it_aux_bank_v1.jsonl`
- `configs/prompts/prompt_selection.yaml`
- `configs/prompts/equiv_sets/*.yaml`

## 3. Pre-generation Cache

For each model, dataset, protocol, split, and condition, the extraction pipeline stores full-layer hidden trajectories at `t0`.

Main artifacts:

- `outputs/full_cache/manifests/unified_full_cache_manifest.json`
- `outputs/full_cache/manifests/extraction_ledger.csv`

## 4. State-Data Surface

The first implementation phase connects final labels to cache entries. It does not train encoders and does not compute `S`, `D`, or `R`.

The chain is:

```text
final manifests -> cache surface -> t0 trajectory bundle -> state dataset manifest
```

Run the exporter:

```bash
python scripts/build_state_dataset.py \
  --manifest data/processed/manifests/conflict_manifest.jsonl \
  --manifest data/processed/manifests/aligned_manifest.jsonl \
  --split-assignment data/processed/manifests/splits/representation_v1/representation_split_assignment_v1.jsonl \
  --model-key qwen3_vl_8b \
  --protocol VT
```

Run the smoke check:

```bash
python scripts/verify_state_data_pipeline.py \
  --manifest data/processed/manifests/conflict_manifest.jsonl \
  --manifest data/processed/manifests/aligned_manifest.jsonl \
  --split-assignment data/processed/manifests/splits/representation_v1/representation_split_assignment_v1.jsonl \
  --model-key qwen3_vl_8b \
  --protocol VT
```

Main artifacts:

- `outputs/state_data/{model_key}/{protocol}/state_dataset_manifest.jsonl`
- `outputs/state_data/{model_key}/{protocol}/state_dataset_summary.json`
- `outputs/state_data/{model_key}/{protocol}/missing_cache_rows.jsonl`
- `outputs/state_data/reports/STATE_DATA_SMOKE.md`

`state_dataset_manifest.jsonl` stores cache indexes and trajectory metadata only. It does not copy hidden-state tensors.

## 5. Representation

All representation supervision is the sample-level `Conflict`/`Aligned` label. View
labels and all Misread-derived fields are forbidden in relation datasets. Data is split
by the committed group-level assignment
`data/processed/manifests/splits/representation_v1/representation_split_assignment_v1.jsonl`.
Missing `master_split` or registered `representation_split` fields are fatal; training
never hashes the full dataset into a new split.

The pre-registered split rule is in
`configs/splits/representation_split_v1.yaml`. Official train is unchanged and is the
only encoder-training partition. Official test is unchanged and is reserved for final
evaluation. The registered scope is all 4,754 valid `Conflict`/`Aligned` rows in the
VT and VA protocol manifests. The legacy `use_in_main` field is retained as provenance
but does not filter representation or state data. No view-level affect label or
`is_clear` field participates in inclusion or supervision. Among official-validation
source groups containing only Aligned samples,
groups are ranked without replacement by `sha256(seed:split_group_id)` using seed
`20260716`; floor 50% are assigned to `aligned_calibration`. Remaining Aligned groups
and every Conflict validation group form `relation_val`. A group cannot cross
`relation_train`, `relation_val`, `aligned_calibration`, or `official_test`, and the same
artifact is shared by all models, protocols, prompts, and representation families.

Rebuild and verify the versioned assignment:

```bash
python scripts/build_representation_splits.py \
  --config configs/splits/representation_split_v1.yaml \
  --output-dir data/processed/manifests/splits/representation_v1
```

The three independent backbone-specific interfaces are:

- `single_point_binary_v1`: versioned final-layer M1/M2/M12 concatenation (`3H`)
  passed directly to a two-logit linear classifier. Configs pin the same explicit
  `architecture_version`; checkpoints containing the retired hidden-projection drift
  are rejected before state loading.
- `trajectory_mlp_binary_v1`: complete per-layer L2-normalized M1/M2/M12 trajectories and a two-logit MLP classifier.
- `tme_proxy_anchor_v1`: architecture `layer_l2_gru_linear_relation_v1`, a shared one-layer GRU over the complete normalized layer sequence, followed by a compact linear projection and unit-normalized condition embedding `z`.

TME computes only the ordered relation features
`u=[1-z1.z2, 1-z12.z1, 1-z12.z2]` and `r=normalize(Wu+b)`. The relation head has no
concatenation, activation, or nonlinear branch. Its only objective is standard Proxy
Anchor with exactly two proxies (`Aligned=0`, `Conflict=1`). SupCon, prompt-consistency,
and cross-entropy are not part of the TME objective. Checkpoint selection uses only
validation balanced accuracy over `sample_type.Aligned` and `sample_type.Conflict`, with
class-code mapping `C=Aligned` and `A=Conflict`. Training may use prompt rows as
augmentation, but validation aggregates all synchronized prompt outputs by `sample_id`
before making one prediction: TME averages and unit-normalizes `r` before proxy scoring,
while baseline classifiers average logits. Early stopping therefore counts each held-out
sample exactly once.

Prompt rows are training augmentations, not independent supervised examples. In each
epoch, every training `sample_id` contributes exactly one prompt selected by the
versioned deterministic rule `(seed, epoch, sample_id)`; selection rotates across the
synchronized prompt set and is identical after checkpoint resume. Validation and test
continue to aggregate every prompt.

Every representation config pins `expected_prompt_count=8`, the exact eight prompt
IDs, `prompt_set_key`, and the prompt-set artifact SHA-256. Training, validation, and
frozen export reject a sample whose M1/M2/M12 rows do not use that exact synchronized
set; a uniformly truncated seven-prompt dataset is an error rather than a smaller run.
Spherical calibration is bound to `model_key`, protocol, prompt-set identity,
representation key, encoder checkpoint SHA-256, split assignment SHA-256, and embedding
manifest SHA-256. Score-to-pattern assignment compares every field and rejects reused
thresholds from another backbone, prompt seed, checkpoint, split, or embedding export.

Training indexes only relation metadata in memory. M1/M2/M12 trajectories are sliced
from safetensors with `safe_open` when each bounded batch is consumed; cache tensors
are never converted to nested Python lists or retained for the full dataset. Frozen
`z`/`r` manifests and sample bundles are written incrementally through atomic temporary
files, with at most one sample's prompt bundle accumulated at a time.

Every spherical normalization is a hard contract. Per-layer trajectory vectors, TME
condition projections `z`, ordered relation projections `r`, Proxy Anchor embeddings,
and both class proxies must have norm greater than `1e-12` before normalization. A
zero or non-finite vector raises an explicit error containing its stage and sample (or
proxy-class) identity; zero vectors are never silently mapped to zero by normalization.

Single-Point and Trajectory MLP remain ordinary two-class cross-entropy baselines.
Single-Point's frozen feature is exactly the M1/M2/M12 final-layer point concatenation
(`3H`), which is also the direct input to its linear two-logit classifier; it has no
extra projection, activation, or spherical normalization. Trajectory MLP's frozen
feature is the `hidden_dim=128` output after its first linear layer and GELU over the
complete `3 x L x H` layer-normalized trajectory. Their held-out exporter streams cache
batches, averages these features and logits over all eight synchronized prompts for
each sample, and writes one frozen row per sample. TME keeps prompt-level `z` and `r`
for state analysis and additionally exports one sample feature by averaging the eight
ordered `r32` prompt vectors and then unit-normalizing the mean. None of these exports
uses Misread labels:

```bash
python scripts/export_baseline_representations.py \
  --dataset outputs/representation_data/<model>/<protocol>/<prompt_set>/relation_dataset.jsonl \
  --checkpoint outputs/representation_train/<model>/<protocol>/<prompt_set>/<repr>/best_checkpoint.pt \
  --representation-split official_test \
  --output-dir outputs/frozen_baselines/<model>/<protocol>/<prompt_set>/<repr>
```

The trained representation smoke chain is:

```text
bundle_manifest
-> relation_dataset
-> tme_proxy_anchor_v1 checkpoint
-> frozen condition-z and relation-r manifests
-> S/D/R scores
-> state patterns
```

Run the trained representation smoke pipeline:

```bash
python scripts/run_representation_training_smoke.py \
  --bundle-manifest outputs/state_bundles/qwen3_vl_8b/VT/vt_primary_v1/bundle_manifest.jsonl \
  --config configs/experiments/representation_qwen3_vl_8b_tme_proxy_anchor_v1.yaml \
  --model-key qwen3_vl_8b \
  --protocol VT \
  --prompt-set-key vt_primary_v1 \
  --output-root . \
  --device cuda \
  --thresholds outputs/states/calibration/qwen3_vl_8b_vt_thresholds.json
```

Main artifacts:

- `outputs/representation_data/{model_key}/{protocol}/{prompt_set_key}/relation_dataset.jsonl`
- `outputs/representation_train/{model_key}/{protocol}/{prompt_set_key}/tme_proxy_anchor_v1/best_checkpoint.pt`
- `outputs/representation_train/{model_key}/{protocol}/{prompt_set_key}/tme_proxy_anchor_v1/last_checkpoint.pt`
- `outputs/representation_train/{model_key}/{protocol}/{prompt_set_key}/tme_proxy_anchor_v1/train_config.yaml`
- `outputs/representation_train/{model_key}/{protocol}/{prompt_set_key}/tme_proxy_anchor_v1/train_metrics.json`
- `outputs/representation_train/{model_key}/{protocol}/{prompt_set_key}/tme_proxy_anchor_v1/train_log.jsonl`
- `outputs/representation/{model_key}/{protocol}/{prompt_set_key}/tme_proxy_anchor_v1/frozen_representations.jsonl`
- `outputs/representation/{model_key}/{protocol}/{prompt_set_key}/tme_proxy_anchor_v1/spherical_embedding_manifest.jsonl`
- `outputs/states/{model_key}/{protocol}/{prompt_set_key}/tme_proxy_anchor_v1/sdr_scores.jsonl`
- `outputs/states/{model_key}/{protocol}/{prompt_set_key}/tme_proxy_anchor_v1/state_patterns.jsonl`
- `outputs/representation_train/reports/REPRESENTATION_TRAINING_SMOKE.md`

## 6. State Analysis

For each condition, the spherical center is the normalized mean across synchronized
prompts. Let `d_g(a,b)=acos(clip(a^T b,-1,1))` and
`mu_c=normalize(sum_p z_cp)`. Per-condition dispersion is
`s_c=(1/P) sum_p d_g(z_cp,mu_c)^2`, with `S=(s_1+s_2+s_12)/3`.
The normalized modality split is
`D=d_g(mu_1,mu_2)/(sqrt(s_1+s_2)+eps)`, and signed Joint Lean is
`R=(d_g(mu_12,mu_2)-d_g(mu_12,mu_1))/(d_g(mu_1,mu_2)+eps)`.
Therefore `R>0` is V lean and `R<0` is T/A lean. Each sample uses
`delta_i=1.96*SE` from a synchronous prompt bootstrap: every replicate resamples
one shared prompt-index vector for all three conditions and recomputes the same
center-based R formula.

Thresholds are calibrated on a separate `aligned_calibration` split only. `kappa` is
the q95 Aligned S value; `tau` is the q95 D value among stable Aligned rows where
`S<=kappa`. State assignment is strictly hierarchical: Confusion (`S>kappa`), then
Consensus (`D<=tau`), then Balanced (`abs(R)<=delta_i`), otherwise Dominant.

The second implementation phase adds the first runnable science core:

```text
state_dataset_manifest
-> prompt-conditioned cache manifest
-> prompt-conditioned bundle manifest
-> raw trajectory embedding manifest
-> S/D/R scores
-> state patterns
```

The prompt-conditioned cache is the source of truth for `state(sample, view, prompt_id)`.
The bundle must not reuse the same view-level `state_cache` for all prompts.

Build or normalize the prompt-conditioned cache manifest from existing model-environment outputs:

```bash
python scripts/build_prompt_conditioned_cache.py \
  --mode A \
  --source-manifest outputs/prompt_conditioned_cache/source_rows.jsonl \
  --model-key qwen3_vl_8b \
  --protocol VT \
  --prompt-set-key vt_primary_v1
```

Run the bundle builder:

```bash
python scripts/build_state_bundles.py \
  --state-dataset-manifest outputs/state_data/qwen3_vl_8b/VT/state_dataset_manifest.jsonl \
  --prompt-cache-manifest outputs/prompt_cache/qwen3_vl_8b/vt_primary_v1/manifest.jsonl \
  --prompt-conditioned-cache-manifest outputs/prompt_conditioned_cache/qwen3_vl_8b/vt/vt_primary_v1/manifest.jsonl \
  --prompt-set configs/prompts/equiv_sets/vt_primary_v1.yaml \
  --prompt-set-key vt_primary_v1 \
  --model-key qwen3_vl_8b \
  --protocol VT
```

Run the smoke chain:

```bash
python scripts/run_state_measurement_smoke.py \
  --state-dataset-manifest outputs/state_data/qwen3_vl_8b/VT/state_dataset_manifest.jsonl \
  --prompt-cache-manifest outputs/prompt_cache/qwen3_vl_8b/vt_primary_v1/manifest.jsonl \
  --prompt-conditioned-cache-manifest outputs/prompt_conditioned_cache/qwen3_vl_8b/vt/vt_primary_v1/manifest.jsonl \
  --prompt-set configs/prompts/equiv_sets/vt_primary_v1.yaml \
  --prompt-set-key vt_primary_v1 \
  --model-key qwen3_vl_8b \
  --protocol VT \
  --repr-key raw_layernorm_mean
```

Run the minimal core SDR pipeline from final manifests and cache manifests:

```bash
python scripts/run_core_sdr_pipeline.py \
  --model-key qwen3_vl_8b \
  --protocol VT \
  --prompt-set-key vt_primary_v1 \
  --repr-key tme_proxy_anchor_v1 \
  --manifest-paths data/processed/manifests/conflict_manifest.jsonl data/processed/manifests/aligned_manifest.jsonl \
  --full-cache-root . \
  --prompt-cache-manifest outputs/prompt_cache/qwen3_vl_8b/vt_primary_v1/manifest.jsonl \
  --prompt-conditioned-cache-manifest outputs/prompt_conditioned_cache/qwen3_vl_8b/vt/vt_primary_v1/manifest.jsonl \
  --prompt-set configs/prompts/equiv_sets/vt_primary_v1.yaml \
  --split-assignment data/processed/manifests/splits/representation_v1/representation_split_assignment_v1.jsonl \
  --output-root . \
  --checkpoint outputs/representation_train/qwen3_vl_8b/VT/vt_primary_v1/tme_proxy_anchor_v1/best_checkpoint.pt \
  --thresholds outputs/states/calibration/qwen3_vl_8b_vt_thresholds.json
```

The final core runner accepts only `tme_proxy_anchor_v1` and requires an existing
checkpoint. Raw layer-normalized representations cannot stand in for final TME.

Main artifacts:

- `outputs/prompt_conditioned_cache/{model_key}/{protocol}/{prompt_set_key}/manifest.jsonl`
- `outputs/state_bundles/{model_key}/{protocol}/{prompt_set_key}/bundle_manifest.jsonl`
- `outputs/representation/{model_key}/{protocol}/{prompt_set_key}/{repr_key}/embedding_manifest.jsonl`
- `outputs/states/scores/`
- `outputs/states/assignments/`
- `outputs/states/summaries/`
- `outputs/states/{model_key}/{protocol}/{prompt_set_key}/{repr_key}/sdr_scores.jsonl`
- `outputs/states/{model_key}/{protocol}/{prompt_set_key}/{repr_key}/state_patterns.jsonl`
- `outputs/states/{model_key}/{protocol}/{prompt_set_key}/{repr_key}/state_summary.json`
- `outputs/states/{model_key}/{protocol}/{prompt_set_key}/{repr_key}/CORE_SDR_SUMMARY.md`
- `outputs/states/reports/STATE_MEASUREMENT_SMOKE.md`

## 7. Baselines and Evaluation

Baselines include simple behavior signals, uncertainty methods, classifier risk, and post-hoc full-response analysis.

Main artifacts:

- `outputs/baselines/`
- `outputs/evaluation/`

## 8. Paper Export

Figures and tables are generated from output artifacts.

Main artifacts:

- `paper/figures/generated/`
- `paper/tables/generated/`
- `outputs/paper_exports/`
