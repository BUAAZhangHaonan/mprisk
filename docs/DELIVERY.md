# mprisk delivery status

This document records the repository state delivered on 2026-07-25. It separates
the reviewed code release from the long-running cache extraction checkout.

## Release boundary

- Delivery branch: `master`
- Delivery commit at the start of this record: `529cffd`
- Delivery clone: `/home/team/zhanghaonan/mprisk-delivery-final-2`
- Active extraction checkout: `/home/team/zhanghaonan/TAFFC/mprisk`
- Active extraction commit: `c25587e`

The active extraction checkout is intentionally frozen at `c25587e` until the
running source-cache jobs finish. It was not modified during repository cleanup,
review, or delivery documentation. Release commits must not be copied into that
checkout while its ledgers and workers are live.

## Completed milestones

| Commit | Delivered change |
| --- | --- |
| `0360ec7` | Removed retired scaffolds, backup files, stale run material, and repository clutter; expanded ignore rules. |
| `b785354` | Hardened cache identity, sidecar/checksum validation, completion audits, and dual-lane cleanup. |
| `77e6ce7` | Defined the strict cross-domain delivery matrix and fail-closed bundle builder. |
| `1db5f1f` | Repaired portable contracts and wrapper semantics, including Gemma 4, Phi-4, and LLaVA-OneVision checks. |
| `9870edd` | Canonicalized internal artifact, configuration, script, and visualization names. |
| `b6dd265` | Published the canonical real-data figure exports while preserving prior figure artifacts. |
| `1556e55` | Removed dead placeholder modules. |
| `59a62ae` | Locked the final TME representation and supervision protocol. |
| `c69eaab` | Consolidated representation experiment configurations. |
| `6a28081` | Removed the unsupported prompt-cache execution mode. |
| `529cffd` | Added the strict portable-locator resolver. |

## Validation record

- The latest full repository test run before the portable-locator resolver
  landed completed with **600 passed**.
- The portable-locator resolver change completed **20 targeted tests**.
- The resolver commit has not yet been followed by another recorded full-suite
  run; the two results above are therefore reported separately.
- The cleanup milestones also passed their focused lint, compile, configuration,
  and fail-closed packaging checks before they were pushed.

## Canonical paper artifacts

The canonical main-paper PDF locations are:

- `paper/figures/generated/fig01_problem_protocol.pdf`
- `paper/figures/generated/fig02_representation_pipeline.pdf`
- `paper/figures/generated/fig03_spherical_sdr.pdf`
- `paper/figures/generated/fig04_sdr_distributions.pdf`
- `paper/figures/generated/fig05_four_state_stacks.pdf`
- `paper/figures/generated/fig06_stable_d_signed_r.pdf`
- `paper/figures/generated/fig07_misread_bias.pdf`
- `paper/figures/generated/fig08_representation_comparison.pdf`
- `paper/figures/generated/fig09_conflict_case.pdf`
- `paper/figures/generated/fig10_four_pattern_cases.pdf`

The canonical Table 2 export is
`outputs/paper_exports/tables/misread/tab02_conflict_misread_baselines.csv`.
Its latency column remains **Pending** because the registered Misread-probe queue
artifacts do not contain probe latency. No value is inferred or substituted.

## Live cache snapshot

The following source-cache snapshot was recorded at **2026-07-25 03:52 +08:00**:

| State | Count |
| --- | ---: |
| Accepted from the frozen delivery bundle | 1 |
| Complete | 11 |
| Ready or still queued | 4 |

The two running model ledgers reported:

| Model | Completed | Expected |
| --- | ---: | ---: |
| `qwen3_vl_8b` | 26,873 | 45,024 |
| `qwen2_5_omni_7b` | 26,600 | 46,416 |

Both running ledgers reported **0 failures**. Target-domain cache extraction had
not started. The stage controller remains fail-closed and must not start the
target stage until the source-stage audit is complete.

## Remaining release blockers

One P1 portability issue remains:

- Portable-locator production consumers have not all migrated to the strict
  resolver.
- The current absolute-path audit reports **42 occurrences in 12 tracked files**.

These paths must be classified and migrated before the release is portable
across machines. Dataset identities, model versions, protocol schema versions,
and immutable delivery identifiers are not internal iteration names and should
not be renamed merely to remove version text.
