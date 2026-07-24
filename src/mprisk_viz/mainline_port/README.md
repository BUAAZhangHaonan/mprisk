# Mainline Port Marker (2026-07-19)

This directory is a marker. Mainline code lives in:
- `mprisk-v2/src/mprisk/representation/losses.py` (contains `ModalitySplitRankingLoss`)
- `mprisk-v2/src/mprisk/representation/training.py` (schema v4 with `enable_state_supervision`)

v2 originals preserved as `*.v2_orig_20260719`.

## Three training schemes
| Scheme | Code Path | Enable |
|--------|-----------|--------|
| A. mainline | losses.py:ModalitySplitRankingLoss + training.py two-stage backward | config: `enable_state_supervision: true`, `d_supervision_weight: 0.2`, `angular_supervision_weight: 0.2`, `d_ranking_margin: 0.25`, `angular_ranking_margin_rad: 0.0873` |
| B. v2-current | mprisk_v2/sdr_loss.py monkey-patch | config: `enable_state_supervision: false`, pipeline: `install_sdr_aware_loss(aux_weight=2.0, margin_D=0.30)` |
| C. v2-S-supervision | mprisk_v2/sdr_loss_v2.py monkey-patch | config: `enable_state_supervision: false`, pipeline: `install_sdr_aware_loss_v2(margin_S=0.05)` |
