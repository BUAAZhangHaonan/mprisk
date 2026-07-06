# Dataset Audit

- Dataset root: `/home/team/zhanghaonan/TAFFC/datasets`
- Generated at: `2026-07-06T16:37:37.721262+00:00`

| Dataset | Exists | Files | Bytes | Modalities | VT native | VA native | IT derived | Labels |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |
| ch_sims | yes | 4049 | 10287912286 | video, text | no | no | no | 2 |
| ch_sims_v2 | yes | 16410 | 33826670191 | video, audio, text | yes | yes | no | 4 |
| cmu_mosi | yes | 2202 | 1459024957 | video, audio, text | no | no | yes | 3 |
| cmu_mosei | yes | 22859 | 45873726486 | video, audio, text, image | no | no | yes | 3 |

## ch_sims

- Root path: `/home/team/zhanghaonan/TAFFC/datasets/ch_sims`
- Label files: Processed/unaligned_39.pkl, label.csv
- Notes: CH-SIMS protocol support is pending audit until local label hints are confirmed.

Label columns:

- `Processed/unaligned_39.pkl`: -
- `label.csv`: video_0001, 0001, 我不想嫁给李茶, -1.0, -1.0, -1.0, -1.0, Negative, train

## ch_sims_v2

- Root path: `/home/team/zhanghaonan/TAFFC/datasets/ch_sims_v2`
- Label files: CH-SIMS v2(s)/Processed/unaligned.pkl, CH-SIMS v2(s)/meta.csv, CH-SIMS v2(u)/Processed/unaligned.pkl, CH-SIMS v2(u)/meta.csv
- Notes: -

Label columns:

- `CH-SIMS v2(s)/Processed/unaligned.pkl`: -
- `CH-SIMS v2(s)/meta.csv`: video_id, clip_id, text, label, label_T, label_A, label_V, annotation, mode
- `CH-SIMS v2(u)/Processed/unaligned.pkl`: -
- `CH-SIMS v2(u)/meta.csv`: video_id, clip_id, text, label, label_T, label_A, label_V, annotation

## cmu_mosi

- Root path: `/home/team/zhanghaonan/TAFFC/datasets/cmu_mosi`
- Label files: Processed/aligned_50.pkl, Processed/unaligned_50.pkl, label.csv
- Notes: -

Label columns:

- `Processed/aligned_50.pkl`: -
- `Processed/unaligned_50.pkl`: -
- `label.csv`: video_id, clip_id, text, label, label_T, label_A, label_V, annotation, mode

## cmu_mosei

- Root path: `/home/team/zhanghaonan/TAFFC/datasets/cmu_mosei`
- Label files: Processed/aligned_50.pkl, Processed/unaligned_50.pkl, label.csv
- Notes: -

Label columns:

- `Processed/aligned_50.pkl`: -
- `Processed/unaligned_50.pkl`: -
- `label.csv`: video_id, clip_id, text, label, annotation, mode, label_T, label_A, label_V
