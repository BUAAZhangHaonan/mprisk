# cache_matrix_20260722 audit

## Status: FAIL

Encoders audited (directly from runs/):
  - ca_tme_gru (`runs/ca_tme_gru/*/train_metrics.json`): 45/45
  - ca_tme_lstm (`runs/ca_tme_lstm/*/train_metrics.json`): 45/45
  - ca_tme_bilstm (`runs/ca_tme_bilstm/*/train_metrics.json`): 45/45
  - mn_tme_e2e (`runs/mn_tme_e2e/*/metrics.json`): 45/45
  - mn_tme_frozen (`runs/mn_tme_frozen/*/metrics.json`): 45/45
  - sp_mlp (`runs/sp_mlp/*/mn_metrics.json`): 48/48
  - t_lstm (`runs/t_lstm/*/mn_metrics.json`): 48/48

## OK

- count[ca_tme_gru] = 45/45
- all ca_tme_gru metrics.json loaded (45 cells)
- ca_tme_gru primary metrics finite (45/45)
- ca_tme_gru canonical 13-model set present
- ca_tme_gru cross-seed std < 0.05 (15 cells checked)
- ca_tme_gru InternVL within 2sigma (delta/sigma=0.20)
- count[ca_tme_lstm] = 45/45
- all ca_tme_lstm metrics.json loaded (45 cells)
- ca_tme_lstm primary metrics finite (45/45)
- ca_tme_lstm canonical 13-model set present
- ca_tme_lstm cross-seed std < 0.05 (15 cells checked)
- ca_tme_lstm InternVL within 2sigma (delta/sigma=0.21)
- count[ca_tme_bilstm] = 45/45
- all ca_tme_bilstm metrics.json loaded (45 cells)
- ca_tme_bilstm primary metrics finite (45/45)
- ca_tme_bilstm canonical 13-model set present
- ca_tme_bilstm InternVL within 2sigma (delta/sigma=0.20)
- count[mn_tme_e2e] = 45/45
- all mn_tme_e2e metrics.json loaded (45 cells)
- mn_tme_e2e primary metrics finite (45/45)
- mn_tme_e2e canonical 13-model set present
- mn_tme_e2e InternVL within 2sigma (delta/sigma=0.36)
- count[mn_tme_frozen] = 45/45
- all mn_tme_frozen metrics.json loaded (45 cells)
- mn_tme_frozen primary metrics finite (45/45)
- mn_tme_frozen canonical 13-model set present
- mn_tme_frozen InternVL within 2sigma (delta/sigma=0.34)
- count[sp_mlp] = 48/48
- sp_mlp primary metrics finite (45/48)
- sp_mlp has 16 models (extra: ['phi4_multimodal'])
- sp_mlp InternVL within 2sigma (delta/sigma=0.20)
- count[t_lstm] = 48/48
- t_lstm primary metrics finite (45/48)
- t_lstm has 16 models (extra: ['phi4_multimodal'])
- t_lstm InternVL within 2sigma (delta/sigma=1.11)
- SDR[gemma3_12b]: 1876 rows, 4 modes
- SDR[gemma3_4b]: 1876 rows, 4 modes
- SDR[gemma4_12b]: 1934 rows, 4 modes
- SDR[glm4_6v_flash]: 1876 rows, 4 modes
- SDR[llava_onevision_qwen2_7b]: 1876 rows, 4 modes
- SDR[llava_v1_5_7b]: 1876 rows, 4 modes
- SDR[minicpm_v_2_6]: 1876 rows, 4 modes
- SDR[minicpm_v_4_5]: 1876 rows, 4 modes
- SDR[internvl3_5_8b]: 1876 rows, 4 modes
- SDR[phi3_5_vision]: 1876 rows, 4 modes
- SDR[qwen2_5_omni_7b]: 1934 rows, 4 modes
- SDR[qwen2_5_vl_7b]: 1876 rows, 4 modes
- SDR[qwen3_5_4b]: 1876 rows, 4 modes
- SDR[qwen3_5_9b]: 1876 rows, 4 modes
- SDR[qwen3_vl_8b]: 1876 rows, 4 modes
- SDR summary: complete=15, missing_thresholds=0, missing_encoder=0, empty=0
- target C/A cells = 135/135
- target C/A val_balanced_accuracy_ac finite (135/135)
- target SDR[gemma3_12b]: 2035 rows, 4 modes
- target SDR[llava_onevision_qwen2_7b]: 2035 rows, 4 modes
- target SDR[minicpm_v_2_6]: 2035 rows, 4 modes
- target SDR[minicpm_v_4_5]: 2035 rows, 4 modes
- target SDR[internvl3_5_8b]: 2035 rows, 4 modes
- target SDR[qwen2_5_vl_7b]: 2035 rows, 4 modes
- target SDR[qwen3_5_4b]: 2035 rows, 4 modes
- target SDR[qwen3_5_9b]: 2035 rows, 4 modes
- target SDR[qwen3_vl_8b]: 2035 rows, 4 modes
- target SDR complete (9/15)

## Problems

- cross-seed std >= 0.05 in ca_tme_bilstm (1 cells): gemma3_4b: std=0.0630
- cross-seed std >= 0.05 in mn_tme_e2e (1 cells): qwen3_5_9b: std=0.0501
- cross-seed std >= 0.05 in mn_tme_frozen (2 cells): llava_onevision_qwen2_7b: std=0.0706; qwen2_5_vl_7b: std=0.0656
- missing/unparseable metrics in sp_mlp (3): phi4_multimodal/seed20260717, phi4_multimodal/seed20260718, phi4_multimodal/seed20260719
- cross-seed std >= 0.05 in sp_mlp (4 cells): minicpm_v_4_5: std=0.1868; phi3_5_vision: std=0.0686; qwen2_5_omni_7b: std=0.0501; qwen3_5_9b: std=0.0900
- missing/unparseable metrics in t_lstm (3): phi4_multimodal/seed20260717, phi4_multimodal/seed20260718, phi4_multimodal/seed20260719
- cross-seed std >= 0.05 in t_lstm (4 cells): glm4_6v_flash: std=0.0561; llava_onevision_qwen2_7b: std=0.0665; minicpm_v_4_5: std=0.0600; qwen3_5_4b: std=0.0521
- target SDR[gemma3_4b]: only 3 modes covered: ['Confusion', 'Consensus', 'Dominant']
- target SDR[gemma4_12b]: only 3 modes covered: ['Confusion', 'Consensus', 'Dominant']
- target SDR[glm4_6v_flash]: only 3 modes covered: ['Confusion', 'Consensus', 'Dominant']
- target SDR[llava_v1_5_7b]: only 3 modes covered: ['Confusion', 'Consensus', 'Dominant']
- target SDR[phi3_5_vision]: only 2 modes covered: ['Confusion', 'Consensus']
- target SDR[qwen2_5_omni_7b]: only 3 modes covered: ['Confusion', 'Consensus', 'Dominant']
