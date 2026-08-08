# cache_matrix_20260722 Final Report

## Section 1: Source C/A (in-domain, 15 models x 3 encoders x 3 seeds)

Balanced accuracy on the Source conflict-attn validation split (mean +/- std across 3 seeds).
Best encoder per model in **bold**.

| Model | GRU | LSTM | BiLSTM |
|---|---|---|---|
| gemma3_12b | 0.9622 ± 0.0090 | **0.9635 ± 0.0105** | 0.9605 ± 0.0028 |
| gemma3_4b | **0.9586 ± 0.0054** | 0.9449 ± 0.0133 | 0.9101 ± 0.0630 |
| gemma4_12b | **0.9068 ± 0.0056** | 0.8975 ± 0.0092 | 0.8742 ± 0.0467 |
| glm4_6v_flash | 0.9387 ± 0.0096 | 0.9319 ± 0.0406 | **0.9489 ± 0.0166** |
| llava_onevision_qwen2_7b | 0.9632 ± 0.0069 | **0.9661 ± 0.0040** | 0.9444 ± 0.0167 |
| llava_v1_5_7b | **0.9645 ± 0.0189** | 0.9401 ± 0.0274 | 0.9541 ± 0.0068 |
| minicpm_v_2_6 | 0.9538 ± 0.0170 | 0.9336 ± 0.0152 | **0.9554 ± 0.0177** |
| minicpm_v_4_5 | **0.9717 ± 0.0097** | 0.9567 ± 0.0107 | 0.9655 ± 0.0074 |
| internvl3_5_8b | 0.9528 ± 0.0175 | 0.9528 ± 0.0044 | **0.9557 ± 0.0072** |
| phi3_5_vision | **0.9534 ± 0.0028** | 0.9417 ± 0.0040 | 0.9427 ± 0.0065 |
| qwen2_5_omni_7b | **0.9654 ± 0.0180** | 0.9622 ± 0.0219 | 0.9611 ± 0.0077 |
| qwen2_5_vl_7b | **0.9680 ± 0.0209** | 0.9606 ± 0.0236 | 0.9638 ± 0.0197 |
| qwen3_5_4b | 0.9430 ± 0.0107 | 0.9374 ± 0.0264 | **0.9596 ± 0.0081** |
| qwen3_5_9b | 0.9746 ± 0.0051 | 0.9661 ± 0.0177 | **0.9752 ± 0.0220** |
| qwen3_vl_8b | 0.9694 ± 0.0059 | 0.9632 ± 0.0147 | **0.9700 ± 0.0069** |

## Section 2: Target C/A (cross-domain, CH-SIMS v2)

Cross-domain transfer: balanced accuracy on the CH-SIMS v2 Target split.
Source->Target drop is averaged across the 3 encoders (positive = degradation).

| Model | GRU | LSTM | BiLSTM | avg Source | avg Target | avg Drop |
|---|---|---|---|---|---|---|
| gemma3_12b | 0.5998 ± 0.0325 | 0.5798 ± 0.0294 | 0.5629 ± 0.0122 | 0.9621 | 0.5808 | +0.3812 |
| gemma3_4b | 0.5857 ± 0.0368 | 0.5997 ± 0.0309 | 0.5551 ± 0.0537 | 0.9379 | 0.5802 | +0.3577 |
| gemma4_12b | 0.4913 ± 0.0098 | 0.5038 ± 0.0185 | 0.5196 ± 0.0281 | 0.8928 | 0.5049 | +0.3879 |
| glm4_6v_flash | 0.5318 ± 0.0108 | 0.5213 ± 0.0116 | 0.5305 ± 0.0049 | 0.9398 | 0.5279 | +0.4120 |
| llava_onevision_qwen2_7b | 0.6037 ± 0.0264 | 0.5667 ± 0.0200 | 0.5593 ± 0.0244 | 0.9579 | 0.5766 | +0.3813 |
| llava_v1_5_7b | 0.6354 ± 0.0493 | 0.5571 ± 0.0648 | 0.5275 ± 0.0075 | 0.9529 | 0.5733 | +0.3796 |
| minicpm_v_2_6 | 0.5544 ± 0.0428 | 0.5237 ± 0.0237 | 0.5498 ± 0.0308 | 0.9476 | 0.5426 | +0.4050 |
| minicpm_v_4_5 | 0.5716 ± 0.0451 | 0.5870 ± 0.0531 | 0.5556 ± 0.0421 | 0.9646 | 0.5714 | +0.3932 |
| internvl3_5_8b | 0.5992 ± 0.0062 | 0.6086 ± 0.0874 | 0.5825 ± 0.0284 | 0.9538 | 0.5968 | +0.3570 |
| phi3_5_vision | 0.5816 ± 0.0522 | 0.5569 ± 0.0371 | 0.5162 ± 0.0223 | 0.9459 | 0.5516 | +0.3944 |
| qwen2_5_omni_7b | 0.5317 ± 0.0681 | 0.5645 ± 0.0589 | 0.5485 ± 0.0231 | 0.9629 | 0.5482 | +0.4147 |
| qwen2_5_vl_7b | 0.5737 ± 0.0256 | 0.5759 ± 0.0197 | 0.5710 ± 0.0073 | 0.9641 | 0.5735 | +0.3906 |
| qwen3_5_4b | 0.5737 ± 0.0016 | 0.5647 ± 0.0214 | 0.5927 ± 0.0055 | 0.9467 | 0.5770 | +0.3696 |
| qwen3_5_9b | 0.5996 ± 0.0319 | 0.5734 ± 0.0299 | 0.5609 ± 0.0207 | 0.9720 | 0.5780 | +0.3940 |
| qwen3_vl_8b | 0.6654 ± 0.0466 | 0.6250 ± 0.0287 | 0.6233 ± 0.0087 | 0.9675 | 0.6379 | +0.3296 |

Biggest avg drop: **qwen2_5_omni_7b** (+0.4147).  
Smallest avg drop: **qwen3_vl_8b** (+0.3296).

_val_D_gap and val_D_mannwhitney_p: n/a — Target eval pipeline only writes val_balanced_accuracy_ac and val_state_separation (always null in current outputs)._

## Section 3: Source M/N (15 models x 4 methods x 3 seeds)

Test accuracy / AUC on the Source M/N split (mean +/- std across 3 seeds).
SP-MLP and T-LSTM do not produce AUC (only accuracy + AP).

| Model | E2E acc | E2E AUC | Frozen acc | Frozen AUC | SP-MLP acc | T-LSTM acc |
|---|---|---|---|---|---|---|
| gemma3_12b | 0.9485 ± 0.0052 | 0.9868 ± 0.0017 | 0.8727 ± 0.0157 | 0.9386 ± 0.0241 | 0.7491 ± 0.0248 | 0.4963 ± 0.0026 |
| gemma3_4b | 0.9061 ± 0.0052 | 0.9659 ± 0.0104 | 0.8818 ± 0.0182 | 0.9438 ± 0.0117 | 0.8401 ± 0.0447 | 0.4984 ± 0.0419 |
| gemma4_12b | 0.7638 ± 0.0273 | 0.7649 ± 0.0120 | 0.7533 ± 0.0120 | 0.7744 ± 0.0144 | 0.5380 ± 0.0352 | 0.4945 ± 0.0025 |
| glm4_6v_flash | 0.8576 ± 0.0139 | 0.8895 ± 0.0275 | 0.7515 ± 0.0189 | 0.8122 ± 0.0606 | 0.7224 ± 0.0234 | 0.5277 ± 0.0561 |
| llava_onevision_qwen2_7b | 0.9455 ± 0.0157 | 0.9806 ± 0.0068 | 0.8788 ± 0.0706 | 0.9386 ± 0.0416 | 0.8871 ± 0.0327 | 0.5191 ± 0.0665 |
| llava_v1_5_7b | 0.6727 ± 0.0241 | 0.7401 ± 0.0148 | 0.6697 ± 0.0458 | 0.7473 ± 0.0643 | 0.6885 ± 0.0234 | 0.5266 ± 0.0392 |
| minicpm_v_2_6 | 0.8273 ± 0.0273 | 0.8960 ± 0.0212 | 0.8091 ± 0.0157 | 0.8669 ± 0.0167 | 0.8198 ± 0.0096 | 0.4582 ± 0.0029 |
| minicpm_v_4_5 | 0.8485 ± 0.0139 | 0.9086 ± 0.0112 | 0.8364 ± 0.0273 | 0.8777 ± 0.0156 | 0.7159 ± 0.1868 | 0.4975 ± 0.0600 |
| internvl3_5_8b | 0.8121 ± 0.0105 | 0.8714 ± 0.0039 | 0.7545 ± 0.0417 | 0.8011 ± 0.0784 | 0.7489 ± 0.0113 | 0.4513 ± 0.0113 |
| phi3_5_vision | 0.7485 ± 0.0210 | 0.8391 ± 0.0282 | 0.6939 ± 0.0229 | 0.7743 ± 0.0392 | 0.5569 ± 0.0686 | 0.4735 ± 0.0090 |
| qwen2_5_omni_7b | 0.8451 ± 0.0164 | 0.9151 ± 0.0154 | 0.8241 ± 0.0434 | 0.8795 ± 0.0514 | 0.6793 ± 0.0501 | 0.4905 ± 0.0133 |
| qwen2_5_vl_7b | 0.9455 ± 0.0000 | 0.9634 ± 0.0070 | 0.8273 ± 0.0656 | 0.9102 ± 0.0499 | 0.8243 ± 0.0152 | 0.6635 ± 0.0175 |
| qwen3_5_4b | 0.8273 ± 0.0091 | 0.8289 ± 0.0092 | 0.6939 ± 0.0139 | 0.7297 ± 0.0158 | 0.6966 ± 0.0330 | 0.5566 ± 0.0521 |
| qwen3_5_9b | 0.8121 ± 0.0501 | 0.8643 ± 0.0113 | 0.6848 ± 0.0378 | 0.7853 ± 0.0342 | 0.6788 ± 0.0900 | 0.4984 ± 0.0382 |
| qwen3_vl_8b | 0.8242 ± 0.0139 | 0.8702 ± 0.0173 | 0.7697 ± 0.0229 | 0.8388 ± 0.0080 | 0.7801 ± 0.0177 | 0.5105 ± 0.0445 |

## Section 4: Source SDR state distribution

Pattern distribution over the relevant SDR sample set. Conflict->Dominant% and Aligned->Consensus% are conditioned on sample_type.

| Model | n | %Consensus | %Confusion | %Balanced | %Dominant | Conflict->Dominant% | Aligned->Consensus% |
|---|---|---|---|---|---|---|---|
| gemma3_12b | 1876 | 52.4 | 14.45 | 0.32 | 32.84 | 79.64 | 85.05 |
| gemma3_4b | 1876 | 61.19 | 22.44 | 0.91 | 15.46 | 38.11 | 90.21 |
| gemma4_12b | 1934 | 63.13 | 32.16 | 0.1 | 4.6 | 6.66 | 80.79 |
| glm4_6v_flash | 1876 | 74.2 | 8.0 | 0.11 | 17.7 | 35.79 | 86.36 |
| llava_onevision_qwen2_7b | 1876 | 58.8 | 13.11 | 1.23 | 26.87 | 63.66 | 93.01 |
| llava_v1_5_7b | 1876 | 63.75 | 8.05 | 0.21 | 27.99 | 69.67 | 94.23 |
| minicpm_v_2_6 | 1876 | 54.42 | 14.13 | 2.19 | 29.26 | 66.26 | 88.64 |
| minicpm_v_4_5 | 1876 | 72.71 | 6.02 | 0.8 | 20.47 | 46.31 | 90.12 |
| internvl3_5_8b | 1876 | 80.12 | 6.02 | 0.48 | 13.38 | 26.09 | 90.47 |
| phi3_5_vision | 1876 | 74.89 | 17.59 | 0.32 | 7.2 | 0.0 | 85.14 |
| qwen2_5_omni_7b | 1934 | 54.34 | 27.66 | 0.16 | 17.84 | 39.36 | 92.59 |
| qwen2_5_vl_7b | 1876 | 52.61 | 16.36 | 0.85 | 30.17 | 66.67 | 85.75 |
| qwen3_5_4b | 1876 | 74.95 | 5.17 | 1.12 | 18.76 | 35.38 | 85.93 |
| qwen3_5_9b | 1876 | 85.18 | 3.73 | 0.37 | 10.71 | 24.86 | 92.13 |
| qwen3_vl_8b | 1876 | 50.85 | 21.96 | 1.28 | 25.91 | 54.51 | 82.69 |

## Section 5: Target SDR state distribution

Pattern distribution over the relevant SDR sample set. Conflict->Dominant% and Aligned->Consensus% are conditioned on sample_type.

| Model | n | %Consensus | %Confusion | %Balanced | %Dominant | Conflict->Dominant% | Aligned->Consensus% |
|---|---|---|---|---|---|---|---|
| gemma3_12b | 2035 | 67.91 | 15.23 | 0.49 | 16.36 | 34.69 | 69.28 |
| gemma3_4b | 2035 | 40.29 | 58.23 | 0.0 | 1.47 | 2.04 | 41.63 |
| gemma4_12b | 2190 | 4.38 | 21.55 | 0.0 | 74.06 | 79.59 | 4.34 |
| glm4_6v_flash | 2035 | 43.88 | 55.97 | 0.0 | 0.15 | 0.68 | 43.75 |
| llava_onevision_qwen2_7b | 2035 | 67.03 | 26.34 | 0.34 | 6.29 | 10.88 | 68.22 |
| llava_v1_5_7b | 2035 | 69.73 | 28.75 | 0.0 | 1.52 | 4.08 | 71.08 |
| minicpm_v_2_6 | 2035 | 25.6 | 72.97 | 0.15 | 1.28 | 1.36 | 25.79 |
| minicpm_v_4_5 | 2035 | 15.72 | 81.38 | 0.1 | 2.8 | 2.04 | 15.62 |
| internvl3_5_8b | 2035 | 64.91 | 32.63 | 0.1 | 2.36 | 0.68 | 65.04 |
| phi3_5_vision | 2035 | 36.12 | 63.88 | 0.0 | 0.0 | 0.0 | 36.49 |
| qwen2_5_omni_7b | 2190 | 7.26 | 92.69 | 0.0 | 0.05 | 0.0 | 7.33 |
| qwen2_5_vl_7b | 2035 | 29.43 | 17.49 | 5.65 | 47.42 | 56.46 | 30.19 |
| qwen3_5_4b | 2035 | 66.0 | 5.36 | 3.73 | 24.91 | 17.69 | 65.1 |
| qwen3_5_9b | 2035 | 70.66 | 27.37 | 0.29 | 1.67 | 2.72 | 71.08 |
| qwen3_vl_8b | 2035 | 49.73 | 32.04 | 2.26 | 15.97 | 26.53 | 51.91 |

## Section 6: Key findings

- **Best Source C/A**: `qwen3_5_9b` + `BiLSTM` at 0.9752 balanced accuracy.
- **Best Target C/A (cross-domain)**: `qwen3_vl_8b` + `GRU` at 0.6654.
- **Smallest Source->Target drop (best transfer)**: `qwen3_vl_8b` (avg drop +0.3296); biggest drop: `qwen2_5_omni_7b` (+0.4147).
- **Best Source M/N**: `gemma3_12b` + `mn_tme_e2e` at 0.9485 test accuracy.
- **State shift Source->Target**: avg %Dominant 19.94 -> 13.09, avg %Consensus 64.90 -> 43.91, avg %Confusion 14.46 -> 42.13. Cross-domain reduces Dominant and inflates Confusion, consistent with misread-ground-truth mismatch on CH-SIMS v2.

_Notes: phi4_multimodal dropped (max_new_tokens=64 bug, 0 judgments). Target M/N: not possible (CH-SIMS v2 has no misread GT). Target D_gap / Mann-Whitney p: not written by the current eval pipeline (val_state_separation is the only auxiliary field and is null in all 135 cells)._
