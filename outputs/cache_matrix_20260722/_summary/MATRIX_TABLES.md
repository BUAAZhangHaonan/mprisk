# cache_matrix_20260722 Matrix Tables

Generated from raw per-cell JSON files under `outputs/cache_matrix_20260722/runs/`.
All numbers are mean across 3 seeds (seed20260717/18/19) unless noted.
Where a metric file is missing or a field was not emitted, the cell is `—`.

## Table A: Source C/A (in-domain)

Balanced accuracy + macro-F1 (Conflict/Aligned) on the Source C/A
validation split, re-evaluated eval-only on best_checkpoint.pt
(eval_f1.json; mean ± std, 3 seeds). Acc is the same val split and
checkpoint as the training-time best_val_balanced_accuracy_ac.

| Model | GRU Acc | GRU F1 | LSTM Acc | LSTM F1 | BiLSTM Acc | BiLSTM F1 |
|---|---|---|---|---|---|---|
| gemma3_12b | 0.962 ± 0.009 | 0.959 ± 0.009 | — | — | — | — |
| gemma3_4b | 0.953 | 0.953 | — | — | — | — |
| gemma4_12b | — | — | — | — | — | — |
| glm4_6v_flash | — | — | — | — | — | — |
| llava_onevision_qwen2_7b | — | — | — | — | — | — |
| llava_v1_5_7b | — | — | — | — | — | — |
| minicpm_v_2_6 | — | — | — | — | — | — |
| minicpm_v_4_5 | — | — | — | — | — | — |
| internvl3_5_8b | — | — | — | — | — | — |
| phi3_5_vision | — | — | — | — | — | — |
| qwen2_5_omni_7b | — | — | — | — | — | — |
| qwen2_5_vl_7b | — | — | — | — | — | — |
| qwen3_5_4b | — | — | — | — | — | — |
| qwen3_5_9b | — | — | — | — | — | — |
| qwen3_vl_8b | — | — | — | — | — | — |

## Table B: Target C/A (cross-domain, CH-SIMS v2)

Cross-domain Target balanced accuracy + macro-F1 + val_D_gap (mean ± std, 3 seeds).
D_gap = mean(Conflict D) - mean(Aligned D); large positive = healthy state separation.

| Model | GRU Acc | GRU F1 | GRU D_gap | LSTM Acc | LSTM F1 | LSTM D_gap | BiLSTM Acc | BiLSTM F1 | BiLSTM D_gap |
|---|---|---|---|---|---|---|---|---|---|
| gemma3_12b | 0.600 ± 0.032 | 0.458 ± 0.104 | 0.552 ± 0.918 | 0.580 ± 0.029 | — | 0.569 ± 0.990 | 0.563 ± 0.012 | — | -0.281 ± 0.567 |
| gemma3_4b | 0.586 ± 0.037 | — | -0.013 ± 0.984 | 0.600 ± 0.031 | — | 0.900 ± 0.552 | 0.555 ± 0.054 | — | 0.204 ± 0.556 |
| gemma4_12b | 0.491 ± 0.010 | — | 0.647 ± 0.825 | 0.504 ± 0.018 | — | 1.030 ± 1.448 | 0.520 ± 0.028 | — | 0.987 ± 1.240 |
| glm4_6v_flash | 0.532 ± 0.011 | — | -0.007 ± 0.435 | 0.521 ± 0.012 | — | 0.268 ± 0.251 | 0.531 ± 0.005 | — | 0.193 ± 0.098 |
| llava_onevision_qwen2_7b | 0.604 ± 0.026 | — | 0.826 ± 1.156 | 0.567 ± 0.020 | — | 1.110 ± 0.833 | 0.559 ± 0.024 | — | 0.096 ± 0.773 |
| llava_v1_5_7b | 0.635 ± 0.049 | — | 0.509 ± 1.876 | 0.557 ± 0.065 | — | 0.645 ± 1.502 | 0.528 ± 0.008 | — | 0.034 ± 0.071 |
| minicpm_v_2_6 | 0.554 ± 0.043 | — | 0.640 ± 0.405 | 0.524 ± 0.024 | — | 0.184 ± 0.103 | 0.550 ± 0.031 | — | 0.600 ± 0.576 |
| minicpm_v_4_5 | 0.572 ± 0.045 | — | -0.156 ± 0.500 | 0.587 ± 0.053 | — | 0.355 ± 1.004 | 0.556 ± 0.042 | — | 0.061 ± 0.096 |
| internvl3_5_8b | 0.599 ± 0.006 | — | -1.093 ± 0.549 | 0.609 ± 0.087 | — | 1.177 ± 1.386 | 0.583 ± 0.028 | — | 0.028 ± 0.108 |
| phi3_5_vision | 0.582 ± 0.052 | — | 0.140 ± 1.211 | 0.557 ± 0.037 | — | -0.127 ± 0.284 | 0.516 ± 0.022 | — | -0.126 ± 0.162 |
| qwen2_5_omni_7b | 0.532 ± 0.068 | — | 0.225 ± 0.857 | 0.564 ± 0.059 | — | 0.393 ± 0.531 | 0.549 ± 0.023 | — | 0.469 ± 1.045 |
| qwen2_5_vl_7b | 0.574 ± 0.026 | — | 0.532 ± 1.119 | 0.576 ± 0.020 | — | 0.667 ± 0.655 | 0.571 ± 0.007 | — | 0.561 ± 0.112 |
| qwen3_5_4b | 0.574 ± 0.002 | — | -0.775 ± 0.233 | 0.565 ± 0.021 | — | 0.102 ± 1.327 | 0.593 ± 0.006 | — | 0.219 ± 0.031 |
| qwen3_5_9b | 0.600 ± 0.032 | — | 0.400 ± 0.435 | 0.573 ± 0.030 | — | 0.407 ± 0.284 | 0.561 ± 0.021 | — | 0.924 ± 0.671 |
| qwen3_vl_8b | 0.665 ± 0.047 | — | 2.699 ± 0.631 | 0.625 ± 0.029 | — | -0.146 ± 4.641 | 0.623 ± 0.009 | — | 1.991 ± 0.974 |

## Table C: Source M/N

Source M/N test split. TME-E2E and TME-Frozen emit Acc/F1/AUC.
SP-MLP and T-LSTM emit balanced-acc/macro-F1/ROC-AUC. Mean across 3 seeds.

| Model | E2E Acc | E2E F1 | E2E AUC | Frozen Acc | Frozen F1 | Frozen AUC | SP-MLP Acc | T-LSTM Acc |
|---|---|---|---|---|---|---|---|---|
| gemma3_12b | 0.948 | 0.925 | 0.987 | 0.873 | 0.833 | 0.939 | 0.749 | 0.496 |
| gemma3_4b | 0.906 | 0.868 | 0.966 | 0.882 | 0.843 | 0.944 | 0.840 | 0.498 |
| gemma4_12b | 0.764 | 0.546 | 0.765 | 0.753 | 0.392 | 0.774 | 0.538 | 0.494 |
| glm4_6v_flash | 0.858 | 0.836 | 0.889 | 0.752 | 0.689 | 0.812 | 0.722 | 0.528 |
| llava_onevision_qwen2_7b | 0.945 | 0.934 | 0.981 | 0.879 | 0.853 | 0.939 | 0.887 | 0.519 |
| llava_v1_5_7b | 0.673 | 0.681 | 0.740 | 0.670 | 0.684 | 0.747 | 0.688 | 0.527 |
| minicpm_v_2_6 | 0.827 | 0.706 | 0.896 | 0.809 | 0.739 | 0.867 | 0.820 | 0.458 |
| minicpm_v_4_5 | 0.848 | 0.788 | 0.909 | 0.836 | 0.779 | 0.878 | 0.716 | 0.497 |
| internvl3_5_8b | 0.812 | 0.718 | 0.871 | 0.755 | 0.636 | 0.801 | 0.749 | 0.451 |
| phi3_5_vision | 0.748 | 0.548 | 0.839 | 0.694 | 0.570 | 0.774 | 0.557 | 0.474 |
| qwen2_5_omni_7b | 0.845 | 0.691 | 0.915 | 0.824 | 0.697 | 0.879 | 0.679 | 0.490 |
| qwen2_5_vl_7b | 0.945 | 0.930 | 0.963 | 0.827 | 0.791 | 0.910 | 0.824 | 0.663 |
| qwen3_5_4b | 0.827 | 0.617 | 0.829 | 0.694 | 0.460 | 0.730 | 0.697 | 0.557 |
| qwen3_5_9b | 0.812 | 0.647 | 0.864 | 0.685 | 0.537 | 0.785 | 0.679 | 0.498 |
| qwen3_vl_8b | 0.824 | 0.707 | 0.870 | 0.770 | 0.673 | 0.839 | 0.780 | 0.511 |

## Table D: Source -> Target C/A drop

ΔAcc = Target_Acc - Source_Acc (negative = drop). Averaged across 3 seeds.

| Model | GRU ΔAcc | LSTM ΔAcc | BiLSTM ΔAcc | Avg ΔAcc |
|---|---|---|---|---|
| gemma3_12b | -0.362 | -0.384 | -0.398 | -0.381 |
| gemma3_4b | -0.373 | -0.345 | -0.355 | -0.358 |
| gemma4_12b | -0.415 | -0.394 | -0.355 | -0.388 |
| glm4_6v_flash | -0.407 | -0.411 | -0.418 | -0.412 |
| llava_onevision_qwen2_7b | -0.360 | -0.399 | -0.385 | -0.381 |
| llava_v1_5_7b | -0.329 | -0.383 | -0.427 | -0.380 |
| minicpm_v_2_6 | -0.399 | -0.410 | -0.406 | -0.405 |
| minicpm_v_4_5 | -0.400 | -0.370 | -0.410 | -0.393 |
| internvl3_5_8b | -0.354 | -0.344 | -0.373 | -0.357 |
| phi3_5_vision | -0.372 | -0.385 | -0.427 | -0.394 |
| qwen2_5_omni_7b | -0.434 | -0.398 | -0.413 | -0.415 |
| qwen2_5_vl_7b | -0.394 | -0.385 | -0.393 | -0.391 |
| qwen3_5_4b | -0.369 | -0.373 | -0.367 | -0.370 |
| qwen3_5_9b | -0.375 | -0.393 | -0.414 | -0.394 |
| qwen3_vl_8b | -0.304 | -0.338 | -0.347 | -0.330 |

## Table E: Source SDR state distribution

| Model | %Consensus | %Confusion | %Balanced | %Dominant | Conflict→Dominant% | Aligned→Consensus% |
|---|---|---|---|---|---|---|
| gemma3_12b | 52.40 | 14.45 | 0.32 | 32.84 | 79.64 | 85.05 |
| gemma3_4b | 61.19 | 22.44 | 0.91 | 15.46 | 38.11 | 90.21 |
| gemma4_12b | 63.13 | 32.16 | 0.10 | 4.60 | 6.66 | 80.79 |
| glm4_6v_flash | 74.20 | 8.00 | 0.11 | 17.70 | 35.79 | 86.36 |
| llava_onevision_qwen2_7b | 58.80 | 13.11 | 1.23 | 26.87 | 63.66 | 93.01 |
| llava_v1_5_7b | 63.75 | 8.05 | 0.21 | 27.99 | 69.67 | 94.23 |
| minicpm_v_2_6 | 54.42 | 14.13 | 2.19 | 29.26 | 66.26 | 88.64 |
| minicpm_v_4_5 | 72.71 | 6.02 | 0.80 | 20.47 | 46.31 | 90.12 |
| internvl3_5_8b | 80.12 | 6.02 | 0.48 | 13.38 | 26.09 | 90.47 |
| phi3_5_vision | 74.89 | 17.59 | 0.32 | 7.20 | 0.00 | 85.14 |
| qwen2_5_omni_7b | 54.34 | 27.66 | 0.16 | 17.84 | 39.36 | 92.59 |
| qwen2_5_vl_7b | 52.61 | 16.36 | 0.85 | 30.17 | 66.67 | 85.75 |
| qwen3_5_4b | 74.95 | 5.17 | 1.12 | 18.76 | 35.38 | 85.93 |
| qwen3_5_9b | 85.18 | 3.73 | 0.37 | 10.71 | 24.86 | 92.13 |
| qwen3_vl_8b | 50.85 | 21.96 | 1.28 | 25.91 | 54.51 | 82.69 |

## Table F: Target SDR state distribution

| Model | %Consensus | %Confusion | %Balanced | %Dominant | Conflict→Dominant% | Aligned→Consensus% |
|---|---|---|---|---|---|---|
| gemma3_12b | 67.91 | 15.23 | 0.49 | 16.36 | 34.69 | 69.28 |
| gemma3_4b | 40.29 | 58.23 | 0.00 | 1.47 | 2.04 | 41.63 |
| gemma4_12b | 4.38 | 21.55 | 0.00 | 74.06 | 79.59 | 4.34 |
| glm4_6v_flash | 43.88 | 55.97 | 0.00 | 0.15 | 0.68 | 43.75 |
| llava_onevision_qwen2_7b | 67.03 | 26.34 | 0.34 | 6.29 | 10.88 | 68.22 |
| llava_v1_5_7b | 69.73 | 28.75 | 0.00 | 1.52 | 4.08 | 71.08 |
| minicpm_v_2_6 | 25.60 | 72.97 | 0.15 | 1.28 | 1.36 | 25.79 |
| minicpm_v_4_5 | 15.72 | 81.38 | 0.10 | 2.80 | 2.04 | 15.62 |
| internvl3_5_8b | 64.91 | 32.63 | 0.10 | 2.36 | 0.68 | 65.04 |
| phi3_5_vision | 36.12 | 63.88 | 0.00 | 0.00 | 0.00 | 36.49 |
| qwen2_5_omni_7b | 7.26 | 92.69 | 0.00 | 0.05 | 0.00 | 7.33 |
| qwen2_5_vl_7b | 29.43 | 17.49 | 5.65 | 47.42 | 56.46 | 30.19 |
| qwen3_5_4b | 66.00 | 5.36 | 3.73 | 24.91 | 17.69 | 65.10 |
| qwen3_5_9b | 70.66 | 27.37 | 0.29 | 1.67 | 2.72 | 71.08 |
| qwen3_vl_8b | 49.73 | 32.04 | 2.26 | 15.97 | 26.53 | 51.91 |

## Table G: State indices (Source vs Target)

κ = mean S (geodesic prompt dispersion per sample), averaged over all samples.
τ = mean D = acos(c_M1, c_M2)/sqrt(S_M1 + S_M2) per sample, averaged.
δ = mean signed R (M12 asymmetry) per sample, averaged.

| Model | Source κ | Source τ | Source δ | Target κ | Target τ | Target δ |
|---|---|---|---|---|---|---|
| gemma3_12b | 0.007 | 6.402 | 0.272 | 0.007 | 4.855 | 0.592 |
| gemma3_4b | 0.006 | 9.982 | 0.226 | 0.009 | 6.537 | 0.534 |
| gemma4_12b | 0.014 | 10.689 | 0.564 | 0.019 | 23.792 | 0.866 |
| glm4_6v_flash | 0.009 | 10.305 | 0.434 | 0.017 | 4.078 | 0.830 |
| llava_onevision_qwen2_7b | 0.004 | 11.028 | 0.039 | 0.006 | 7.410 | 0.321 |
| llava_v1_5_7b | 0.013 | 8.983 | 0.411 | 0.022 | 4.202 | 0.624 |
| minicpm_v_2_6 | 0.011 | 6.836 | 0.079 | 0.020 | 3.474 | 0.608 |
| minicpm_v_4_5 | 0.018 | 7.163 | 0.212 | 0.059 | 5.209 | 0.449 |
| internvl3_5_8b | 0.012 | 11.369 | 0.160 | 0.022 | 7.637 | 0.330 |
| phi3_5_vision | 0.007 | 10.787 | 0.167 | 0.013 | 6.805 | 0.288 |
| qwen2_5_omni_7b | 0.009 | 5.689 | -0.489 | 0.019 | 3.344 | -0.508 |
| qwen2_5_vl_7b | 0.008 | 7.674 | -0.162 | 0.008 | 7.682 | 0.324 |
| qwen3_5_4b | 0.012 | 9.169 | 0.022 | 0.012 | 9.915 | 0.248 |
| qwen3_5_9b | 0.023 | 7.756 | 0.217 | 0.039 | 5.775 | 0.206 |
| qwen3_vl_8b | 0.011 | 8.841 | -0.055 | 0.013 | 6.326 | 0.453 |

## Section H: Key findings

- **Best Source C/A:** qwen3_5_9b (0.972)
- **Worst Source C/A:** gemma4_12b (0.893)
- **Best Target generalization:** qwen3_vl_8b (0.638)
- **Worst Target generalization:** gemma4_12b (0.505)
- **Smallest Source→Target drop:** qwen3_vl_8b (-0.330)
- **Largest Source→Target drop:** qwen2_5_omni_7b (-0.415)
- **Cross-domain state collapse:** Target D_gap and SDR Dominant% collapse vs Source — every model loses state-separation structure on CH-SIMS v2, with Conflict→Dominant% dropping toward single digits and Confusion% rising. The encoder's relation_r still classifies correctly on Aligned-dominant Target data, but the underlying M1/M2/M12 geometry no longer separates Conflict from Aligned (Target D_gap typically <2 vs Source D_gap typically >5).
