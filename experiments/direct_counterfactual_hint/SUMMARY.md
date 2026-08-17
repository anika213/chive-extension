# Direct counterfactual (method A) — hint introspection

**Headline = the pooled in-distribution result**: 600 claims per (model, arm) = 300 sycophancy + 300 CoT-hint (MMLU, explain), each 150 supported / 150 inert. AUROC with 95% bootstrap CI (2000 resamples, seed 0); accuracy with Wilson 95% CI. Opus = external transcript reader (no privileged access) -> trained-Opus is the introspection signal.

> Regenerate: `python experiments/direct_counterfactual_hint/analyze.py`

## Headline — pooled sycophancy + mcq_explain (the paper figure)

| model | base AUROC | Opus AUROC | trained AUROC | trained−base | trained−Opus | base/Opus/tr acc |
|---|---|---|---|---|---|---|
| Qwen3-8B | 0.509 [0.46,0.56] | 0.892 [0.86,0.92] | 0.820 [0.79,0.85] | +0.311 | -0.072 | 50%/76%/80% |
| Qwen3.5-397B | 0.878 [0.85,0.90] | 0.940 [0.92,0.96] | 0.956 [0.94,0.97] | +0.079 | +0.017 | 54%/64%/72% |

## Paper table (tab:hint_accuracy) — pooled accuracy + per-setting FPR

FPR = fraction of negative (inert-cue) cases answered Yes; TPR for completeness.

| model | arm | pooled acc [95% CI] | syc FPR | syc TPR | mcq FPR | mcq TPR |
|---|---|---|---|---|---|---|
| Qwen3-8B | Base | 0.498 [0.458, 0.538] | 1.000 | 1.000 | 0.993 | 0.987 |
| Qwen3-8B | Trained | 0.758 [0.723, 0.791] | 0.087 | 0.820 | 0.527 | 0.827 |
| Qwen3-8B | Opus | 0.802 [0.768, 0.832] | 0.120 | 0.673 | 0.100 | 0.753 |
| Qwen3.5-397B | Base | 0.543 [0.503, 0.583] | 0.000 | 0.000 | 0.013 | 0.187 |
| Qwen3.5-397B | Trained | 0.638 [0.599, 0.676] | 0.000 | 0.113 | 0.007 | 0.447 |
| Qwen3.5-397B | Opus | 0.722 [0.684, 0.756] | 0.027 | 0.320 | 0.027 | 0.620 |

## Per-setting cells

| setting | model | arm | n | n_true/n_false | AUROC [95% CI] | acc | Brier |
|---|---|---|---|---|---|---|---|
| mcq_explain | qwen3_8b | base | 300 | 150/150 | 0.543 [0.480, 0.610] | 49.7% | 0.500 |
| mcq_explain | qwen3_8b | trained | 300 | 150/150 | 0.749 [0.690, 0.804] | 65.0% | 0.212 |
| mcq_explain | qwen3_8b | opus | 300 | 150/150 | 0.910 [0.874, 0.940] | 82.7% | 0.138 |
| sycophancy | qwen3_8b | base | 300 | 150/150 | 0.471 [0.405, 0.540] | 50.0% | 0.498 |
| sycophancy | qwen3_8b | trained | 300 | 150/150 | 0.911 [0.873, 0.945] | 86.7% | 0.172 |
| sycophancy | qwen3_8b | opus | 300 | 150/150 | 0.884 [0.842, 0.920] | 77.7% | 0.168 |
| mcq_explain | qwen35_397b | base | 300 | 150/150 | 0.896 [0.857, 0.929] | 58.7% | 0.360 |
| mcq_explain | qwen35_397b | trained | 300 | 150/150 | 0.974 [0.954, 0.989] | 72.0% | 0.198 |
| mcq_explain | qwen35_397b | opus | 300 | 150/150 | 0.962 [0.940, 0.981] | 79.7% | 0.143 |
| sycophancy | qwen35_397b | base | 300 | 150/150 | 0.891 [0.852, 0.926] | 50.0% | 0.483 |
| sycophancy | qwen35_397b | trained | 300 | 150/150 | 0.942 [0.914, 0.965] | 55.7% | 0.367 |
| sycophancy | qwen35_397b | opus | 300 | 150/150 | 0.909 [0.874, 0.940] | 64.7% | 0.218 |

## Per-cue AUROC (descriptive)

| setting | model | arm | cue | n | n_true/n_false | AUROC |
|---|---|---|---|---|---|---|
| mcq_explain | qwen3_8b | base | answer_comment | 44 | 15/29 | 0.644 |
| mcq_explain | qwen3_8b | base | answer_mark | 52 | 26/26 | 0.635 |
| mcq_explain | qwen3_8b | base | previous_answer | 53 | 16/37 | 0.602 |
| mcq_explain | qwen3_8b | base | professor | 69 | 41/28 | 0.511 |
| mcq_explain | qwen3_8b | base | sycophancy_school | 82 | 52/30 | 0.563 |
| mcq_explain | qwen3_8b | trained | answer_comment | 44 | 15/29 | 0.633 |
| mcq_explain | qwen3_8b | trained | answer_mark | 52 | 26/26 | 0.849 |
| mcq_explain | qwen3_8b | trained | previous_answer | 53 | 16/37 | 0.784 |
| mcq_explain | qwen3_8b | trained | professor | 69 | 41/28 | 0.654 |
| mcq_explain | qwen3_8b | trained | sycophancy_school | 82 | 52/30 | 0.821 |
| mcq_explain | qwen3_8b | opus | answer_comment | 44 | 15/29 | 0.922 |
| mcq_explain | qwen3_8b | opus | answer_mark | 52 | 26/26 | 0.915 |
| mcq_explain | qwen3_8b | opus | previous_answer | 53 | 16/37 | 0.884 |
| mcq_explain | qwen3_8b | opus | professor | 69 | 41/28 | 0.919 |
| mcq_explain | qwen3_8b | opus | sycophancy_school | 82 | 52/30 | 0.906 |
| sycophancy | qwen3_8b | base | hint_a | 137 | 120/17 | 0.559 |
| sycophancy | qwen3_8b | base | hint_b | 163 | 30/133 | 0.670 |
| sycophancy | qwen3_8b | trained | hint_a | 137 | 120/17 | 0.772 |
| sycophancy | qwen3_8b | trained | hint_b | 163 | 30/133 | 0.727 |
| sycophancy | qwen3_8b | opus | hint_a | 137 | 120/17 | 0.646 |
| sycophancy | qwen3_8b | opus | hint_b | 163 | 30/133 | 0.829 |
| mcq_explain | qwen35_397b | base | answer_comment | 52 | 20/32 | 0.884 |
| mcq_explain | qwen35_397b | base | answer_mark | 63 | 30/33 | 0.888 |
| mcq_explain | qwen35_397b | base | previous_answer | 39 | 19/20 | 0.911 |
| mcq_explain | qwen35_397b | base | professor | 97 | 61/36 | 0.908 |
| mcq_explain | qwen35_397b | base | sycophancy_school | 49 | 20/29 | 0.933 |
| mcq_explain | qwen35_397b | trained | answer_comment | 52 | 20/32 | 0.978 |
| mcq_explain | qwen35_397b | trained | answer_mark | 63 | 30/33 | 0.991 |
| mcq_explain | qwen35_397b | trained | previous_answer | 39 | 19/20 | 0.989 |
| mcq_explain | qwen35_397b | trained | professor | 97 | 61/36 | 0.984 |
| mcq_explain | qwen35_397b | trained | sycophancy_school | 49 | 20/29 | 0.945 |
| mcq_explain | qwen35_397b | opus | answer_comment | 52 | 20/32 | 0.948 |
| mcq_explain | qwen35_397b | opus | answer_mark | 63 | 30/33 | 0.976 |
| mcq_explain | qwen35_397b | opus | previous_answer | 39 | 19/20 | 0.979 |
| mcq_explain | qwen35_397b | opus | professor | 97 | 61/36 | 0.983 |
| mcq_explain | qwen35_397b | opus | sycophancy_school | 49 | 20/29 | 0.910 |
| sycophancy | qwen35_397b | base | hint_a | 142 | 107/35 | 0.879 |
| sycophancy | qwen35_397b | base | hint_b | 158 | 43/115 | 0.789 |
| sycophancy | qwen35_397b | trained | hint_a | 142 | 107/35 | 0.918 |
| sycophancy | qwen35_397b | trained | hint_b | 158 | 43/115 | 0.921 |
| sycophancy | qwen35_397b | opus | hint_a | 142 | 107/35 | 0.854 |
| sycophancy | qwen35_397b | opus | hint_b | 158 | 43/115 | 0.874 |
