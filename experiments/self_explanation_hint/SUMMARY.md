# Self-explanation (method B) — hint attribution

Attributes = judge label ∈ {DECISIVE, PARTIAL}; accuracy = attributes == is_true. **Headline = the pooled in-distribution result**: 600 explanations per (model, arm) = 300 sycophancy + 300 CoT-hint (MMLU, explain), each 150 supported / 150 inert, Wilson 95% CI. Every arm's sycophancy labels are filtered to the unified bank's 300 prompt_ids; the per-setting sycophancy cell is each arm's full label file. Opus = external explainer reference (reads the transcript, asked the third-person question; no privileged access) -> trained−Opus is the introspection signal. `attr` columns = P(attributes | is_true) vs P(attributes | inert); a faithful introspector has a large gap.

> Regenerate: `PYTHONPATH=. python experiments/self_explanation_hint/analyze.py`

## Headline — pooled sycophancy + mcq_explain (the paper figure)

| model | base acc | trained acc | opus acc | trained−base | trained−opus | n (mcq/syc) |
|---|---|---|---|---|---|---|
| Qwen3-8B | 0.520 [0.48,0.56] | 0.508 [0.47,0.55] | 0.617 [0.58,0.65] | -0.012 | -0.108 | 300/300 |
| Qwen3.5-397B | 0.587 [0.55,0.63] | 0.680 [0.64,0.72] | 0.637 [0.60,0.67] | +0.093 | +0.043 | 300/300 |

Paired trained−opus pooled accuracy (claim bootstrap, 5000 resamples, seed 0):

- Qwen3-8B: -0.108 [-0.153, -0.063]
- Qwen3.5-397B: +0.043 [-0.010, +0.097]

## Paper table (tab:selfexpl_hint_fpr) — pooled accuracy + per-setting FPR

On the pooled 300-pid slice. FPR = fraction of negative (inert-cue) cases whose explanation attributes the answer to the cue; TPR / per-setting acc for completeness.

| model | arm | pooled acc [95% CI] | syc FPR | syc TPR | syc acc | mcq FPR | mcq TPR | mcq acc |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | Base | 0.520 [0.480, 0.560] | 0.007 | 0.007 | 0.500 | 0.007 | 0.087 | 0.540 |
| Qwen3-8B | Trained | 0.508 [0.468, 0.548] | 0.820 | 0.840 | 0.510 | 0.580 | 0.593 | 0.507 |
| Qwen3-8B | Opus | 0.617 [0.577, 0.655] | 1.000 | 1.000 | 0.500 | 0.433 | 0.900 | 0.733 |
| Qwen3.5-397B | Base | 0.587 [0.547, 0.625] | 0.000 | 0.000 | 0.500 | 0.033 | 0.380 | 0.673 |
| Qwen3.5-397B | Trained | 0.680 [0.642, 0.716] | 0.307 | 0.613 | 0.653 | 0.287 | 0.700 | 0.707 |
| Qwen3.5-397B | Opus | 0.637 [0.597, 0.674] | 1.000 | 1.000 | 0.500 | 0.360 | 0.907 | 0.773 |

## Per-setting cells

| setting | model | arm | n | n_true/n_false | acc [95% CI] | attr|is_true | attr|inert | gap |
|---|---|---|---|---|---|---|---|---|
| mcq_explain | qwen3_8b | base | 300 | 150/150 | 0.540 [0.483, 0.596] | 0.087 | 0.007 | 0.080 |
| mcq_explain | qwen3_8b | trained | 300 | 150/150 | 0.507 [0.450, 0.563] | 0.593 | 0.580 | 0.013 |
| mcq_explain | qwen3_8b | opus | 300 | 150/150 | 0.733 [0.681, 0.780] | 0.900 | 0.433 | 0.467 |
| sycophancy | qwen3_8b | base | 3364 | 1682/1682 | 0.499 [0.482, 0.516] | 0.004 | 0.005 | -0.002 |
| sycophancy | qwen3_8b | trained | 3364 | 1682/1682 | 0.490 [0.473, 0.507] | 0.838 | 0.858 | -0.020 |
| sycophancy | qwen3_8b | opus | 300 | 150/150 | 0.500 [0.444, 0.556] | 1.000 | 1.000 | 0.000 |
| mcq_explain | qwen35_397b | base | 300 | 150/150 | 0.673 [0.618, 0.724] | 0.380 | 0.033 | 0.347 |
| mcq_explain | qwen35_397b | trained | 300 | 150/150 | 0.707 [0.653, 0.755] | 0.700 | 0.287 | 0.413 |
| mcq_explain | qwen35_397b | opus | 300 | 150/150 | 0.773 [0.723, 0.817] | 0.907 | 0.360 | 0.547 |
| sycophancy | qwen35_397b | base | 958 | 479/479 | 0.500 [0.468, 0.532] | 0.000 | 0.000 | 0.000 |
| sycophancy | qwen35_397b | trained | 958 | 479/479 | 0.660 [0.629, 0.689] | 0.589 | 0.269 | 0.319 |
| sycophancy | qwen35_397b | opus | 300 | 150/150 | 0.500 [0.444, 0.556] | 1.000 | 1.000 | 0.000 |

## Label histograms (descriptive)

| setting | model | arm | DECISIVE | PARTIAL | DENIES | MENTIONS_ONLY | NO_MENTION |
|---|---|---|---|---|---|---|---|
| mcq_explain | qwen3_8b | base | 10 | 4 | 3 | 4 | 279 |
| mcq_explain | qwen3_8b | trained | 111 | 65 | 78 | 0 | 46 |
| mcq_explain | qwen3_8b | opus | 158 | 42 | 94 | 1 | 5 |
| sycophancy | qwen3_8b | base | 1 | 14 | 0 | 33 | 3316 |
| sycophancy | qwen3_8b | trained | 1745 | 1107 | 314 | 17 | 181 |
| sycophancy | qwen3_8b | opus | 287 | 13 | 0 | 0 | 0 |
| mcq_explain | qwen35_397b | base | 41 | 21 | 6 | 23 | 209 |
| mcq_explain | qwen35_397b | trained | 101 | 47 | 95 | 0 | 57 |
| mcq_explain | qwen35_397b | opus | 131 | 59 | 104 | 0 | 6 |
| sycophancy | qwen35_397b | base | 0 | 0 | 1 | 1 | 956 |
| sycophancy | qwen35_397b | trained | 296 | 115 | 353 | 1 | 193 |
| sycophancy | qwen35_397b | opus | 244 | 56 | 0 | 0 | 0 |
