# Truly unfiltered evaluation (concreteness 1-2 included) — results

**Takeaways (2026-07-17 run).** (1) The interpretability-tool null is Filter-1-invariant: on the full unfiltered Gemma bank (n=4,814) AO/NLA/SAE sit at ΔAUROC +0.001/−0.003/+0.004 vs the transcript-only baseline, every CI straddling zero — same on the conc1-2-only slice. (2) The trained direct-counterfactual models EXCEED the Opus reader on the unfiltered set (8B +0.019 [+0.007, +0.032]; 397B +0.035 [+0.024, +0.045]), driven by the vague-mechanism conc1-2 claims that Filter 1 removes (8B trained 0.848 vs plain 0.818 there) — on the filtered set the gap was ~0. (3) Absolute difficulty is slightly higher unfiltered (plain drops ~0.01–0.04) but conclusions are unchanged.
Populations recomputed from source each run of this script (see module docstring). AUROC via rank-sum; 95% bootstrap CIs (2000 resamples, seed 0); ΔAUROC paired vs the transcript-only Opus baseline on the identical claim set. `acc` at the 0.5 threshold with the TRUE base rate shown (the no-filters population is the full bank, not re-balanced).

## gemma-3-27b-it

### no filters  —  n=4433  (2300T/2133F)

| arm | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|
| plain | 0.777 [0.764, 0.791] | 71.0% | 0.193 | — |
| ao | 0.777 [0.763, 0.790] | 70.7% | 0.194 | +0.000 [-0.007, +0.007] |
| nla | 0.772 [0.759, 0.785] | 69.3% | 0.199 | -0.005 [-0.012, +0.002] |
| sae | 0.780 [0.766, 0.793] | 70.2% | 0.194 | +0.003 [-0.004, +0.009] |

### all filters  —  n=1294  (682T/612F)

| arm | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|
| plain | 0.814 [0.792, 0.837] | 73.6% | 0.179 | — |
| ao | 0.809 [0.787, 0.832] | 72.7% | 0.180 | -0.005 [-0.017, +0.007] |
| nla | 0.803 [0.779, 0.826] | 71.3% | 0.186 | -0.012 [-0.025, +0.001] |
| sae | 0.810 [0.785, 0.833] | 71.9% | 0.180 | -0.005 [-0.016, +0.006] |

### Concreteness breakdown (within no filters)

| arm | conc 1-2 AUROC (n) | conc 3-4 AUROC (n) | Δ vs plain (1-2) |
|---|---|---|---|
| plain | 0.775 (2663) | 0.777 (1770) |  |
| ao | 0.773 (2663) | 0.780 (1770) | -0.002 |
| nla | 0.770 (2663) | 0.771 (1770) | -0.005 |
| sae | 0.778 (2663) | 0.779 (1770) | +0.003 |

## qwen3-8b

### no filters  —  n=4041  (1928T/2113F)

| arm | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|
| plain | 0.807 [0.794, 0.821] | 73.8% | 0.182 | — |
| base | 0.536 [0.518, 0.554] | 49.4% | 0.463 | -0.271 [-0.292, -0.251] |
| trained | 0.822 [0.809, 0.834] | 74.5% | 0.172 | +0.015 [+0.002, +0.027] |

### all filters  —  n=1497  (778T/719F)

| arm | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|
| plain | 0.815 [0.794, 0.836] | 74.7% | 0.177 | — |
| base | 0.543 [0.513, 0.572] | 52.9% | 0.434 | -0.272 [-0.309, -0.238] |
| trained | 0.814 [0.793, 0.836] | 74.5% | 0.176 | -0.001 [-0.021, +0.020] |

### Concreteness breakdown (within no filters)

| arm | conc 1-2 AUROC (n) | conc 3-4 AUROC (n) | Δ vs plain (1-2) |
|---|---|---|---|
| plain | 0.822 (1944) | 0.791 (2097) |  |
| base | 0.536 (1944) | 0.531 (2097) | -0.287 |
| trained | 0.848 (1944) | 0.796 (2097) | +0.026 |

## qwen3.5-397b

### no filters  —  n=3972  (2048T/1924F)

| arm | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|
| plain | 0.813 [0.799, 0.826] | 74.3% | 0.180 | — |
| base | 0.693 [0.677, 0.709] | 62.3% | 0.291 | -0.120 [-0.134, -0.105] |
| trained | 0.845 [0.833, 0.857] | 76.2% | 0.168 | +0.033 [+0.021, +0.044] |

### all filters  —  n=1076  (566T/510F)

| arm | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|
| plain | 0.837 [0.813, 0.860] | 75.8% | 0.171 | — |
| base | 0.719 [0.688, 0.749] | 64.7% | 0.261 | -0.118 [-0.146, -0.089] |
| trained | 0.850 [0.828, 0.870] | 76.3% | 0.168 | +0.013 [-0.009, +0.035] |

### Concreteness breakdown (within no filters)

| arm | conc 1-2 AUROC (n) | conc 3-4 AUROC (n) | Δ vs plain (1-2) |
|---|---|---|---|
| plain | 0.811 (2546) | 0.812 (1426) |  |
| base | 0.690 (2546) | 0.695 (1426) | -0.120 |
| trained | 0.852 (2546) | 0.834 (1426) | +0.041 |

## Consistency check vs published all-filters numbers

Published numbers use the population WITHOUT the train-overlap exclusion (8B: 8 extra claims), so small diffs are expected there and near-zero elsewhere.

| run | arm | published | recomputed | diff |
|---|---|---|---|---|
| gemma | plain | 0.817 | 0.814 | -0.003 |
| gemma | ao | 0.814 | 0.809 | -0.005 |
| gemma | nla | 0.807 | 0.803 | -0.004 |
| gemma | sae | 0.813 | 0.810 | -0.003 |
| qwen8b | plain | 0.818 | 0.815 | -0.003 |
| qwen8b | base | 0.547 | 0.543 | -0.004 |
| qwen8b | trained | 0.818 | 0.814 | -0.004 |
| q397 | plain | 0.837 | 0.837 | -0.000 |
| q397 | base | 0.722 | 0.719 | -0.003 |
| q397 | trained | 0.852 | 0.850 | -0.002 |

## Yes/No format adherence (first-token argmax), base + trained on the complement

| file | adherent | rate |
|---|---|---|
| 397b_base_unfiltered.jsonl | 2663/2663 | 100.0% |
| 397b_trained_unfiltered.jsonl | 2663/2663 | 100.0% |
| 8b_base_unfiltered.jsonl | 2171/2171 | 100.0% |
| 8b_trained_unfiltered.jsonl | 2171/2171 | 100.0% |

### Known data notes
- gemma `prompt_002230_c0` is in the conc3 eval set but absent from the current bank (bank regenerated after the eval set was frozen); it is kept in both populations via the eval-set file.
- 8 claims in the published 8B conc3 eval set overlap the 8B training data (the training build deduped against the old n1000 set); they are excluded from BOTH populations here, and the published canonical numbers are left untouched.
