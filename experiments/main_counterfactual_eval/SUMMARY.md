# Binary causal-claim eval — final results (cf7, QUALITY slice)

Answerer: `claude-opus-4-8`, effort `high`. AUROC via rank-sum; 95% bootstrap CIs (2000 resamples, seed 0). `acc` = pred_true vs is_true. Brier = mean((prob_true − is_true)²). ΔAUROC is paired (tool − plain) over the identical claim set. cf7: one unsigned claim per factor, `is_true = supported`.

> **Population: the QUALITY slice** (concreteness≥3 ∩ cf-correct ∩ not-confounded, via `gold_slice.quality_funnel`) — the paper's primary population, consistent with the direct-prediction and self-explanation figures. The comparator-dependent baseline-unsure gate is NOT applied here (see the post-hoc/gold section below for that stress slice). Every arm within a run scores the identical slice, so paired analysis uses the common set. ~3 dead-zone claims carry known common-mode label noise: <0.1% absolute AUROC, cancels in paired Δ.

> Regenerate with `analyze.py` (also writes the plots/ figures); reproduce the whole eval from scratch with `run_all.sh`.


## gemma-3-27b-it  —  n=1294 claims  (682 supported / 612 inert)

| arm | detail | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|---|
| Transcript Baseline | — | 0.814 [0.791, 0.838] | 73.6% | 0.179 | — |
| Counterfactual | — | 1.000 [1.000, 1.000] | 100.0% | 0.012 | +0.186 [+0.162, +0.209] |
| AO | latentqa_cls_past_lens_gemma-3-27b-it | 0.809 [0.786, 0.831] | 72.7% | 0.180 | -0.005 [-0.017, +0.007] |
| NLA | gemma3-27b-L41 | 0.803 [0.778, 0.825] | 71.3% | 0.186 | -0.012 [-0.024, +0.001] |
| SAE | gemma-scope-2-27b-it_layer_40_width_65k_l0_medium | 0.810 [0.786, 0.831] | 71.9% | 0.180 | -0.005 [-0.016, +0.006] |

## qwen3-8b  —  n=1497 claims  (778 supported / 719 inert)

| arm | detail | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|---|
| Transcript Baseline | — | 0.815 [0.793, 0.836] | 74.7% | 0.177 | — |
| Counterfactual | — | 1.000 [1.000, 1.000] | 100.0% | 0.012 | +0.185 [+0.163, +0.207] |
| AO | latentqa_cls_past_lens_addition_Qwen3-8B | 0.812 [0.791, 0.833] | 74.0% | 0.178 | -0.003 [-0.013, +0.008] |
| NLA | qwen3-8b-baseline-nosalvage | 0.809 [0.786, 0.831] | 73.8% | 0.181 | -0.006 [-0.017, +0.005] |
| SAE | qwen3-8b-saes_trainer_2 | 0.812 [0.791, 0.834] | 74.1% | 0.177 | -0.002 [-0.014, +0.008] |

## qwen3.5-397b (5000)  —  n=1076 claims  (566 supported / 510 inert)

| arm | detail | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|---|
| Transcript Baseline | — | 0.837 [0.813, 0.860] | 75.8% | 0.171 | — |
| Counterfactual | — | 1.000 [1.000, 1.000] | 100.0% | 0.013 | +0.163 [+0.139, +0.186] |

## qwen3.5-397b (petri)  —  n=580 claims  (304 supported / 276 inert)

| arm | detail | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|---|
| Transcript Baseline | — | 0.879 [0.849, 0.905] | 80.7% | 0.151 | — |
| Counterfactual | — | 1.000 [1.000, 1.000] | 100.0% | 0.010 | +0.121 [+0.095, +0.150] |

## qwen3-8b (petri)  —  n=1195 claims  (617 supported / 578 inert)

| arm | detail | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |
|---|---|---|---|---|---|
| Transcript Baseline | — | 0.832 [0.809, 0.855] | 76.2% | 0.169 | — |
| Counterfactual | — | 1.000 [1.000, 1.000] | 100.0% | 0.012 | +0.168 [+0.145, +0.191] |
| AO | latentqa_cls_past_lens_addition_Qwen3-8B | 0.827 [0.802, 0.850] | 75.5% | 0.170 | -0.006 [-0.016, +0.005] |
| NLA | qwen3-8b-baseline-nosalvage | 0.833 [0.811, 0.855] | 74.8% | 0.171 | +0.001 [-0.011, +0.012] |


---

# Post-hoc subset: clean, cf-solvable, baseline-unsure

Per run, the subset = claims where **(1)** the counterfactual arm is correct, **(2)** the Sonnet `edit_cleanliness` rating is not `confounded`, and **(3)** the transcript-only baseline is unsure (`prob_true` ∈ [0.25, 0.75]). This is the strongest test for the activation tools: a clean, answerable claim where the transcript alone is ambiguous — so there is genuine headroom and a recoverable answer. If AO/NLA/SAE still don't beat plain here, they don't help even on the slice most favorable to them.

> **Note:** the `counterfactual` row is tautologically AUROC 1.0 / acc 100% — the subset is *defined* by cf being correct. The meaningful comparison is **transcript-only vs AO/NLA/SAE**: on this clean, recoverable, baseline-unsure slice the activation tools still sit at/below plain.


## gemma-3-27b-it — subset n=902  (funnel: scored 1637 → cf-correct 1433 → not-confounded 1294 → gold 902)

| arm | detail | n | AUROC | acc | ΔAUROC vs plain |
|---|---|---|---|---|---|
| Transcript Baseline | — | 902 | 0.739 | 67.3% | — |
| Counterfactual | — | 902 | 1.000 | 100.0% | +0.261 |
| AO | latentqa_cls_past_lens_gemma-3-27b-it | 902 | 0.728 | 66.6% | -0.011 |
| NLA | gemma3-27b-L41 | 902 | 0.718 | 64.9% | -0.021 |
| SAE | gemma-scope-2-27b-it_layer_40_width_65k_l0_medium | 902 | 0.730 | 65.1% | -0.009 |

## qwen3-8b — subset n=973  (funnel: scored 1870 → cf-correct 1666 → not-confounded 1497 → gold 973)

| arm | detail | n | AUROC | acc | ΔAUROC vs plain |
|---|---|---|---|---|---|
| Transcript Baseline | — | 973 | 0.720 | 67.5% | — |
| Counterfactual | — | 973 | 1.000 | 100.0% | +0.280 |
| AO | latentqa_cls_past_lens_addition_Qwen3-8B | 973 | 0.712 | 66.6% | -0.008 |
| NLA | qwen3-8b-baseline-nosalvage | 973 | 0.710 | 66.6% | -0.010 |
| SAE | qwen3-8b-saes_trainer_2 | 973 | 0.716 | 66.9% | -0.005 |

## qwen3.5-397b (5000) — subset n=751  (funnel: scored 1309 → cf-correct 1173 → not-confounded 1076 → gold 751)

| arm | detail | n | AUROC | acc | ΔAUROC vs plain |
|---|---|---|---|---|---|
| Transcript Baseline | — | 751 | 0.738 | 68.4% | — |
| Counterfactual | — | 751 | 1.000 | 100.0% | +0.262 |

## qwen3.5-397b (petri) — subset n=387  (funnel: scored 708 → cf-correct 630 → not-confounded 580 → gold 387)

| arm | detail | n | AUROC | acc | ΔAUROC vs plain |
|---|---|---|---|---|---|
| Transcript Baseline | — | 387 | 0.790 | 73.6% | — |
| Counterfactual | — | 387 | 1.000 | 100.0% | +0.210 |

## qwen3-8b (petri) — subset n=748  (funnel: scored 1441 → cf-correct 1276 → not-confounded 1195 → gold 748)

| arm | detail | n | AUROC | acc | ΔAUROC vs plain |
|---|---|---|---|---|---|
| Transcript Baseline | — | 748 | 0.742 | 69.0% | — |
| Counterfactual | — | 748 | 1.000 | 100.0% | +0.258 |
| AO | latentqa_cls_past_lens_addition_Qwen3-8B | 748 | 0.724 | 67.8% | -0.018 |
| NLA | qwen3-8b-baseline-nosalvage | 748 | 0.743 | 67.2% | +0.001 |
