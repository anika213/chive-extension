# Subject behavior similarity: Qwen3-8B vs Llama-3.1-8B on the eval banks

For every (prompt, behavior_question) unit in each bank, the OTHER subject model's 30
baseline completions (same prompt pool, identical conversations) were graded with the
bank's classifier question (grader: Sonnet 4.6, thinking 30k — same model and prompt
construction as the pipeline that produced the stored rates; the pipeline used thinking
2048, drift measured by the self re-grade below). `rate_self` = the bank's stored
`k_baseline/30`; `rate_cross` = the other model's graded `k/30`.

> Regenerate: `run_all.sh` in this folder (grade → analyze → spot check).

## Headline

**The two subjects do NOT behave the same on the banks' probed behaviors** — the
privileged-access cross-predictor parity is not explained by behavioral identity of the
subjects. Pooled over
1251 units, the cross-model rate correlates with the stored rate at only
r = 0.19 (mean |Δ| = 41pp; 72.7% of units differ by >20pp, 32.1% by >50pp), while the self re-grade
reproducibility check shows the grading instrument itself is tight (r ≈ 0.96, mean |Δ| ≈ 2-3pp)
— the disagreement is real behavioral divergence, not grader noise. The effect is symmetric
and signed: each bank's behaviors are far more characteristic of its own subject (mean rate
0.64 / 0.56 on the Qwen / Llama banks) than of the other model (0.32 / 0.31),
as expected since classifier questions are authored from observed subject behavior.
Even where the bank subject is fully deterministic (30/30), the other model matches (≥27/30)
only 30.9% of the time.

Interpretation caveats: (1) self rates are >0 in ~97% of units by construction, so the
'>0% present' agreement/κ rows are near-degenerate — the graded thresholds (≥10/30/50%) are
the informative ones, and κ is ≈0-0.14 everywhere; (2) qualitative review
(`results/spot_check.md`) confirms top disagreements are genuine behavior differences
(tool-call target choice, empty-array vs function call, response language), not grader
artifacts.

## Rate agreement

| metric | Qwen3-8B bank → Llama-3.1-8B | Llama-3.1-8B bank → Qwen3-8B | pooled |
|---|---|---|---|
| n_units | 649 | 602 | 1251 |
| pearson_r | 0.215 | 0.165 | 0.191 |
| spearman_rho | 0.229 | 0.143 | 0.189 |
| mean_abs_delta_pp | 41.644 | 39.839 | 40.775 |
| median_abs_delta_pp | 36.667 | 36.667 | 36.667 |

## Disagreement prevalence

| metric | Qwen3-8B bank → Llama-3.1-8B | Llama-3.1-8B bank → Qwen3-8B | pooled |
|---|---|---|---|
| frac_abs_delta_gt20pp | 71.5% | 73.9% | 72.7% |
| frac_abs_delta_gt50pp | 33.4% | 30.6% | 32.1% |

## Binarized 'behavior present' agreement (threshold sensitivity)

| metric | Qwen3-8B bank → Llama-3.1-8B | Llama-3.1-8B bank → Qwen3-8B | pooled |
|---|---|---|---|
| agree >0% (k>=1) | 0.752 | 0.628 | 0.692 |
| kappa >0% (k>=1) | 0.044 | -0.012 | 0.009 |

| metric | Qwen3-8B bank → Llama-3.1-8B | Llama-3.1-8B bank → Qwen3-8B | pooled |
|---|---|---|---|
| agree >=10% (k>=3) | 0.610 | 0.537 | 0.575 |
| kappa >=10% (k>=3) | 0.025 | 0.013 | 0.017 |

| metric | Qwen3-8B bank → Llama-3.1-8B | Llama-3.1-8B bank → Qwen3-8B | pooled |
|---|---|---|---|
| agree >=30% (k>=9) | 0.448 | 0.422 | 0.436 |
| kappa >=30% (k>=9) | 0.021 | 0.029 | 0.025 |

| metric | Qwen3-8B bank → Llama-3.1-8B | Llama-3.1-8B bank → Qwen3-8B | pooled |
|---|---|---|---|
| agree >=50% (k>=15) | 0.490 | 0.535 | 0.512 |
| kappa >=50% (k>=15) | 0.102 | 0.118 | 0.110 |

## Deterministic-behavior slices

Units where the bank's subject is deterministic; does the other model match within 3/30?

| metric | Qwen3-8B bank → Llama-3.1-8B | Llama-3.1-8B bank → Qwen3-8B | pooled |
|---|---|---|---|
| n_self_0of30 | 22 | 9 | 31 |
| frac_match_le3_given_self_0 | 54.5% | 66.7% | 58.1% |
| n_self_30of30 | 112 | 40 | 152 |
| frac_match_ge27_given_self_30 | 29.5% | 35.0% | 30.9% |

## Self re-grade reproducibility check

Same instrument as the cross grading, applied to the bank subject's own completions;
measures grader drift vs the stored pipeline rates (graded with thinking 2048).

| direction | n | pearson r | mean |Δ| (pp) | max |Δ| (pp) |
|---|---|---|---|---|
| Qwen3-8B bank self re-grade | 47 | 0.962 | 2.8 | 46.7 |
| Llama-3.1-8B bank self re-grade | 43 | 0.947 | 2.2 | 33.3 |

## Notes / caveats

- Truncation (pre-existing, both sides): fraction of graded completions with `finish_reason=='length'`: Llama-3.1-8B 43.3%, Qwen3-8B 50.1%. The stored rates were computed on equally-truncated self completions.
- Baseline behavior only; counterfactual-side rates for the other model do not exist (see the caveats in README.md).
- Qualitative review: `results/spot_check.md` (make_spot_check.py).
