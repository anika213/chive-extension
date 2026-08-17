# Thread A — direct counterfactual prediction (base vs trained vs Opus)

Canonical arm: format A (claim as a follow-up turn on the transcript), no-think, first-token Yes/No logprob. Base and trained share this exact prompt; Opus `plain` is the high-effort transcript-reader reference.

**Main-body numbers are the QUALITY subset**: concreteness≥3 ∩ cf-correct ∩ not-confounded, via `experiments/common/gold_slice.quality_funnel` (recomputed from source). The comparator-dependent baseline-unsure gate is deliberately excluded. The cf arm is omitted from the figures because the slice is defined as cf-correct.

> Regenerate with `analyze.py` (writes SUMMARY.md, plots/). Reproduce the eval from scratch with `run_all.sh`.

## Per-run (full scored set)

| model | arm | eval set | n | acc | AUROC | Brier |
|---|---|---|---|---|---|---|
| 8b | base | regular | 1870 | 51.7% | 0.536 | 0.447 |
| 8b | trained | regular | 1870 | 73.1% | 0.799 | 0.183 |
| 8b | base | petri | 1442 | 50.7% | 0.537 | 0.412 |
| 8b | trained | petri | 1442 | 69.8% | 0.784 | 0.190 |
| 397b | base | regular | 1309 | 62.8% | 0.691 | 0.273 |
| 397b | trained | regular | 1309 | 75.1% | 0.835 | 0.173 |
| 397b | base | petri | 708 | 65.4% | 0.706 | 0.257 |
| 397b | trained | petri | 708 | 75.6% | 0.837 | 0.177 |

## Quality subset — base vs Opus vs trained (AUROC [95% CI] and acc)

AUROC with 95% bootstrap CI (2000 resamples, seed 0); acc with Wilson 95% CI. Δ columns are trained minus Opus on the identical quality claim set.

| model | eval set | quality n | base AUROC | Opus AUROC | trained AUROC | trained−Opus | base acc | Opus acc | trained acc |
|---|---|---|---|---|---|---|---|---|---|
| 8b | regular | 1497 | 0.543 [0.514, 0.573] | 0.815 [0.793, 0.836] | 0.814 [0.793, 0.835] | -0.001 | 52.9% | 74.7% | 74.5% |
|  |  | _funnel: scored 1870 → cf-correct 1666 → not-confounded (quality) 1497_ | | | | | | | | |
| 8b | petri | 1196 | 0.547 [0.515, 0.579] | 0.833 [0.808, 0.854] | 0.807 [0.782, 0.832] | -0.025 | 51.7% | 76.3% | 72.0% |
|  |  | _funnel: scored 1442 → cf-correct 1277 → not-confounded (quality) 1196_ | | | | | | | | |
| 397b | regular | 1076 | 0.719 [0.687, 0.747] | 0.837 [0.814, 0.860] | 0.850 [0.828, 0.871] | +0.013 | 64.7% | 75.8% | 76.3% |
|  |  | _funnel: scored 1309 → cf-correct 1173 → not-confounded (quality) 1076_ | | | | | | | | |
| 397b | petri | 580 | 0.736 [0.696, 0.777] | 0.879 [0.851, 0.905] | 0.877 [0.849, 0.904] | -0.001 | 67.2% | 80.7% | 77.6% |
|  |  | _funnel: scored 708 → cf-correct 630 → not-confounded (quality) 580_ | | | | | | | | |
