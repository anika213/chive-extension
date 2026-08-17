# experiments/common — shared analysis code

Not an experiment: shared slice definitions and analysis helpers imported by the other
`experiments/` dirs.

- `gold_slice.py` — the canonical dataset-slice definitions for the binary claim eval:
  `quality_ids`/`quality_funnel` (the paper's primary population: concreteness ≥ 3 ∩ cf-correct ∩
  not-confounded) and the stricter `gold_ids`/`gold_funnel` (adds the baseline-unsure difficulty
  gate, used only in secondary stress-test analyses). Recomputed from source files on every call;
  never cached. Imported (as `experiments.common.gold_slice`) by the `analyze.py`/figure scripts
  of `main_counterfactual_eval`, `cross_model`, `tool_hyperparameter_sweep`, `self_explanation`,
  `direct_counterfactual_evaluation`, `privileged_access`,
  `unfiltered_evaluation`, and `thinking_interp_eval`.
- `analyze_solvable_slice.py` — standalone CLI that re-scores every arm on the "solvable AND
  hard" subset (cf-correct ∩ plain-unsure) of a run's `binary_evals/`; e.g.
  `PYTHONPATH=. python experiments/common/analyze_solvable_slice.py --binary-evals-dir
  data/runs/<run>/binary_evals --label conc3`.
