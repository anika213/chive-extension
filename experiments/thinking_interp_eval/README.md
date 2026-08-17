# Interpretability-tool eval on the reasoning-mode (thinking) run

**Purpose.** Re-run the interpretability-tool evaluation of the main body on the Qwen3-8B
**reasoning-enabled** (thinking budget 1024) run, to test whether the tool-vs-transcript
comparison changes when the transcript contains an explicit reasoning trace.

**Paper artifact.** Figure `fig:cot_interp` in Appendix `app:cot`
(`plots/thinking_overall_score.png`, copied by `paper/collect_figures.sh`).

## Setup / Reproduce

- **Run dir / bank:** `data/runs/opus_investigates_qwen3_8b_think1024_5000_thorough_n30` with its
  `eval_set_conc3_n2000.jsonl`. The pipeline run itself and the plain/cf/AO/NLA arms are produced
  by `experiments/thinking_investigation/` (`eval_build_and_plain.sh`, `eval_gpu_arms.sh`); see
  that README.
- **SAE arm:** `jobs/eval_sae_arm.sh` (sbatch; CPU-only — the SAE arm is a pure cache lookup, no
  GPU/server) scores the SAE tool arm into the run's `binary_evals/` with `--label conc3` so
  `make_figure.py` auto-discovers it.
- **Figure:**

      PYTHONPATH=. python -m experiments.thinking_interp_eval.make_figure

  reuses `experiments/main_counterfactual_eval/analyze.py`'s arm discovery, quality-slice
  restriction (`experiments.common.gold_slice.quality_funnel`) and plotting code, so the style,
  metrics and population are identical to the main figure. Output: `plots/thinking_overall_score.png`.

## Caveats

- The reported population is the paper's quality slice (concreteness ≥ 3 ∩ cf-correct ∩
  not-confounded) intersected with the claims scored by every arm; the `cf` arm is dropped from
  the bars (near-ceiling on this slice by construction).
- The `cf` arm for this run must re-run the target in thinking mode
  (`--cf-thinking-budget 1024 --cf-max-tokens 2048`); a no-think re-run does not reproduce
  thinking-mode behaviors (see `experiments/thinking_investigation/README.md`).
