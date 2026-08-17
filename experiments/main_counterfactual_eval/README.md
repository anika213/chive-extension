# Main counterfactual-tool evaluation (binary causal-claim eval, cf7)

**Purpose.** The paper's main interpretability-tool evaluation: score the transcript-only
baseline, the counterfactual tool, and the activation tools (AO/NLA/SAE) as predictor-agent
arms on the binary counterfactual-claim banks of all five runs (Gemma-3-27B-IT and Qwen3-8B /
Qwen3.5-397B on WildChat and PETRI), and test whether any tool provides uplift over the
baseline.

**Paper artifact.** Figure `fig:interp_eval_main` (`figures/gemma_overall_score.png`,
§`sec:interp_eval`) and Figure `fig:qwen_interp` (`figures/qwen8b_overall_score.png`, Appendix
`app:qwen_interp`), both copied in by `paper/collect_figures.sh`; the funnel counts of Table
`tab:eval_dataset_funnel` (Appendix `app:eval_dataset`) and the all-runs numbers of Appendix
`app:allruns` are also sourced here. All numbers live in the paper and the auto-generated
`SUMMARY.md`.

## Setup

- **Runs:** the five run dirs under `data/runs/` named in `run_all.sh`
  (`opus_investigates_{gemma3_27b_it,qwen3_8b,qwen3p5_397b}_5000_thorough_n30` and the
  `*_petri_thorough_n30` variants). Each bank is the run's `eval_set_conc3_n2000.jsonl`;
  scored arms live in the run's `binary_evals/*_claude-opus-4-8_high_conc3.jsonl`.
- **Arms per run:** Gemma: plain, ao, cf, nla, sae. Qwen3-8B (5000 + petri): plain,
  ao (single + multi LoRA), cf, nla. Qwen3.5-397B (5000 + petri): plain, cf only (no
  activation tools). Answerer is `claude-opus-4-8` at effort `high` throughout.
- **Reported population:** `analyze.py` restricts every run to the paper's final evaluation
  dataset — the quality slice (concreteness≥3 ∩ cf-correct ∩ not-confounded, via
  `experiments/common/gold_slice.quality_funnel`) — and appends a post-hoc section on the
  gold slice (quality ∩ baseline-unsure).

## Reproduce

    # end-to-end (Slurm): cf7 bank build -> concreteness -> eval set -> GPU servers +
    # NLA/SAE caches -> all scored arms -> analysis. Idempotent; every arm uses --resume.
    bash experiments/main_counterfactual_eval/run_all.sh

    # analysis only (CPU; reads the scored arms shipping with the run dirs)
    python experiments/main_counterfactual_eval/analyze.py --mode both

Stages A/B (bank + concreteness + eval set) are CPU + Anthropic API (batch mode, ~1h/run).
The Gemma/Qwen arms need live vLLM servers (cf target, AO server, and a capture server for the
NLA/SAE caches; 1 GPU each, submitted by the script). The 397B cf arm runs against a Tinker
bf16 target — API only, no local GPU. `analyze.py` writes `SUMMARY.md` (all numbers;
auto-generated, do not hand-edit) and `plots/` (four figures per run slug).

The thinking-target run is analyzed by `experiments/thinking_interp_eval/` (the paper's
`fig:cot_interp`).

## Caveats

- All metrics are paired on each run's common claim set across arms, on the quality slice;
  ~3 dead-zone claims carry known common-mode label noise, which cancels in paired deltas.
- In the post-hoc/gold section the counterfactual row is tautological (the slice is *defined*
  by cf being correct); the meaningful comparison there is transcript-only vs AO/NLA/SAE.
- `run_all.sh` assumes each run dir already contains the upstream pipeline outputs
  (completions → screening → investigations → verification) from `chive/pipeline/run_pipeline.py`.
