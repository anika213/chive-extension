# Direct counterfactual prediction — base vs trained vs Opus

**Purpose.** Measures whether counterfactual-prediction training generalizes to held-out
investigations: each model (Qwen3-8B and Qwen3.5-397B-A17B) answers held-out binary counterfactual
claims about its own behavior, from an in-distribution prompt source (WildChat) and an
out-of-distribution one (PETRI), compared against its base model and an Opus transcript-reading
reference.

**Paper artifact.** Figure `fig:heldout_direct` in §`sec:training_heldout` of `paper/main.tex`
(`plots/direct_counterfactual_evaluation_fig_auroc.png`, copied into the paper by
`paper/collect_figures.sh`; an accuracy companion `..._fig_accuracy.png` is generated alongside).
All numbers live in the paper and the auto-generated `SUMMARY.md`.

## Setup

- **Banks:** `eval_set_conc3_n2000.jsonl` from the four run dirs
  `data/runs/opus_investigates_{qwen3_8b,qwen3_8b_petri,qwen3p5_397b,qwen3p5_397b_petri}_*_n30/`.
  Each model is scored on **its own** investigations' claims.
- **Canonical arm:** format A — the claim is presented as a real follow-up turn on the model's own
  chat-templated transcript — no-think, first-token Yes-vs-No logprob softmax → graded `prob_true`.
  Base and trained share this **byte-identical** prompt; the only difference is whether the LoRA
  adapter is loaded. Both go through `run_structured_claim_eval.py` (vLLM backend for 8B, Tinker
  for 397B). The Opus `plain` reference is read from each run dir's
  `binary_evals/plain_claude-opus-4-8_high_conc3.jsonl`.
- **Reported population:** `analyze.py` restricts the figures and the main table to the paper's
  quality slice (concreteness≥3 ∩ cf-correct ∩ not-confounded) via
  `experiments/common/gold_slice.quality_funnel`, recomputed fresh from source.

## Reproduce

```bash
bash experiments/direct_counterfactual_evaluation/run_all.sh
# 8 runs: 8B = 4 parallel sbatch vLLM jobs (jobs/eval_8b_vllm.sh, 1 GPU each);
# 397B = 4 sequential Tinker runs (CPU/API only). Outputs land in
# data/experiments/direct_counterfactual_evaluation/results/canonical/, then analyze.py runs.

python experiments/direct_counterfactual_evaluation/analyze.py   # CPU; writes SUMMARY.md + plots/
```

Trained checkpoints are referenced in `run_all.sh`: the 8B LoRA is the released
`adamkarvonen/qwen3_8b_counterfactual_e1_kl0` (download to `checkpoints/`, see the script header);
the 397B runs via its `tinker://` sampler weights (adapter weights also released at
`adamkarvonen/qwen3p5_397b_counterfactual_e1_kl0`).

### Prompt-format ablation (folded into the canonical arm)

Before the canonical arm was fixed, the **base-model** prompting was ablated: format (a)
(transcript as the model's own chat turns) vs (b) (flattened "Opus-style" text, ± system prompt),
each no-think first-token Yes/No vs think→JSON-probability. The outcome is baked into the
canonical arm above (format A, no-think, identical for base and trained); the ablation's
side scripts were removed after the choice was fixed.

## Caveats

- The quality slice is defined as cf-correct, so the **cf arm is excluded from the figures by
  construction**; the comparator-dependent baseline-unsure gate is deliberately excluded from the
  population (see `experiments/common/gold_slice.py`).
- `run_all.sh` calls an optional cluster-local preflight helper
  (`jobs/find_zombie_gpu_processes.sh`); the call is `|| true`-guarded and can be ignored
  off-cluster.
