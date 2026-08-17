# Direct counterfactual (method A) — hint introspection

**Purpose.** A hint (cue) is injected into a prompt; the model produces a behavior. Can the model
report that the hint drove its answer? Method A asks the model directly whether removing the hint
would flip its answer and scores the first-token Yes/No **logprob** (`prob_true`, graded by
AUROC). This tests whether counterfactual-prediction training generalizes to a fully held-out
setting.

**Paper artifact.** Figure `fig:hint_direct` (the left panel of Figure `fig:hint`) in
§`sec:training_hint` of `paper/main.tex` — `plots/direct_counterfactual_hint_pooled_fig.png`,
copied into the paper by `paper/collect_figures.sh`. Dataset construction is Appendix
`app:hint_datasets`. All numbers live in the paper and the auto-generated `SUMMARY.md`.

## Setup

**Settings** (one unified Claim bank each, 300 claims = 150 supported / 150 inert):
- `mcq_explain` — MMLU MCQs with an injected cue (professor / answer-mark / answer-comment /
  sycophancy-school / previous-answer). Built by `chive/hint_datasets/cot_hint/`
  (`generate_until.py` is the entry point); banks at
  `data/hint_banks/cot_hint/mcq_explain/cf_eval_<model>.jsonl`.
- `sycophancy` — AITA posts where the user states their own opinion (the cue). Built by
  `chive/hint_datasets/sycophancy_aita/generate_dataset.py`; banks at
  `data/hint_banks/sycophancy_aita/<model>/cf_eval_bank.jsonl`.
**Arms** on Qwen3-8B and Qwen3.5-397B:
- `base` / `trained` — local logprob scoring via
  `experiments/direct_counterfactual_evaluation/run_structured_claim_eval.py` (vLLM 8B / Tinker
  397B). These consume `messages` + `chosen_completion` as the model's OWN chat-templated
  conversation.
- `opus` — external transcript-reader reference (no privileged access) via
  `python -m chive.binary_eval.run_binary_eval --answerer plain`, which reads the bank's plain-text
  `transcript` field. The banks are BinaryClaim supersets, so it loads them directly (no adapter).

## Reproduce

```bash
python experiments/direct_counterfactual_hint/analyze.py
# CPU: reads data/experiments/direct_counterfactual_hint/results/scored/,
# writes SUMMARY.md + plots/direct_counterfactual_hint_pooled_fig.png (the paper figure)

sbatch experiments/direct_counterfactual_hint/jobs/eval_8b_cf.sh <bank> <out> <base|lora-dir>
# score one 8B arm (vLLM, 1 GPU); 397B local arms run run_structured_claim_eval.py
# --backend tinker directly (CPU/API only)

bash experiments/direct_counterfactual_hint/jobs/run_opus_baseline.sh
# score all Opus arms (API only, resumable)
```

The headline (paper figure) is the POOLED in-distribution result: 300 sycophancy + 300
mcq_explain = 600 claims per (model, arm), one error bar (bootstrap AUROC / Wilson accuracy).

Scored per-claim outputs:
`data/experiments/direct_counterfactual_hint/results/scored/<setting>_<model>_<arm>.jsonl`
(`{claim_id, is_true, prob_true, pred_true, ...}`).