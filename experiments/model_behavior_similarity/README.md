# Subject behavior similarity: Qwen3-8B vs Llama-3.1-8B

**Purpose.** Companion check for the privileged-access cross-predictor experiment
(`experiments/privileged_access`): does its predictor parity merely reflect the two subject
models behaving identically on the probed behaviors? For every (prompt, behavior_question)
unit in each model's claim bank, grade the *other* subject model's 30 baseline completions
with the bank's classifier question and compare behavior rates.

**Paper artifact.** None — supporting analysis for Appendix `app:privileged`, not directly
cited in `paper/main.tex`. All numbers live in the auto-generated `SUMMARY.md`.

## Setup

- **Runs:** `data/runs/opus_investigates_qwen3_8b_5000_thorough_n30` and
  `data/runs/opus_investigates_llama3_1_8b_5000_thorough_n30`. Both cover the identical
  5,000-prompt pool with byte-identical conversations and 30 baseline completions per prompt,
  so the cross-model generations already exist — this is a pure API-grading experiment, no GPU.
- **Unit:** (prompt_id, behavior_question); 698 Qwen-bank + 643 Llama-bank units.
  `rate_self` is the bank's stored `k_baseline/30`; `rate_cross` is graded here with the
  pipeline's own grading instrument (`grade_completions` from `chive/pipeline/investigate.py`,
  grader Sonnet 4.6). A 50-unit-per-bank *self re-grade* of the bank subject's own completions
  measures grader drift against the stored rates.
- **Outputs:** `data/experiments/model_behavior_similarity/results/verdicts.jsonl` (ships with
  the release data); the qualitative review is rendered to `results/spot_check.md` in the same
  data dir by `make_spot_check.py`.

## Reproduce

    bash experiments/model_behavior_similarity/run_all.sh   # grade -> analyze -> spot check

CPU/API only. `grade_cross.py` is resumable (rows already in `verdicts.jsonl` are skipped);
`analyze.py` and `make_spot_check.py` re-run from the shipped verdicts alone. `analyze.py`
writes `SUMMARY.md` (all numbers; auto-generated, do not hand-edit) and `plots/`.

## Caveats

- **Baseline behavior only.** The banks' `is_true` labels also depend on *counterfactual*
  rates, and the other model's completions for the modified conversations do not exist. This
  experiment therefore answers "do the subjects behave the same on the probed behaviors," not
  "would the banks' labels transfer across subjects." (This is the caveat the `SUMMARY.md`
  notes section points to.)
- The cross grading uses the same grader model and prompt construction as the pipeline that
  produced the stored rates but a larger thinking budget (30k vs the pipeline's 2048); the
  self re-grade check bounds the resulting drift.
- Stage-1 completions are heavily length-truncated on both sides (pre-existing and symmetric:
  the stored rates were computed on equally-truncated self completions).
- Classifier questions are authored from the bank subject's observed behavior, so self and
  cross rates are not exchangeable populations; self rates are >0 for ~97% of units by
  construction, making the ">0% present" agreement rows near-degenerate.
