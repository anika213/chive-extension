# Freeform self-explanation — hint attribution

**Purpose.** A hint (cue) is injected into a prompt and the model produces a behavior. When asked
*why* it answered that way (the bank's `freeform_question`), does its open-ended self-explanation
attribute the answer to the hint? A blinded judge labels each explanation on a 5-state causal
scale (DECISIVE / PARTIAL / DENIES / MENTIONS_ONLY / NO_MENTION); "attributes" = DECISIVE ∪
PARTIAL; accuracy = attributes == `is_true` (the measured counterfactual flip from the hint
banks). Compared for base vs. trained Qwen3-8B and Qwen3.5-397B, plus an **Opus reference**: an
external explainer (no privileged access) that reads the plain-text transcript and is asked the
third-person question ("Why did the model answer (C)?"), judged by the same blinded extractor —
trained > Opus is the introspection signal (mirrors the Figure-4 Opus reference).

**Paper artifact.** Figure `fig:hint_selfexpl` in §`sec:training_hint`
(`self_explanation_hint_pooled_accuracy.png`, copied by `paper/collect_figures.sh`). All numbers
live in the paper and the auto-generated `SUMMARY.md`.

## Setup

Settings / arms (per model: base, trained, opus):

- `mcq_explain` — the MMLU-cue banks in `data/hint_banks/cot_hint/`
  (`mcq_explain/cf_eval_qwen3_8b.jsonl`, `mcq_explain/cf_eval_qwen35_397b.jsonl`).
- `sycophancy` — AITA banks under `data/hint_banks/sycophancy_aita/<model>/`.

Pipeline (each stage resumable, outputs under
`data/experiments/self_explanation_hint/results/`):

1. **Generate** (`generate_selfexpl.py`, generic over any unified claim bank): conversation =
   cued prompt → behavior completion (assistant) → `freeform_question` (user), rendered no-think
   in the exact training format; temperature 0.7, n=1. Jobs: `jobs/selfexpl_8b.sh` (vLLM, 1 GPU),
   `jobs/selfexpl_397b_tinker.sh` (Tinker, API-only). → `results/cache/`.
   The opus arm uses `generate_opus_explanation.py` instead (claude-opus-4-8, effort high,
   API-only): the bank's `transcript` field plus the third-person `freeform_question`; the neutral
   prompt never mentions hints/cues. Job: `jobs/opus_explain_extract.sh` (both stages).
   Scale: 300 rows × 4 banks = 1,200 Opus calls at high effort (order ~$50-100).
2. **Extract** (`extract_attribution.py`, blinded Sonnet judge, resumable):
   `jobs/extract_mcq_sonnet.sh` (API-only). → `results/extracted/`.
3. **Analyze** (`analyze.py`, CPU, instant): pooled headline (300 sycophancy + 300 mcq_explain =
   600 per model/arm, Wilson CI — the paper figure) + per-setting tables + label histograms.
   → `SUMMARY.md` (auto-generated, do not hand-edit),
   `plots/self_explanation_hint_pooled_accuracy.png`.

## Caveats

- **Sycophancy label coverage.** The base/trained sycophancy label files cover more prompts than
  the unified banks (3364 for Qwen3-8B, 958 for Qwen3.5-397B); the banks' 300 prompt_ids are a
  subset. For the pooled headline, `analyze.py` filters every arm to those 300 (exactly 150
  supported / 150 inert; full coverage asserted), so all arms are paired on identical claims. The
  per-setting sycophancy cell reports each arm's full label file, hence the larger `n` there.
- The judge never sees `is_true`, the counterfactual rate, or the model name; a mention of the
  cue is not counted as a causal claim (see `extract_attribution.py`).
