# Unfiltered-dataset evaluation (concreteness 1–2 included)

**Purpose.** Test whether the paper's two main results (no interp-tool uplift; the trained
direct-counterfactual models reach the Opus transcript reader) survive removing the dataset
filters, by scoring every arm on the **full unfiltered claim bank** of each run ("no filters")
and comparing against the paper's quality slice ("all filters").

**Paper artifact.** Appendix `app:allruns` ("Unfiltered dataset results"): Figures
`fig:unfiltered_tools` (`plots/gemma_unfiltered_overall_score.png`) and `fig:unfiltered_trained`
(`plots/unfiltered_direct_cf_auroc.png`), copied by `paper/collect_figures.sh`. All numbers live
in the paper and the auto-generated `SUMMARY.md`.

## Setup

- **Runs (WildChat only):** Gemma-3-27B-IT, Qwen3-8B, Qwen3.5-397B
  (`data/runs/opus_investigates_{gemma3_27b_it,qwen3_8b,qwen3p5_397b}_5000_thorough_n30`).
- **Arms:** Opus `plain` on all three; interp tools (AO/NLA/SAE) on Gemma only; trained
  direct-counterfactual models (base + trained) on 8B (vLLM LoRA) and 397B (Tinker). No cf arm.
- **Populations** (recomputed from source on every `analyze.py` call):
  *no filters* = conc3 eval set ∪ the unfiltered complement
  (`build_unfiltered_complement.py` → `data/experiments/unfiltered_evaluation/eval_sets/`),
  minus train-overlap claim ids; *all filters* = `gold_slice.quality_ids` minus the same overlap.
  Scored rows per arm = the run dir's published `*_conc3.jsonl` ∪ this experiment's
  `*_unfiltered.jsonl` (`data/experiments/unfiltered_evaluation/results/`), merged by claim id.

## Reproduce

    bash experiments/unfiltered_evaluation/run_all.sh   # idempotent; every Opus arm --resume

Stages: build complements (CPU) → trained-model arms (2× 1-GPU vLLM sbatch for 8B; Tinker
API-only for 397B) → Gemma GPU infra (capture + AO servers, NLA/SAE cache precompute over the
complement) → Opus arms sequentially (API) → `analyze.py`, which writes `SUMMARY.md`
(auto-generated, do not hand-edit). Figures: `make_figures.py`
(`PYTHONPATH=. python experiments/unfiltered_evaluation/make_figures.py`), which reuses the
main-body plotting code so populations match `SUMMARY.md` exactly.

## Caveats

- **Train-overlap exclusion.** The trained models' data was deduped against an older n1000 eval
  set, so a small number of bank claims share a (whitespace/case-normalized) conversation with
  the training data. Those claim ids (`results/train_overlap_<run>.json`) are excluded from BOTH
  populations here, for every arm, so all arms score the identical set. Gemma has no trained arm
  and no exclusion. In particular, 8 claims of the published 8B conc3 eval set are excluded here
  while the published canonical numbers are left untouched (≤0.4% of the set); `analyze.py`
  prints a consistency check against the published all-filters numbers.
- **The no-filters population is the full bank, not re-balanced** (47–53% TRUE), so AUROC is the
  primary metric and `acc` is reported with the TRUE base rate.
- **Format adherence.** `analyze.py` reports the Yes/No first-token adherence rate of the
  base/trained arms on the complement (conc1–2 transcripts are off the distribution where 100%
  adherence was verified); a rate <99% invalidates the base comparison rather than just
  degrading it.
- One Gemma claim (`prompt_002230_c0`) is in the conc3 eval set but absent from the current bank
  (the bank was regenerated after the eval set was frozen); it is kept via the eval-set file.
- The complements deliberately do not match the `eval_set_conc3_n*.jsonl` glob that
  `gold_slice._eval_set_path` asserts on, and the unfiltered arm files are kept out of the run
  dirs' canonical `binary_evals/` so the main plotters never pick them up.
