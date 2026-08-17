# Freeform self-explanation — counterfactual simulatability

**Purpose.** We train a target model to explain its own behavior. Does handing that freeform
self-explanation to an external *simulator* help the simulator predict the target's behavior on a
counterfactual claim? Five simulators (weak→strong) each predict the same binary claims from the
transcript alone (`plain`) vs. transcript + the **trained** target's self-explanation (`selfexpl`)
vs. transcript + the **base** model's zero-shot explanation (`selfexpl_base`, negative control),
paired within simulator so Δ isolates the explanation.

**Paper artifact.** Figure `fig:heldout_selfexpl` in §`sec:training_heldout`
(`figure_selfexpl_by_target.png`, Qwen3-8B target) and Figure `fig:selfexpl_397b` in Appendix
`app:selfexpl_397b` (`figure_selfexpl_397b.png`, Qwen3.5-397B target); both copied into the paper
by `paper/collect_figures.sh`. All numbers live in the paper and the auto-generated `SUMMARY.md`
(written by `analyze.py` to `data/experiments/self_explanation/results/SUMMARY.md`).

## Design

- **Targets × eval sets (3 runs):** `qwen8b` (`data/runs/opus_investigates_qwen3_8b_5000_thorough_n30`),
  `q397_5000` (`.../opus_investigates_qwen3p5_397b_5000_thorough_n30`), `q397_petri`
  (`.../opus_investigates_qwen3p5_397b_petri_thorough_n30`), each on its
  `eval_set_conc3_n2000.jsonl` bank. `analyze.py` restricts every reported number to the paper's
  quality slice (`experiments.common.gold_slice.quality_ids`).
- **Simulator panel (weak→strong, 3 families):** Gemma-3-12B, Gemma-3-27B (vLLM), Qwen3.6-27B
  (vLLM, thinking on), Haiku-4.5, Opus-4.8 (Anthropic API).
- **Output channel:** plain-text `PROBABILITY: <p>` for every simulator; parsing is strict (a
  non-conforming output is dropped and counted, never guessed). Metric: AUROC with paired
  claim-bootstrap ΔAUROC CIs. Prompts live verbatim in `run_simulator.py`.
- A secondary `--variant prioritize` (predict *solely* from the explanation) writes to separate
  `results/<slug>_prioritize/` dirs and `SUMMARY_prioritize.md`; the paper figures use the default
  `follow` variant.

## Reproduce

    bash experiments/self_explanation/run_all.sh

which sequences: `jobs/precompute_base_selfexpl.sh` (base explanation caches: 8B via vLLM, 1 GPU;
397B via Tinker, API-only) → one `jobs/run_sim_vllm.sh <hf_model_id>` job per local simulator
(1 GPU each) plus the two Anthropic simulators inline (`run_simulator.py --backend anthropic`,
no GPU) → `analyze.py` (CPU). The **trained** self-explanation caches ship with the data release
in `data/experiments/self_explanation/results/<slug>/cache/`; regenerating them needs the trained
checkpoints (`jobs/precompute_selfexpl_8b.sh`, `precompute_selfexpl_cache.py`). Scored arm jsonl +
`.meta.json` sidecars (skip/truncation counts) land in
`data/experiments/self_explanation/results/<slug>/`; figures are re-plotted by
`plot_selfexpl_by_target.py` and `plot_selfexpl_397b.py` into `plots/`.

## Caveats

- **Same-framing rule.** All arms use the same simulate-framing system prompt
  (`SIM_SYSTEM_BASE`); the selfexpl arms only append one sentence plus the explanation block.
  Mixing framings between arms inflates the weak-simulator Δ by 2–3×, so do not re-run arms with
  different system prompts and compare them.
- **Leakage-free 8B checkpoint.** The qwen8b `selfexpl` arm uses the `_filtered` (train/eval
  leakage-free) retrain of the trained explainer; an earlier leaked checkpoint showed a large
  spurious weak-simulator gain.
- **Baseline tie-back.** `plain` here uses the simulate framing and text channel, so Opus's plain
  AUROC is not byte-identical to the main-eval Opus baseline (tool channel); it matches within
  noise.
- **Base-explanation prompt.** The base (un-fine-tuned) arm appends `BASE_EXPLANATION_SUFFIX` to
  the self-explanation question (without it the base model apologizes and regenerates instead of
  explaining); the trained arm gets the bare question, byte-identical to training.
- `q397_petri` is analyzed in `SUMMARY.md` but not shown in the paper figures.
