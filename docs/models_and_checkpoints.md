# Models and checkpoints (release)

Everything released for the paper lives on HuggingFace under `adamkarvonen/`. Base models come
from their original repos (`Qwen/Qwen3-8B`, `Qwen/Qwen3-32B-FP8`, `google/gemma-3-27b-it`,
`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3.5-397B-A17B-FP8`). All adapters are LoRA repos
with the adapter files at the repo root — pass the hub id directly (PEFT / vLLM) or
`hf download <id> --local-dir checkpoints/<name>` first (the job scripts assume the latter).

## Dataset

- `adamkarvonen/chive-data` — investigation datasets + prompt pools.
  Materialize locally with `python scripts/download_data.py` (mapping in `scripts/hf_layout.py`).
  Note: the HF dataset is datasets-only; per-experiment eval result files (`data/experiments/`,
  run-dir `binary_evals/`, NLA/SAE caches) are not part of it.

## Trained LoRAs (Section: training models to explain their own behavior)

| HF repo | Base | What |
|---|---|---|
| `qwen3_8b_counterfactual_e1_kl0` | Qwen3-8B | main counterfactual-prediction LoRA (the paper's trained 8B) |
| `qwen3_8b_counterfactual_e1_kl1` | Qwen3-8B | KL-regularized variant (activation-fidelity appendix) |
| `qwen3_8b_selfexpl_structured_e3_kl0` | Qwen3-8B | freeform self-explanation LoRA (structured target; the paper's selfexpl arm) |
| `qwen3_8b_selfexpl_first_person_e3_kl0` | Qwen3-8B | freeform self-explanation LoRA (first-person target variant) |
| `qwen3_32b_selfexpl_structured_e3_kl0` | Qwen3-32B | 32B self-explanation LoRA |
| `qwen3_32b_selfexpl_synth_e1_kl1` | Qwen3-32B | 32B self-explanation LoRA (+synthetic data, KL) |
| `qwen3p5_397b_counterfactual_e1_kl0` | Qwen3.5-397B-A17B | 397B counterfactual-prediction LoRA (trained/served via Tinker for the paper) |
| `qwen3p5_397b_selfexpl_e3_kl0` | Qwen3.5-397B-A17B | 397B freeform self-explanation LoRA (Tinker) |

## Privileged-access self/cross pairs (size-matched 24k, appendix)

| HF repo | Trained on data about | Base |
|---|---|---|
| `qwen3_8b_on_qwen3_8b_counterfactual_24k_e1_kl0` | Qwen3-8B | Qwen3-8B |
| `llama3_1_8b_on_qwen3_8b_counterfactual_24k_e1_kl0` | Qwen3-8B | Llama-3.1-8B |
| `llama3_1_8b_on_llama3_1_8b_counterfactual_24k_e1_kl0` | Llama-3.1-8B | Llama-3.1-8B |
| `qwen3_8b_on_llama3_1_8b_counterfactual_24k_e1_kl0` | Llama-3.1-8B | Qwen3-8B |

The paper's figure uses these four pairs; `experiments/privileged_access/jobs/eval_traditional.sh`
downloads and scores them.

## Interpretability-tool checkpoints used by the eval

- **Activation oracles** (pre-existing releases, referenced by the eval scripts):
  - Gemma-3-27B: `adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it`
  - Qwen3-8B (single-layer): `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`
  - Qwen3-8B (multi-layer): `adamkarvonen/checkpoints_500k_pl_31k_spqav2_199k_sqav3_50k_hb_126k_cls`
- **NLA / SAE**: the predictor agents read precomputed verbalization / feature caches
  (`nla_cache/`, `sae_cache/` next to each claim bank; not part of the HF dataset). The NLA
  models themselves are not part of this release.
