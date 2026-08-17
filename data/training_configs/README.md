# Training configs

Configs for `chive/training/model_understanding.py` (local GPUs) and
`chive/training/model_understanding_tinker.py` (Tinker API, used for the 397B runs).

## Released LoRA → config

The top-level configs are the ones behind the paper's runs and the released
adapters (see `docs/models_and_checkpoints.md` for the HF repos):

| Released adapter | Config |
|---|---|
| `qwen3_8b_counterfactual_e1_kl0` | `model_understanding/qwen3_8b_binary_counterfactual_e1_kl0.json` |
| `qwen3_8b_counterfactual_e1_kl1` | `model_understanding/qwen3_8b_binary_counterfactual_e1_kl1.json` |
| `qwen3_8b_selfexpl_structured_e3_kl0` | `model_understanding/qwen3_8b_50k_structured_answer_e3_kl0_filtered.json` |
| `qwen3_32b_selfexpl_structured_e3_kl0` | `model_understanding/qwen3_32b_structured_e3_kl0.json` |
| `qwen3_32b_selfexpl_synth_e1_kl1` | `model_understanding/qwen3_32b_structured_synth_e1_kl1.json` |
| `qwen3p5_397b_counterfactual_e1_kl0` | `tinker/qwen3p5_397b_binary_counterfactual_e1.json` |
| `qwen3p5_397b_selfexpl_e3_kl0` | `tinker/mu_qwen35_397b_inv_e3_vs7_bs16.json` |
| privileged-access self/cross pairs (4) | `model_understanding/{qwen3_8b_on_qwen3_8b_sub24k,llama31_8b_on_qwen3_8b_sub24k,llama31_8b_on_llama31_8b,qwen3_8b_on_llama31_8b}_binary_counterfactual_e1_kl0.json` |

Tinker configs take `project_id` / `owner` from the `TINKER_PROJECT_ID` /
`TINKER_OWNER` env vars (or fill in the config fields directly).

## `archive/` subdirs

`model_understanding/archive/` and `tinker/archive/` hold the configs from
exploratory and superseded runs (hypothesis-source ablations, synthetic-data
mixes, smoke tests, earlier model targets). They are kept as provenance only:
many reference internal data paths that are not part of the public data
release, so they will not run as-is.
