# SAE tool — Gemma Scope 2 precompute + Neuronpedia labels

The third interp tool for the binary eval (alongside AO and NLA): per-token **sparse
autoencoder feature labels**. Everything is precomputed; the agent-facing tool
(`../sae_tool.py`) is a pure CPU lookup at eval time.

## Components

| file | what it does |
|---|---|
| `gemma_scope_sae.py` | hand-rolled JumpReLU SAE (load from `google/gemma-scope-2-27b-it` safetensors, encode/decode/fvu). Deliberately not SAELens — one matmul + threshold, one code path for non-gemma SAEs later. |
| `fetch_neuronpedia_labels.py` | downloads all 65,536 autointerp labels for one Neuronpedia source from the public S3 dataset dump (the per-feature API is deprecated for bulk) → `data/sae_labels/<model>_<source>.jsonl`. CPU, ~2 min. |
| `precompute_sae_cache.py` | per investigation: capture the layer-40 residual stream over the chat-templated tokens via the **running AO vllm-lens server** (same HTTP path as NLA precompute), JumpReLU-encode all positions on the job's GPU, store top-60 `(feature, activation)` per position → `data/scratch/sae_cache/<stem>.jsonl`. |
| `jobs/precompute_sae_cache.sh` | 1-GPU sbatch wrapper. |
| `qc_sae_cache.py` | manual QC / spot-read of a precomputed cache (aggregate stats + untruncated random rows). Not part of the eval pipeline. |

## The SAE

`google/gemma-scope-2-27b-it` / `resid_post/layer_40_width_65k_l0_medium`:
JumpReLU, d_sae=65536, trained L0=60, hook `model.layers.40.output` — the SAE behind
Neuronpedia's `gemma-3-27b-it/40-gemmascope-2-res-65k` source (medium L0 is what
Neuronpedia loads for the res-65k sources).

**Layer convention (verified empirically, 2026-06-11):** vllm-lens
`output_residual_stream=[40]` == HF `model.layers.40.output` == the SAE's hook point.
Checked with `--verify-layers 39,40,41` on 3 transcripts: layer 40 gives median L0
≈ 60 and the lowest FVU on every transcript (41 is ~20-30% worse FVU). The precompute
asserts `--capture-layer == SAE hook layer`, plus per-question median-L0 / FVU sanity
bounds, so an off-by-one or wrong-model capture cannot pass silently.

## Sanity audit (2026-06-11, post-cache)

Every cache row stores its question's `median_l0` and `fvu` (computed from the full
dense activations at precompute time — the stored top-60 cannot reconstruct L0 later).
Cross-validation of the pipeline against Google's own shipped stats:

| check | result |
|---|---|
| median-of-medians L0 (n=1254) | 58.7 vs trained target 60; their eval corpus (`sum(feature_frequencies)`) = 50 |
| FVU distribution | mean 0.121, p10 0.077, p90 0.211, max 0.391 |
| per-feature firing freq vs `examples.safetensors` `feature_frequencies` | log-log r = 0.76 over 55k features (different corpora) |
| per-feature max act vs their corpus max | median ratio 0.67, p90 0.98 — rules out any scale/convention error (a missing √d would be ~73×) |
| FVU on plain encyclopedic prose (same pipeline, layer, code) | **0.51–0.56** — whether chat-formatted or not |

Interpretation: **~0.10 FVU is the in-distribution level for this -it SAE on
gemma-generated chat transcripts**; generic encyclopedic prose is far off its training
distribution. Per-position error is diffuse (top-5 positions ≈ 1.5% of total error),
worst on function words/punctuation — normal SAE behavior, no outlier artifact.
Note the per-question asserts catch gross miswiring only; an off-by-one capture layer
changes FVU by ~0.01 and is instead prevented by the `capture_layer == config hook
layer` assert + the one-time `--verify-layers` empirical check.

## Coverage choices

- **Every position is cached, BOS included** — content tokens AND chat-template tokens
  (`<start_of_turn>`, `<end_of_turn>`, role markers), which carry turn-level features.
  Same full-coverage invariant as NLA.
- BOS produces garbage SAE features (its L0 explodes; attention sink). It is cached like
  any other position — no special-casing in the data — and the query_sae tool prompt
  tells the agent to ignore position 0. QC stats (median L0 / FVU gates) exclude it.
- Cache stores top-60 (= trained L0) `(feature_index, activation)` pairs per position,
  zero-activation entries dropped. The tool displays **labels only, no activation
  values** (values confuse agents); rank order carries the signal.
- The label↔SAE pairing (Neuronpedia `40-gemmascope-2-res-65k` ↔ HF `l0_medium`) is
  pinned EMPIRICALLY, not by metadata: per-feature max activations in our cache
  correlate with Neuronpedia's `maxActApprox` (log-log r≈0.70, median ratio 0.85);
  a wrong L0 variant would give ~0 correlation.

## Runbook

```bash
# 1. labels (once per SAE source; CPU)
python -m chive.binary_eval.agent.SAE.fetch_neuronpedia_labels

# 2. cache (1-GPU job; needs the AO serve job running for capture)
sbatch chive/binary_eval/agent/SAE/jobs/precompute_sae_cache.sh \
    <run_dir>/binary_counterfactual_questions.jsonl http://node-XX:8414 \
    google/gemma-3-27b-it --capture-layer 40
# --bank accepts a full claim bank or an eval set (rows deduped by prompt_id); precompute
# on the FULL bank. Output: <bank_dir>/sae_cache/<sae_slug>.jsonl (next to the bank).

# 3. eval (CPU + API only)
.venv/bin/python -m chive.binary_eval.run_binary_eval \
    --bank <run_dir>/eval_set_conc3_n1000.jsonl --answerer agent --tool sae \
    --base-model google/gemma-3-27b-it
# --sae-cache defaults to the single cache in <bank_dir>/sae_cache/ (the eval set lives
# next to the bank, so the bank's cache is found automatically); --sae-labels is derived
# from the SAE recorded in the cache.
```
