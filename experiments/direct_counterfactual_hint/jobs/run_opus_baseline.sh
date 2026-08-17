#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Opus transcript-only (`plain`) method-A baseline over the 4 unified hint banks. API/CPU only (no GPU).
# The banks are BinaryClaim supersets, so run_binary_eval loads them directly (no adapter step).
# Resumable (skips claim_ids already in the out file). Lands results/scored/<setting>_<model>_opus.jsonl
# (same BinaryResult schema as the base/trained arms).
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root
export HF_HOME=/workspace-vast/$USER/hf_cache

DATA=${CHIVE_DATA:-data}
CH=$DATA/hint_banks/cot_hint
SYC=$DATA/hint_banks/sycophancy_aita
S=$DATA/experiments/direct_counterfactual_hint/results/scored

declare -A BANK=(
    [mcq_explain_qwen3_8b]="$CH/mcq_explain/cf_eval_qwen3_8b.jsonl"
    [mcq_explain_qwen35_397b]="$CH/mcq_explain/cf_eval_qwen35_397b.jsonl"
    [sycophancy_qwen3_8b]="$SYC/Qwen3-8B/cf_eval_bank.jsonl"
    [sycophancy_qwen35_397b]="$SYC/Qwen3.5-397B-A17B-FP8/cf_eval_bank.jsonl"
)

for name in \
    mcq_explain_qwen3_8b mcq_explain_qwen35_397b \
    sycophancy_qwen3_8b sycophancy_qwen35_397b ; do
    echo "=== Opus plain (high): $name ==="
    python -m chive.binary_eval.run_binary_eval --answerer plain --bank "${BANK[$name]}" \
        --answerer-model claude-opus-4-8 --reasoning-effort high \
        --out "$S/${name}_opus.jsonl" --resume --concurrency 100
    echo "-> $S/${name}_opus.jsonl"
done
echo "ALL OPUS DONE"
