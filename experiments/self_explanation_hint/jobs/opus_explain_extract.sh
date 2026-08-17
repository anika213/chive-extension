#!/bin/bash
# Opus reference arm: stage 1 (generate_opus_explanation.py, Opus external explainer) then
# stage 2 (extract_attribution.py, Sonnet judge — matches the base/trained arms) over the 4
# unified hint banks. API/CPU only (no GPU) — run inline/background. Both stages resumable.
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root
export HF_HOME=/workspace-vast/$USER/hf_cache

SE=experiments/self_explanation_hint
DATA=${CHIVE_DATA:-data}
C=$DATA/experiments/self_explanation_hint/results/cache
X=$DATA/experiments/self_explanation_hint/results/extracted

declare -A BANK=(
    [mcq_explain_qwen3_8b]="$DATA/hint_banks/cot_hint/mcq_explain/cf_eval_qwen3_8b.jsonl"
    [mcq_explain_qwen35_397b]="$DATA/hint_banks/cot_hint/mcq_explain/cf_eval_qwen35_397b.jsonl"
    [sycophancy_qwen3_8b]="$DATA/hint_banks/sycophancy_aita/Qwen3-8B/cf_eval_bank.jsonl"
    [sycophancy_qwen35_397b]="$DATA/hint_banks/sycophancy_aita/Qwen3.5-397B-A17B-FP8/cf_eval_bank.jsonl"
)

for name in \
    mcq_explain_qwen3_8b mcq_explain_qwen35_397b \
    sycophancy_qwen3_8b sycophancy_qwen35_397b ; do
    echo "=== opus explain: $name ==="
    python $SE/generate_opus_explanation.py --eval-set "${BANK[$name]}" \
        --out "$C/${name}_opus.jsonl" --concurrency 50
    echo "=== extract (Sonnet): $name ==="
    python $SE/extract_attribution.py --cache "$C/${name}_opus.jsonl" \
        --out "$X/${name}_opus.jsonl" --judge-model claude-sonnet-4-6 --concurrency 60
done
echo "OPUS ARM DONE"
