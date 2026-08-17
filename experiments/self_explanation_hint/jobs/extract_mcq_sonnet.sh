#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Method-B stage 2 (MCQ): Sonnet 5-state attribution extraction over each MCQ self-explanation cache.
# Waits (up to ~90 min) for each cache to be written by stage 1, then extracts (resumable). Run inline/
# background (no GPU). Grader = Sonnet (claude-sonnet-4-6). Sycophancy was handled by the one-off rescore.
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

for name in \
    mcq_explain_qwen3_8b_base mcq_explain_qwen3_8b_trained \
    mcq_explain_qwen35_397b_base mcq_explain_qwen35_397b_trained ; do
    for _ in $(seq 1 180); do
        [ -s "$C/$name.jsonl" ] && break
        sleep 30
    done
    if [ ! -s "$C/$name.jsonl" ]; then
        echo "[extract_mcq] TIMEOUT waiting for $name.jsonl — skipping"; continue
    fi
    echo "[extract_mcq] extracting $name with Sonnet"
    python $SE/extract_attribution.py \
        --cache "$C/$name.jsonl" --out "$X/$name.jsonl" \
        --judge-model claude-sonnet-4-6 --concurrency 60
done
echo "[extract_mcq] DONE"
