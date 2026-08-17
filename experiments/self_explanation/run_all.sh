#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# End-to-end driver for the self-explanation counterfactual-simulatability experiment.
#
# Ordering matters: the `selfexpl_base` arm loads the base-model self-explanation caches at
# startup, so the base precompute MUST finish before any simulator runs. We submit the base
# precompute first and gate everything on it (vLLM sims via sbatch --dependency; the Anthropic
# sims wait inline). The trained `selfexpl` caches already exist in results/<slug>/cache/
# (under data/experiments/self_explanation/).
set -euo pipefail
cd /workspace-vast/adamk/chive
SE=experiments/self_explanation

# 1. Base-model self-explanations (8B vLLM + 397B Tinker) over the full eval bank.
BASE_JID=$(sbatch --parsable -J selfexpl_base $SE/jobs/precompute_base_selfexpl.sh)
echo "[run_all] base precompute job=$BASE_JID"

# 2. vLLM simulators — one job each, gated on the base precompute. Each does all 3 runs x 3 arms.
for m in google/gemma-3-12b-it google/gemma-3-27b-it Qwen/Qwen3.6-27B; do
    tag=$(echo "$m" | tr '/.' '__')
    sbatch --parsable --dependency=afterok:$BASE_JID -J "selfexpl_sim_$tag" \
        $SE/jobs/run_sim_vllm.sh "$m"
done

# 3. Anthropic simulators (no GPU) — wait for the base caches, then run inline in parallel.
echo "[run_all] waiting for base precompute ($BASE_JID) before Anthropic sims..."
while squeue -h -j "$BASE_JID" -u "$USER" >/dev/null 2>&1 && [ -n "$(squeue -h -j "$BASE_JID" 2>/dev/null)" ]; do
    sleep 60
done
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root
python $SE/run_simulator.py --backend anthropic --simulator claude-haiku-4-5-20251001 --concurrency 50 &
python $SE/run_simulator.py --backend anthropic --simulator claude-opus-4-8           --concurrency 50 &
wait

# 4. Analysis + figure (after all sims finish — run manually once the vLLM jobs complete too).
python $SE/analyze.py
