#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=selfexpl_8b
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --qos=high
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out

# Precompute the Qwen3-8B self-explanation cache (vLLM offline + the self-expl LoRA).
# Args: $1 = eval set (bank) path.  Optional: $2 = --max-prompts N (smoke test).
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root
source scripts/gpu_preflight.sh

export HF_HOME=/workspace-vast/$USER/hf_cache

BANK="$1"; shift || true
SE=experiments/self_explanation
RES=${CHIVE_DATA:-data}/experiments/self_explanation/results

# Trained 8B self-explanation cache (the `selfexpl` arm). --out keeps it in the experiment's
# results/<slug>/cache/ (where the kept cache already lives) so a re-run reproduces it in place.
# Released adapter — download first (the --checkpoint-name label is kept for cache-file continuity):
#   hf download adamkarvonen/qwen3_8b_selfexpl_structured_e3_kl0 --local-dir checkpoints/qwen3_8b_selfexpl_structured_e3_kl0
python $SE/precompute_selfexpl_cache.py \
    --bank "$BANK" \
    --base-model Qwen/Qwen3-8B \
    --checkpoint-name mu_qwen3_8b_50k_structured_answer_e3_kl0_filtered \
    --lora-path checkpoints/qwen3_8b_selfexpl_structured_e3_kl0 \
    --out $RES/qwen8b/cache/mu_qwen3_8b_50k_structured_answer_e3_kl0_filtered.jsonl \
    "$@"
