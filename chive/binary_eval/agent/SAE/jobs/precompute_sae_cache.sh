#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=sae_precompute
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --qos=high
#SBATCH --output=/workspace-vast/adamk/chive/chive/binary_eval/agent/logs/%x_%j.out

# Precompute the Gemma Scope 2 SAE cache (capture via the running AO server over HTTP;
# JumpReLU encode on this job's GPU). See precompute_sae_cache.py docstring.
#
# Output: <bank_dir>/sae_cache/<sae_slug>.jsonl
#
# Usage:
#   sbatch ... precompute_sae_cache.sh <bank.jsonl> <capture_url> <base_model> [extra precompute args...]
# e.g. verification:
#   sbatch ... precompute_sae_cache.sh <bank> http://node-12:8414 google/gemma-3-27b-it --verify-layers 39,40,41 --max-questions 3
# cache build:
#   sbatch ... precompute_sae_cache.sh <bank> http://node-12:8414 google/gemma-3-27b-it --capture-layer 40

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
source scripts/gpu_preflight.sh

export HF_HOME=/workspace-vast/adamk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

BANK=$1
CAPTURE_URL=$2
BASE_MODEL=$3
shift 3

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

srun python -m chive.binary_eval.agent.SAE.precompute_sae_cache \
    --bank "$BANK" --capture-url "$CAPTURE_URL" --base-model "$BASE_MODEL" "$@"
