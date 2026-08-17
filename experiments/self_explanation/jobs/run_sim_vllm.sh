#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=selfexpl_sim
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out

# vLLM-offline simulator for the self-explanation experiment. One job per local simulator;
# scores all 3 target runs x 3 arms (plain/selfexpl/selfexpl_base) internally, text PROBABILITY.
# Usage: sbatch -J selfexpl_sim_<tag> run_sim_vllm.sh <hf_model_id> [extra args...]
#   e.g. sbatch run_sim_vllm.sh google/gemma-3-12b-it
#        sbatch run_sim_vllm.sh Qwen/Qwen3.6-27B          # thinking on (auto-detected)
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source scripts/gpu_preflight.sh

MODEL="${1:?usage: run_sim_vllm.sh <hf_model_id> [extra args]}"
shift || true

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

SE=experiments/self_explanation
srun python $SE/run_simulator.py --backend vllm --simulator "$MODEL" "$@"
echo "[run_sim_vllm] DONE $MODEL"
