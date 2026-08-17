#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Generic vLLM server job. Submit FROM THE REPO ROOT.
#
# Usage:
#   sbatch chive/pipeline/slurm_scripts/submit_vllm_server.sh [MODEL] [PORT] [DP] [extra vllm args...]
#
# The #SBATCH defaults below are for a 1-GPU server. For data-parallel replicas,
# override resources on the sbatch command line (Slurm CLI flags beat the
# in-file directives) and pass the matching DP:
#   # Qwen3-8B, 2 replicas:
#   sbatch --gres=gpu:2 --cpus-per-task=16 --mem=128G \
#       chive/pipeline/slurm_scripts/submit_vllm_server.sh Qwen/Qwen3-8B 8000 2
#   # Qwen3-32B FP8, 4 replicas:
#   sbatch --gres=gpu:4 --cpus-per-task=32 --mem=256G \
#       chive/pipeline/slurm_scripts/submit_vllm_server.sh Qwen/Qwen3-32B 8000 4 --quantization fp8
# (gemma-3-27b-it has its own script: submit_vllm_server_gemma3_27b_dp2.sh —
#  it needs --language-model-only and NCCL env tweaks.)
#SBATCH --job-name=vllm_server
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --qos=high
#SBATCH --output=chive/pipeline/logs/slurm_vllm_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .env
export HF_HOME="${HF_HOME:-/workspace-vast/$USER/hf_cache}"
source scripts/gpu_preflight.sh

MODEL="${1:-Qwen/Qwen3-14B}"
PORT="${2:-8000}"
DP="${3:-1}"

echo "Starting vLLM server: model=${MODEL} port=${PORT} dp=${DP} node=$(hostname)"
echo "Connect with: --vllm-url http://$(hostname):${PORT}"

.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --data-parallel-size "$DP" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 6000 \
    "${@:4}"
