#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=vllm_gemma27_dp2
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/adamk/chive/chive/pipeline/logs/slurm_vllm_gemma27_dp2_%j.out

# google/gemma-3-27b-it vLLM server, data-parallel=2 (2 replicas on 2 H200s behind
# one port). Separate from the generic submit_vllm_server.sh because Gemma needs
# the flags/env below.
#   - Gemma 3 is multimodal: --language-model-only strips the vision tower so the
#     server starts as a pure text LM.
#   - No tool/reasoning parser: Gemma doesn't use the Qwen protocols and the
#     target model never makes tool calls (Opus is the agent; Gemma emits text).
#   - --max-model-len 4000: opus-investigates needs only short context
#     (prompts <= ~1500 tokens, max_tokens=500).
#   - NO tensor parallelism; each replica fits on one H200.
# Usage: sbatch <this script> [PORT]
#   (pass the printed --exclude=... from jobs/find_zombie_gpu_processes.sh before it)

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate

export HF_HOME=/workspace-vast/$USER/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source scripts/gpu_preflight.sh

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

MODEL="google/gemma-3-27b-it"
PORT="${1:-8233}"

echo "Starting vLLM server: model=${MODEL} port=${PORT} dp=2 node=$(hostname)"
echo "Connect with spec: vllm:${MODEL}@http://$(hostname):${PORT}"

vllm serve "${MODEL}" \
    --port "${PORT}" \
    --served-model-name "${MODEL}" \
    --data-parallel-size 2 \
    --language-model-only \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4000 \
    --disable-custom-all-reduce
