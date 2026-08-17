#!/bin/bash
set -euo pipefail

RUNPOD_REPO="${RUNPOD_REPO:-/root/chive}"
cd "${RUNPOD_REPO}"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required in ${RUNPOD_REPO}/.env or the environment" >&2
    exit 1
fi

VLLM_ENV="${VLLM_ENV:-/root/vllm_envs/qwen36}"
if [[ ! -x "${VLLM_ENV}/bin/vllm" ]]; then
    echo "Missing vLLM env at ${VLLM_ENV}; run tools/vllm_pool/runpod/setup_runpod_vllm_env.sh" >&2
    exit 127
fi
source "${VLLM_ENV}/bin/activate"

export HF_HOME="${HF_HOME:-/root/hf_cache}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
PORT="${PORT:-8000}"

COMMON_FLAGS=(
    --served-model-name "$MODEL"
    --language-model-only
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --reasoning-config '{}'
    --gpu-memory-utilization 0.90
    --max-model-len 81920
    --gdn-prefill-backend triton
)
CONFIG_HASH=$(printf '%s\n' "$MODEL" "${COMMON_FLAGS[@]}" | sha256sum | cut -c1-16)

mkdir -p /root/vllm_logs

echo "$(date) Starting RunPod vLLM server"
echo "  host=$(hostname)"
echo "  model=${MODEL}"
echo "  port=${PORT}"
echo "  config_hash=${CONFIG_HASH}"
echo "  HF_HOME=${HF_HOME}"
echo "  vllm=$(python -c 'import vllm; print(vllm.__version__)')"
echo "  torch=$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"

exec vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    "${COMMON_FLAGS[@]}"
