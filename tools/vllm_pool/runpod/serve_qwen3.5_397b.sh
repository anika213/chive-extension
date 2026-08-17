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
if [[ -z "${VLLM_API_KEY:-}" ]]; then
    echo "VLLM_API_KEY is required in ${RUNPOD_REPO}/.env or the environment" >&2
    exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set explicitly for each TP4 RunPod server" >&2
    exit 1
fi

VLLM_ENV="${VLLM_ENV:-/root/vllm_envs/qwen35}"
if [[ ! -x "${VLLM_ENV}/bin/vllm" ]]; then
    echo "Missing vLLM env at ${VLLM_ENV}; run setup_runpod_vllm_env.sh with VLLM_ENV=${VLLM_ENV}" >&2
    exit 127
fi
source "${VLLM_ENV}/bin/activate"

export HF_HOME="${HF_HOME:-/root/hf_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/workspace/vllm_cache}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-0}"

MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
PORT="${PORT:?PORT must be set, e.g. 8000 or 8001}"
SERVER_ID="${SERVER_ID:-runpod-${RUNPOD_POD_ID:-unknown}-${PORT}}"

SEMANTIC_FLAGS=(
    --host 0.0.0.0
    --served-model-name "$MODEL"
    --tensor-parallel-size 4
    --language-model-only
    --reasoning-parser qwen3
    --reasoning-config '{}'
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --gpu-memory-utilization 0.90
    --max-model-len 81920
    --max-num-seqs 1024
    --max-num-batched-tokens 8192
    --gdn-prefill-backend triton
    --no-enable-flashinfer-autotune
    --disable-custom-all-reduce
    --max-num-partial-prefills 1
    --max-long-partial-prefills 1
    --long-prefill-token-threshold 8192
)
PERF_FLAGS=(
    --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
)
CONFIG_HASH=$(printf '%s\n' "$MODEL" "${SEMANTIC_FLAGS[@]}" | sha256sum | cut -c1-16)

mkdir -p /root/vllm_logs

echo "$(date -u) Starting RunPod 397B TP4 vLLM server"
echo "  server_id=${SERVER_ID}"
echo "  host=$(hostname)"
echo "  model=${MODEL}"
echo "  port=${PORT}"
echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "  config_hash=${CONFIG_HASH}"
echo "  public_url=http://${RUNPOD_PUBLIC_IP:-unknown}:${RUNPOD_TCP_PORT_8000:-unknown}/v1 or :${RUNPOD_TCP_PORT_8001:-unknown}/v1"
echo "  HF_HOME=${HF_HOME}"
echo "  VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "  vllm=$(python -c 'import vllm; print(vllm.__version__)')"
echo "  torch=$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"

exec vllm serve "$MODEL" \
    --port "$PORT" \
    "${SEMANTIC_FLAGS[@]}" \
    "${PERF_FLAGS[@]}"
