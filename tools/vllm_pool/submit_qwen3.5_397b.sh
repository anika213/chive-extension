#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=mu_pool_397b
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=5-00:00:00
#SBATCH --output=tools/vllm_pool/logs/%x_%A_%a.out

# 4-GPU Qwen/Qwen3.5-397B-A17B-FP8 vLLM server (pure TP=4, no PP/DP/EP) that
# self-registers into the prod 397B pool dir so pool_proxy.py can affinity-route
# across however many are alive.
#
# Flags include `--reasoning-config '{}'` which is REQUIRED for thinking_token_budget
# requests not to 400. Without it the server starts fine and serves /v1/models,
# but every Stage 3 agent request fails silently. See SERVER_TUNING.md for details.
#
# Tool-call parser: this script keeps `--enable-auto-tool-choice --tool-call-parser
# qwen3_coder` (qwen3_coder = the MoE/Coder tool-call format that Qwen3.5-397B emits).
# It is REQUIRED for TOOL-USE petri audits (--target-tools synthetic, structured
# tools=) and is harmless/inert for everything else (the MU investigation + WildChat
# + no-tools audits send no tools=, so the parser never runs and tool calls pass
# through as text). NOTE the per-model rule: the dense Qwen3-8B uses `hermes`, NOT
# qwen3_coder (different tool-call format) — see submit_qwen3_8b.sh. PETRI audits now
# default to --target-tools none, so a parser is only needed if you opt back into tools.
#
# config_hash with this flag set: 3cb5a3806115a6d0 (matches the patched RunPod 397B servers).
#
# QoS / array / job-name are overridden per submission on the sbatch CLI:
#   sbatch --qos=high --job-name=mu_pool_397b_hi --array=0-1 submit_qwen3.5_397b.sh
#   sbatch --qos=low  --job-name=mu_pool_397b_lo --array=0-3 submit_qwen3.5_397b.sh

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env

# vLLM multiprocessing + cluster-required NCCL knobs.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# DeepGEMM JIT fails in this env (cu129 torch + CUDA 12.8 toolkit) → kills EngineCore.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
# FlashInfer FP8 MoE path produced garbage output for Qwen3.5 FP8 on H200 — defensive.
export VLLM_USE_FLASHINFER_MOE_FP8=0
# Caches off node-local disk.
export HF_HOME=/workspace-vast/$USER/hf_cache

# Fail fast if assigned GPU isn't actually free.
export GPU_MIN_FREE_MIB=130000
source scripts/gpu_preflight.sh

MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
# Qwen3.5-397B needs vLLM >= 0.21, but the repo venv is pinned to vllm==0.19.1 (the
# vllm-lens AO-capture dependency), so this launcher CANNOT use the repo venv. Build a
# dedicated serving venv and point VLLM_ENV at it before submitting, e.g.:
#   uv venv /path/to/venv-vllm-qwen35 && uv pip install -p /path/to/venv-vllm-qwen35 'vllm>=0.21'
if [ -z "${VLLM_ENV:-}" ] || [ ! -f "${VLLM_ENV}/bin/activate" ]; then
  echo "ERROR: set VLLM_ENV to a venv with vllm>=0.21 (Qwen3.5 support); the repo venv's vllm==0.19.1 cannot serve this model." >&2
  exit 1
fi
REG_DIR="/workspace-vast/$USER/vllm_pool_397b"
mkdir -p "$REG_DIR"
REG_FILE="$REG_DIR/${SLURM_JOB_ID}.json"

# Pick a free ephemeral port: several servers can land on one node.
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
HOST=$(hostname)
QOS="${SLURM_JOB_QOS:-unknown}"

cleanup() {
    rm -f "$REG_FILE"
    [ -n "${VLLM_PID:-}" ] && kill -TERM "$VLLM_PID" 2>/dev/null
    kill -TERM -$$ 2>/dev/null
    wait
}
trap cleanup SIGTERM SIGINT SIGQUIT EXIT

# Flags shared by every 397B pool member (everything EXCEPT --port). Order and
# content must match the probe launcher exactly so manually-injected probe
# registry entries get the same config_hash.
COMMON_FLAGS=(
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
CONFIG_HASH=$(printf '%s\n' "$MODEL" "${COMMON_FLAGS[@]}" | sha256sum | cut -c1-16)

echo "Starting 397B pool server: model=$MODEL host=$HOST port=$PORT qos=$QOS config=$CONFIG_HASH job=$SLURM_JOB_ID"

source "${VLLM_ENV}/bin/activate"
vllm serve "$MODEL" --port "$PORT" "${COMMON_FLAGS[@]}" &
VLLM_PID=$!

# Register only once the server actually answers. 397B weight load is slow off
# NFS (~378GB checkpoint), so give it 3h to settle. Bail if vLLM dies during startup.
DEADLINE=$(( SECONDS + 180 * 60 ))
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited during startup" >&2
        exit 1
    fi
    if (( SECONDS > DEADLINE )); then
        echo "vLLM did not become healthy within 180 min" >&2
        exit 1
    fi
    sleep 10
done

printf '{"url": "http://%s:%s/v1", "model": "%s", "config_hash": "%s", "node": "%s", "port": %s, "qos": "%s", "job_id": "%s"}\n' \
    "$HOST" "$PORT" "$MODEL" "$CONFIG_HASH" "$HOST" "$PORT" "$QOS" "$SLURM_JOB_ID" > "$REG_FILE"
echo "Registered $REG_FILE -> http://$HOST:$PORT/v1"

wait "$VLLM_PID"
