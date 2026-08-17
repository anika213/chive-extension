#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=mu_pool_gemma
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=tools/vllm_pool/logs/%x_%A_%a.out

# Single-GPU google/gemma-3-27b-it vLLM server that self-registers into a Gemma-
# specific pool dir so a dedicated pool_proxy.py can route across it.
#
# Gemma 3 is multimodal: --language-model-only strips the vision tower so the
# server starts as a pure text LM (smaller mem footprint, faster startup).
#
# Differences from submit_qwen3.6_35b.sh (Qwen):
#   - MODEL is google/gemma-3-27b-it (instruct).
#   - No --tool-call-parser / --reasoning-parser / --reasoning-config: Gemma does
#     not use the Qwen tool/reasoning protocols, and the target model in this
#     run never makes tool calls (Opus is the agent; Gemma just emits text).
#   - --max-model-len 4000: opus-investigates runs only need short context
#     (prompts ≤ ~1500 tokens, max_tokens=500). No self-investigation here.
#   - REG_DIR is vllm_pool_gemma so this pool stays isolated from the prod Qwen
#     proxy on 8999 / dev proxy on 9000.
#
# Submit (3-GPU pool for the opus_investigates_gemma3_27b_it run):
#   sbatch --qos=high --array=0-2 tools/vllm_pool/submit_gemma3_27b.sh

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate

export HF_HOME=/workspace-vast/$USER/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source scripts/gpu_preflight.sh

MODEL="google/gemma-3-27b-it"
REG_DIR="/workspace-vast/$USER/vllm_pool_gemma"
mkdir -p "$REG_DIR"
REG_FILE="$REG_DIR/${SLURM_JOB_ID}.json"

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

COMMON_FLAGS=(
    --served-model-name "$MODEL"
    --language-model-only
    --enable-prefix-caching
    --gpu-memory-utilization 0.90
    --max-model-len 4000
    --disable-custom-all-reduce
)

# config_hash captures output-semantic flags only so the proxy can quarantine a
# member that would silently mix a different model / context window into the
# pool. Perf knobs are not hashed.
HASH_FLAGS=(
    --served-model-name "$MODEL"
    --language-model-only
    --max-model-len 4000
)
CONFIG_HASH=$(printf '%s\n' "$MODEL" "${HASH_FLAGS[@]}" | sha256sum | cut -c1-16)

echo "Starting pool server: model=$MODEL host=$HOST port=$PORT qos=$QOS config=$CONFIG_HASH job=$SLURM_JOB_ID"

vllm serve "$MODEL" --port "$PORT" "${COMMON_FLAGS[@]}" &
VLLM_PID=$!

until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    sleep 5
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM exited during startup"; exit 1; }
done

printf '{"url": "http://%s:%s/v1", "model": "%s", "config_hash": "%s", "node": "%s", "port": %s, "qos": "%s", "job_id": "%s"}\n' \
    "$HOST" "$PORT" "$MODEL" "$CONFIG_HASH" "$HOST" "$PORT" "$QOS" "$SLURM_JOB_ID" > "$REG_FILE"
echo "Registered $REG_FILE -> http://$HOST:$PORT/v1"

wait "$VLLM_PID"
