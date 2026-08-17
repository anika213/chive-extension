#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=mu_pool_qwen32b
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=tools/vllm_pool/logs/%x_%A_%a.out

# Single-GPU Qwen/Qwen3-32B-FP8 vLLM server that self-registers into a Qwen3-32B-
# specific pool dir so a dedicated pool_proxy.py can affinity-route across it.
# Backs the opus_investigates_qwen3_32b_5000_thorough_n30 test-data run (stage 1
# completions + stage 3 counterfactual run_prompt against the target).
#
# Adapted from submit_qwen3_8b.sh:
#   - MODEL is Qwen/Qwen3-32B-FP8 (pre-quantized FP8 checkpoint per AGENTS.md;
#     fits on one H200, ~85GB left for KV at 0.9 util). NO tensor parallelism.
#   - NO tool-call parser: the MU pipeline uses TOOLS-AS-TEXT (tool defs in the
#     system prompt), so tool calls come back as plain <tool_call> text. Enabling
#     the hermes parser errors on truncated tool-call JSON -> empty completions
#     -> Stage 1's >2% empty assert crashes the run.
#   - --max-model-len 8192: WildChat prompts are short (<= ~1500 tok) and stage 3
#     counterfactuals are similar; stage 2/4 judges run on the Anthropic API.
#   - reasoning-parser/-config kept so the server is robust, but generation is
#     ALWAYS no-think: the pipeline sends chat_template_kwargs={enable_thinking:
#     False} (thinking_budget=None) for both stage 1 and stage 3 targets.
#   - REG_DIR is vllm_pool_qwen32b so this pool stays isolated from other proxies.
#
# Submit (4-GPU pool):
#   sbatch --qos=high --array=0-3 tools/vllm_pool/submit_qwen3_32b.sh

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate

export HF_HOME=/workspace-vast/$USER/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source scripts/gpu_preflight.sh

MODEL="Qwen/Qwen3-32B-FP8"
REG_DIR="/workspace-vast/$USER/vllm_pool_qwen32b"
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
    --enable-prefix-caching
    --reasoning-parser qwen3
    --reasoning-config '{}'
    --gpu-memory-utilization 0.90
    --max-model-len 8192
    --disable-custom-all-reduce
)

# config_hash is over output-semantic flags only so the proxy can quarantine a
# drifted member that would silently mix a different model / context window in.
HASH_FLAGS=(
    --served-model-name "$MODEL"
    --reasoning-parser qwen3
    --reasoning-config '{}'
    --max-model-len 8192
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
