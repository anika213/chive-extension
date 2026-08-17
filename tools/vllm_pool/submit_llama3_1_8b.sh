#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=mu_pool_llama8b
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=tools/vllm_pool/logs/%x_%A_%a.out

# Single-GPU meta-llama/Llama-3.1-8B-Instruct (bf16) vLLM server that self-registers
# into a Llama-8B-specific pool dir so a dedicated pool_proxy.py can affinity-route
# across it. Backs the opus_investigates_llama3_1_8b_5000_thorough_n30 run (stage 1
# completions + stage 3 counterfactual run_prompt against the target).
#
# Adapted from submit_qwen3_8b.sh; differences:
#   - MODEL is Llama-3.1-8B-Instruct. No reasoning parser: Llama has no thinking
#     protocol (the qwen3 parser would misparse; the pipeline's enable_thinking
#     kwarg is accepted harmlessly by the Llama chat template — established).
#   - Same NO tool-call parser rationale as the qwen script: everything we run
#     against this server uses tools-as-text.
#   - --max-model-len 8192: covers every prompt the MU pipeline keeps.
#
# Submit (2-GPU pool):
#   sbatch --qos=high --array=0-1 tools/vllm_pool/submit_llama3_1_8b.sh

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate

export HF_HOME=/workspace-vast/$USER/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source scripts/gpu_preflight.sh

MODEL="meta-llama/Llama-3.1-8B-Instruct"
REG_DIR="/workspace-vast/$USER/vllm_pool_llama8b"
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
    --gpu-memory-utilization 0.90
    --max-model-len 8192
    --disable-custom-all-reduce
)

# config_hash is over output-semantic flags only so the proxy can quarantine a
# drifted member that would silently mix a different model / context window in.
HASH_FLAGS=(
    --served-model-name "$MODEL"
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
