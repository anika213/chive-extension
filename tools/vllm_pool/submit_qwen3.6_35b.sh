#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=mu_pool
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=tools/vllm_pool/logs/%x_%A_%a.out

# Single-GPU Qwen3.6-35B-A3B-FP8 vLLM server that self-registers into the prod
# pool dir so pool_proxy.py can affinity-route across however many are alive.
#
# QoS / array / job-name are overridden per submission on the sbatch CLI, so this
# one script backs BOTH the high-prio core and the low-prio scavengers:
#   sbatch --qos=high --job-name=mu_pool_hi --array=0-7  tools/vllm_pool/submit_qwen3.6_35b.sh
#   sbatch --qos=low  --job-name=mu_pool_lo --array=0-11 tools/vllm_pool/submit_qwen3.6_35b.sh
#
# Every pool member is byte-identical on the flags below (same model / quant /
# max-model-len / reasoning) so the proxy's config_hash guard can detect a
# drifted member; mixing configs behind one endpoint would silently blend
# different behavior into one dataset. For a separate dev pool that won't
# receive prod traffic, use submit_qwen3.6_35b_dev.sh (same flags, different
# registry dir).

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate

# Caches / NCCL per cluster policy (keep caches off node-local disk).
export HF_HOME=/workspace-vast/$USER/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Fail fast if the assigned GPU isn't actually free.
source scripts/gpu_preflight.sh

MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
REG_DIR="/workspace-vast/$USER/vllm_pool"
mkdir -p "$REG_DIR"
REG_FILE="$REG_DIR/${SLURM_JOB_ID}.json"

# Pick a free ephemeral port: several single-GPU servers can land on one node, so
# a fixed port would collide.
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
HOST=$(hostname)
QOS="${SLURM_JOB_QOS:-unknown}"

# Drop our registry entry and tear down the vLLM process group on exit/preempt so
# the proxy stops routing to us promptly. The proxy's health check is the backstop
# if a SIGKILL beats this trap.
# Remove our registry entry FIRST (so the proxy stops routing immediately), then
# kill the vLLM pid explicitly AND signal the process group (vLLM spawns worker
# children). Health check is the backstop if a SIGKILL beats this trap.
cleanup() {
    rm -f "$REG_FILE"
    [ -n "${VLLM_PID:-}" ] && kill -TERM "$VLLM_PID" 2>/dev/null
    kill -TERM -$$ 2>/dev/null
    wait
}
trap cleanup SIGTERM SIGINT SIGQUIT EXIT

# Flags shared by every pool member (everything EXCEPT --port, which is per-server).
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
    --no-enable-flashinfer-autotune
    --disable-custom-all-reduce
)

# config_hash is over the subset of flags that change OUTPUT semantics (so the
# proxy can quarantine a member that would silently mix a different model /
# parser / context window into the pool). Perf / reliability knobs
# (prefix-caching, gpu-memory-utilization, gdn-prefill-backend, ...) are NOT
# hashed so adding/changing those doesn't force a full pool rotation.
HASH_FLAGS=(
    --served-model-name "$MODEL"
    --language-model-only
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --reasoning-config '{}'
    --max-model-len 81920
)
CONFIG_HASH=$(printf '%s\n' "$MODEL" "${HASH_FLAGS[@]}" | sha256sum | cut -c1-16)

echo "Starting pool server: model=$MODEL host=$HOST port=$PORT qos=$QOS config=$CONFIG_HASH job=$SLURM_JOB_ID"

vllm serve "$MODEL" --port "$PORT" "${COMMON_FLAGS[@]}" &
VLLM_PID=$!

# Register only once the server actually answers, so the proxy never routes to a
# still-loading member. Bail if vLLM dies during startup.
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    sleep 5
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM exited during startup"; exit 1; }
done

printf '{"url": "http://%s:%s/v1", "model": "%s", "config_hash": "%s", "node": "%s", "port": %s, "qos": "%s", "job_id": "%s"}\n' \
    "$HOST" "$PORT" "$MODEL" "$CONFIG_HASH" "$HOST" "$PORT" "$QOS" "$SLURM_JOB_ID" > "$REG_FILE"
echo "Registered $REG_FILE -> http://$HOST:$PORT/v1"

wait "$VLLM_PID"
