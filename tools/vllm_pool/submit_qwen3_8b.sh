#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=mu_pool_qwen8b
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=tools/vllm_pool/logs/%x_%A_%a.out

# Single-GPU Qwen/Qwen3-8B (bf16) vLLM server that self-registers into a Qwen3-8B-
# specific pool dir so a dedicated pool_proxy.py can affinity-route across it.
# Backs the opus_investigates_qwen3_8b_5000_thorough run (stage 1 completions +
# stage 3 counterfactual run_prompt against the target).
#
# Differences from submit_qwen3.6_35b.sh:
#   - MODEL is Qwen/Qwen3-8B (dense, bf16) — fits trivially on one H200, no FP8.
#   - Dropped MoE/hybrid perf flags (--gdn-prefill-backend, flashinfer-autotune).
#   - NO tool-call parser. Everything we run against this server uses TOOLS-AS-TEXT
#     (the MU pipeline puts tool defs in the system prompt, no structured tools= in
#     requests; the WildChat run has no tools; PETRI audits default to
#     --target-tools none). So tool calls come back as plain <tool_call>{...} TEXT
#     and nothing needs to parse them. This is the one clean config for all uses.
#
#     >>> ONLY if you want TOOL-USE petri audits (--target-tools synthetic, for
#     >>> richer agentic scenarios) does the target server need a parser. The
#     >>> parser must match the model's tool-call FORMAT:
#     >>>   Qwen3-8B (dense, this script): add
#     >>>       --enable-auto-tool-choice --tool-call-parser hermes
#     >>>   Qwen3.5-397B-A17B (MoE, submit_qwen3.5_397b.sh): uses
#     >>>       --enable-auto-tool-choice --tool-call-parser qwen3_coder
#     >>> Do NOT enable a parser for the investigation/WildChat: on the dense 8B
#     >>> the hermes parser errors on truncated tool-call JSON (max_tokens cutoff)
#     >>> -> empty completions -> Stage 1's >2% empty assert crashes the run.
#     >>> (tested 2026-06-25; see experiments/dataset_build/README.md).
#   - --max-model-len 8192: covers every prompt the MU pipeline keeps — build_shards
#     drops PETRI prompts >4096 tokens, and WildChat prompts are short (≤~1500 tok).
#     Long PETRI *audit* conversations may 400 on their deepest turns (>~7k tokens),
#     but those deep-turn prefixes would be dropped at the 4096 build_shards cap
#     anyway, so raising the window buys nothing downstream.
#   - reasoning-parser/-config kept so the server is robust, but generation is
#     ALWAYS no-think: the pipeline sends chat_template_kwargs={enable_thinking:
#     False} (thinking_budget=None) for both stage 1 and stage 3, matching how AO
#     #1 (Qwen3-8B) was trained.
#   - REG_DIR is vllm_pool_qwen8b so this pool stays isolated from other proxies.
#
# Submit (2-GPU pool):
#   sbatch --qos=high --array=0-1 tools/vllm_pool/submit_qwen3_8b.sh

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate

export HF_HOME=/workspace-vast/$USER/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source scripts/gpu_preflight.sh

MODEL="Qwen/Qwen3-8B"
REG_DIR="/workspace-vast/$USER/vllm_pool_qwen8b"
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
