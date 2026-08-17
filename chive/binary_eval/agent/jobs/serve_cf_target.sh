#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=cf_target
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --qos=high
#SBATCH --output=/workspace-vast/adamk/chive/chive/binary_eval/agent/logs/%x_%j.out

# Plain vLLM OpenAI server for the binary-eval counterfactual arm (--tool cf). The
# run_counterfactual tool hits this server to run the live target model on agent-specified
# modified conversations. No vllm-lens / capture here — just generation. Single H200, tp=1
# (an 8B fits comfortably; CLAUDE.md: never use tensor parallelism for inference).
#
# Usage:  sbatch jobs/serve_cf_target.sh <MODEL> <PORT|""> <CONF_JSON>
#   e.g.  sbatch .../serve_cf_target.sh Qwen/Qwen3-8B "" \
#             data/scratch/cf_server.json
# PORT optional: pass "" to auto-pick a free port. Writes {"url","base_model","slurm_job"}
# to CONF_JSON on startup — run_binary_eval reads url/base_model from there by default.

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
source scripts/gpu_preflight.sh

export HF_HOME=/workspace-vast/adamk/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL="$1"
PORT="${2:-}"
CONF_JSON="$3"
# Optional 4th arg: extra vLLM flags (word-split). To make the target THINKING-capable (so the
# cf re-runs can reproduce a reasoning-mode behavior), pass: --reasoning-parser qwen3 --reasoning-config {}
# Leave empty for the default no-think target (e.g. gemma, or no-think Qwen runs).
EXTRA_FLAGS="${4:-}"

pick_free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'; }

# Port hygiene (vLLM binds with SO_REUSEPORT): two servers given the SAME port on the SAME
# node silently share it (kernel load-balances → ~half the requests hit the wrong server).
# Auto-pick a free port if none given, or if the requested port already answers on THIS node.
if [ -z "$PORT" ]; then
  PORT=$(pick_free_port)
  echo "[serve] no port given — auto-picked free port ${PORT}"
elif curl -sf -m 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  NEW=$(pick_free_port)
  echo "[serve] WARNING: port ${PORT} already serving on $(hostname) — switching to ${NEW}"
  PORT=$NEW
fi

URL="http://$(hostname):${PORT}"
mkdir -p "$(dirname "$CONF_JSON")"
printf '{"url": "%s", "base_model": "%s", "slurm_job": "%s"}\n' \
  "${URL}" "${MODEL}" "${SLURM_JOB_ID}" > "$CONF_JSON"
echo "[serve] node=$(hostname) port=${PORT} job=${SLURM_JOB_ID} model=${MODEL} -> ${CONF_JSON}"

# max-model-len 8192: the full multi-turn transcript (prompt) + response tokens, with headroom
# (the source pipeline ran these same conversations at 6000; a thinking run needs prompt + 2048).
# gpu-mem-util 0.90 per CLAUDE.md. Sampling (temp/max_tokens/thinking) is set client-side by
# run_endpoint_prompt; EXTRA_FLAGS only makes the server *capable* of thinking.
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  $EXTRA_FLAGS
