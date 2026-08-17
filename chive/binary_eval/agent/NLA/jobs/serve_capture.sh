#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=nla_capture
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --qos=high
#SBATCH --output=/workspace-vast/adamk/chive/chive/binary_eval/agent/NLA/logs/%x_%j.out

# Generic vllm-lens BASE CAPTURE server (the vllm-lens plugin auto-registers, so the
# OpenAI HTTP endpoint carries activations out). Clients capture a layer's residual
# stream via POST /v1/completions with vllm_xargs={"output_residual_stream":"[L]"}
# (JSON-encoded list). Used by the NLA round-trip validator and the cache precompute.
#
# Usage:  sbatch jobs/serve_capture.sh <MODEL> <PORT|""> <CONF_JSON>
#   e.g.  sbatch .../serve_capture.sh Qwen/Qwen3-8B "" .../results/capture_qwen3.json   # auto-port (recommended)
#         sbatch .../serve_capture.sh google/gemma-3-27b-it 8431 .../results/capture_gemma.json
# PORT is optional: pass "" to auto-pick a free port. If a non-empty PORT already answers
# on the assigned node it is auto-switched (SO_REUSEPORT collision avoidance, see below).
# Writes {"url","base_model","slurm_job"} to CONF_JSON on startup. ALWAYS read the actual
# url/port from CONF_JSON — do not assume the port you passed (it may have been switched).

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

pick_free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'; }

# Port hygiene (vLLM binds with SO_REUSEPORT): two servers given the SAME port that land
# on the SAME node SILENTLY share it — the kernel load-balances connections between them,
# so ~half the requests hit the wrong server/model (404 / cross-model). This bit us once
# (a gemma + a qwen capture server both on :8422, same node). Guard against it: auto-pick
# a free port if none is given, or if the requested port ALREADY answers on THIS node
# (another server is already there). Clients must read the ACTUAL url from CONF_JSON.
if [ -z "$PORT" ]; then
  PORT=$(pick_free_port)
  echo "[serve] no port given — auto-picked free port ${PORT}"
elif curl -sf -m 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
  NEW=$(pick_free_port)
  echo "[serve] WARNING: port ${PORT} already serving on $(hostname) — switching to ${NEW} (SO_REUSEPORT collision avoidance)"
  PORT=$NEW
fi

URL="http://$(hostname):${PORT}"
mkdir -p "$(dirname "$CONF_JSON")"
printf '{"url": "%s", "base_model": "%s", "slurm_job": "%s"}\n' \
  "${URL}" "${MODEL}" "${SLURM_JOB_ID}" > "$CONF_JSON"
echo "[serve] node=$(hostname) port=${PORT} job=${SLURM_JOB_ID} model=${MODEL}"
echo "[serve] wrote ${CONF_JSON} -> ${URL}"

vllm serve "${MODEL}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port "${PORT}"
