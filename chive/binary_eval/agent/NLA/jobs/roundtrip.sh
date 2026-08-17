#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=nla_roundtrip
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --qos=high
#SBATCH --output=/workspace-vast/adamk/chive/chive/binary_eval/agent/NLA/logs/%x_%j.out

# NLA round-trip validator: AV decode (offline) + AR critic, capturing from a
# stood-up vllm-lens server. Waits for the capture server's CONF_JSON + readiness,
# then runs the shared harness (nla_roundtrip.py) for one spec from the NLASpec
# registry in nla_inference.py.
#
# Extra args after CONF_JSON pass straight through to the python CLI
# (e.g. --capture-mode fineweb, or --capture-mode parquet --parquet-path ...).
#
# Usage:  sbatch jobs/roundtrip.sh <SPEC> <CONF_JSON> [extra nla_roundtrip args...]
#   e.g.  sbatch .../roundtrip.sh qwen2.5-7b-L20            .../results/capture_qwen25.json
#         sbatch .../roundtrip.sh qwen3-8b-baseline-nosalvage .../results/capture_qwen3.json \
#                --capture-mode fineweb

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
source scripts/gpu_preflight.sh

export HF_HOME=/workspace-vast/adamk/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SPEC="$1"
CONF_JSON="$2"
shift 2

# Wait for the capture server to publish its URL and become ready (up to ~15 min).
echo "[roundtrip] waiting for capture server conf ${CONF_JSON} ..."
for i in $(seq 1 90); do
  [ -f "$CONF_JSON" ] && break
  sleep 10
done
URL=$(python -c "import json,sys; print(json.load(open('$CONF_JSON'))['url'])")
echo "[roundtrip] capture url=${URL}; waiting for /v1/models ..."
for i in $(seq 1 90); do
  curl -sf "${URL}/v1/models" >/dev/null 2>&1 && break
  sleep 10
done
curl -sf "${URL}/v1/models" >/dev/null 2>&1 || { echo "[roundtrip] server not ready"; exit 1; }

echo "[roundtrip] spec=${SPEC} url=${URL} extra args: $*"
python -m chive.binary_eval.agent.NLA.nla_roundtrip \
  --spec "${SPEC}" --capture-url "${URL}" "$@"
