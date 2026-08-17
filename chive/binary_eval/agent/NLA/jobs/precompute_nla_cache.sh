#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=nla_precompute
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=8:00:00
#SBATCH --qos=high
#SBATCH --output=/workspace-vast/adamk/chive/chive/binary_eval/agent/NLA/logs/%x_%j.out

# Single-job NLA cache precompute (small banks / smoke tests; use precompute_nla_array.sh
# for full banks). Loads the AV model OFFLINE (vllm.LLM, async_scheduling=False) and
# decodes every addressable token's verbalization; captures activations from the AO
# vllm-lens server over HTTP. async_scheduling=False is REQUIRED on vLLM < 0.22 — async
# scheduling corrupts prompt_embeds decodes into `<<<<` loops (root-caused; see
# precompute_nla_cache.py header + investigations/nla_debug/README.md).
#
# Output: <bank_dir>/nla_cache/<spec.name>.jsonl
#
# Usage:
#   sbatch jobs/precompute_nla_cache.sh <bank.jsonl> <capture_url> <spec> [extra flags...]
# e.g.
#   sbatch chive/binary_eval/agent/NLA/jobs/precompute_nla_cache.sh \
#     <run_dir>/binary_counterfactual_questions.jsonl http://node-1:8414 \
#     qwen3-8b-baseline-nosalvage --max-questions 50

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
source scripts/gpu_preflight.sh

export HF_HOME=/workspace-vast/adamk/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

BANK="$1"
CAPTURE_URL="$2"
SPEC="$3"
shift 3

python -m chive.binary_eval.agent.NLA.precompute_nla_cache \
  --bank "${BANK}" \
  --capture-url "${CAPTURE_URL}" \
  --spec "${SPEC}" \
  "$@"
