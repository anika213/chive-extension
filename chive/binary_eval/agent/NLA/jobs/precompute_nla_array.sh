#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=nla_pc_array
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=6:00:00
#SBATCH --qos=high
#SBATCH --array=0-4
#SBATCH --output=/workspace-vast/adamk/chive/chive/binary_eval/agent/NLA/logs/%x_%A_%a.out

# Parallel NLA-cache precompute: one array task per shard, sharded INSIDE the precompute
# (--num-shards/--shard-idx — no shard input files). Each task loads its OWN offline AV
# engine (vllm.LLM, async_scheduling=False — REQUIRED on vLLM < 0.22 to avoid the
# prompt_embeds `<<<<` corruption; see precompute_nla_cache.py header) on its own GPU
# and decodes its shard, capturing activations from the ONE shared AO vllm-lens server
# over HTTP.
#
# Shard outputs land in <bank_dir>/nla_cache/shards/<av>_shard{i}of{N}.jsonl. When ALL
# tasks finish, combine into the cache the eval's default lookup finds:
#   cat <bank_dir>/nla_cache/shards/<av>_shard*of5.jsonl > <bank_dir>/nla_cache/<av>.jsonl
# then verify line counts and load it once via nla_tool.load_nla_cache (asserts
# full coverage / no duplicates / degeneration rate).
#
# Usage (NUM_SHARDS must match #SBATCH --array above):
#   sbatch jobs/precompute_nla_array.sh <bank.jsonl> <capture_url> <spec>
# e.g.
#   sbatch chive/binary_eval/agent/NLA/jobs/precompute_nla_array.sh \
#     data/runs/opus_investigates_qwen3_8b_5000_thorough_n30/eval_set_conc3_n1000.jsonl \
#     http://node-3:8422 qwen3-8b-baseline-nosalvage

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
NUM_SHARDS=5   # must match #SBATCH --array=0-4 above

echo "[array task ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}] node=$(hostname) bank=${BANK} capture=${CAPTURE_URL} spec=${SPEC}"

python -m chive.binary_eval.agent.NLA.precompute_nla_cache \
  --bank "${BANK}" \
  --capture-url "${CAPTURE_URL}" \
  --spec "${SPEC}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-idx "${SLURM_ARRAY_TASK_ID}"
