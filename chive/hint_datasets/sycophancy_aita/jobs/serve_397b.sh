#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Launch the Qwen3.5-397B-A17B-FP8 vLLM server used to GENERATE the 397B sycophancy data.
# Thin wrapper around tools/vllm_pool/submit_qwen3.5_397b.sh (TP=4; full crash-prevention flag
# set lives there); we only override walltime + job name. The server self-registers into the
# pool registry dir — read the connect URL from the registry JSON (see tools/vllm_pool/README.md).
# Weight load off NFS takes ~30-60 min. Cancel with: scancel -u $USER --name=syc397b_serve
set -euo pipefail
cd "$(dirname "$0")/../../../.."
EXCLUDE=$(./jobs/find_zombie_gpu_processes.sh --exclude || true)
sbatch $EXCLUDE --time="${1:-08:00:00}" -J syc397b_serve \
    tools/vllm_pool/submit_qwen3.5_397b.sh
