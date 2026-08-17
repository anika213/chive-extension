#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=syc397b_gen_client
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/adamk/chive/chive/hint_datasets/sycophancy_aita/logs/gen_397b_%j.out

# CPU-only client: generate the Qwen3.5-397B-A17B-FP8 sycophancy AITA dataset by hitting the RUNNING
# 397B vLLM server (started by jobs/serve_397b.sh). Pure async OpenAI-client generation + Sonnet
# judging. Writes data/hint_banks/sycophancy_aita/Qwen3.5-397B-A17B-FP8/{raw_responses,
# sycophancy_eval_dataset}.json. no_cot only (the direct-counterfactual eval is no-think).
#
#   sbatch jobs/gen_397b.sh <SERVER_V1_URL> [NUM_POSTS]
#   e.g.  sbatch jobs/gen_397b.sh http://node-18:8000/v1 1200
#
# Posts are the seed-42 metaeval/scruples[train] shuffle, so NUM_POSTS=1200 selects the SAME first
# 1200 posts as the 8B run -> clean paired 8B-vs-397B comparison.
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate

SERVER_URL="${1:?need server /v1 URL, e.g. http://node-18:8000/v1}"
NUM_POSTS="${2:-1200}"

python -m chive.hint_datasets.sycophancy_aita.generate_dataset \
    --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --server-url "${SERVER_URL}" \
    --num-posts "${NUM_POSTS}" \
    --no-cot-only \
    --gen-concurrency 64 \
    --judge-concurrency 48
