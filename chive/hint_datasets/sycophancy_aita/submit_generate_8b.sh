#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Generate AITA sycophancy dataset for Qwen3-8B.
# Usage: sbatch chive/hint_datasets/sycophancy_aita/submit_generate_8b.sh

#SBATCH --job-name=gen_syc_aita_8b
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /workspace-vast/adamk/chive
source .venv/bin/activate
source .env

echo "=== Generating AITA sycophancy dataset for Qwen3-8B ==="
# --num-posts 3000 --no-cot-only matches the shipped 8B bank (the 200-post default
# is a smoke-test size and does not clear the >=150-per-cell balance assert).
python -m chive.hint_datasets.sycophancy_aita.generate_dataset --model Qwen/Qwen3-8B \
    --num-posts 3000 --no-cot-only

echo "=== Done ==="
