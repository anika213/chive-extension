#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=syc397b_judge
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/adamk/chive/chive/hint_datasets/sycophancy_aita/logs/judge_resume_397b_%j.out

# CPU-only: (re-)judge the saved Qwen3.5-397B-A17B-FP8 raw_responses.json with Sonnet and write the
# final sycophancy_eval_dataset.json. Resumable (skips already-judged completions) + retry-hardened
# (SDK max_retries=8), so safe to re-run if Anthropic is overloaded (529). No generation, no GPU.
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate

python -m chive.hint_datasets.sycophancy_aita.generate_dataset \
    --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --resume-judge \
    --no-cot-only \
    --judge-concurrency 48
