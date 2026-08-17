#!/bin/bash
# Score the SAE tool arm on the reasoning-mode (thinking) Qwen3-8B eval set, for the CoT
# appendix figure (app:cot). Pure cache-lookup tool: NO GPU, NO server — Anthropic-agent +
# the precomputed sae_cache. CPU-only Slurm job (compute nodes reach the Anthropic API, as
# the AO/cf arms do). Idempotent: --resume. Writes label=conc3 into the run's binary_evals/
# so experiments/thinking_interp_eval/make_figure.py auto-discovers the arm.
#SBATCH --job-name=sae_think_eval
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/adamk/chive/chive/binary_eval/agent/logs/%x_%j.out

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate

export HF_HOME=/workspace-vast/adamk/hf_cache
export ANTHROPIC_API_KEY_HIGH_PRIORITY="${ANTHROPIC_API_KEY_HIGH_PRIORITY:-$ANTHROPIC_API_KEY}"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

T=data/runs/opus_investigates_qwen3_8b_think1024_5000_thorough_n30

srun python -m chive.binary_eval.run_binary_eval \
  --bank "$T/eval_set_conc3_n2000.jsonl" \
  --answerer agent --tool sae --base-model Qwen/Qwen3-8B \
  --label conc3 --concurrency 80 --resume \
  --out-dir "$T/binary_evals"
