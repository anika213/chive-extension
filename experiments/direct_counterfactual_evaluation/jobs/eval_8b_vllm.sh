#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=dcf_eval_8b
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/adamk/exp/logs/%x_%j.out

# One canonical 8B direct-counterfactual eval (format A, no-think, first-token Yes/No) via
# run_structured_claim_eval.py — base and trained share this exact
# prompt (the only difference is whether --lora-path is passed).
#   $1 = eval-set jsonl   $2 = output jsonl   $3 = "base" | <lora adapter dir>
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root
source scripts/gpu_preflight.sh

EVAL_SET="$1"; OUT="$2"; LORA="$3"
EVAL=experiments/direct_counterfactual_evaluation/run_structured_claim_eval.py

echo "=== 8B eval: set=$EVAL_SET out=$OUT lora=$LORA ==="
if [ "$LORA" = "base" ]; then
    python "$EVAL" --backend vllm --eval-set "$EVAL_SET" --out "$OUT"
else
    python "$EVAL" --backend vllm --eval-set "$EVAL_SET" --out "$OUT" --lora-path "$LORA"
fi
echo "=== DONE ($OUT) ==="
