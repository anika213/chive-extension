#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Score the 7 arms of the traditional privileged-access test (the paper figure),
# one cell per array task. Per target bank: {self, cross} trained predictors + base arms.
# (The 8th arm — Qwen3-8B base on the Qwen bank — reuses the canonical
# direct_counterfactual_evaluation scoring; the Opus reference comes from each run's
# canonical transcript-only scoring. See plot_traditional.arm_path.)
#
# The four trained adapters are released on HF — download first:
#   hf download adamkarvonen/qwen3_8b_on_qwen3_8b_counterfactual_24k_e1_kl0 --local-dir checkpoints/qwen3_8b_on_qwen3_8b_counterfactual_24k_e1_kl0
#   hf download adamkarvonen/llama3_1_8b_on_qwen3_8b_counterfactual_24k_e1_kl0 --local-dir checkpoints/llama3_1_8b_on_qwen3_8b_counterfactual_24k_e1_kl0
#   hf download adamkarvonen/llama3_1_8b_on_llama3_1_8b_counterfactual_24k_e1_kl0 --local-dir checkpoints/llama3_1_8b_on_llama3_1_8b_counterfactual_24k_e1_kl0
#   hf download adamkarvonen/qwen3_8b_on_llama3_1_8b_counterfactual_24k_e1_kl0 --local-dir checkpoints/qwen3_8b_on_llama3_1_8b_counterfactual_24k_e1_kl0
# (or retrain them: see README.md / jobs/train_cross_predictor.sh). Submit:
#   sbatch experiments/privileged_access/jobs/eval_traditional.sh
#SBATCH --job-name=eval_privileged_traditional
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/adamk/exp/logs/%x_%A_%a.out

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root
source scripts/gpu_preflight.sh

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

EVAL=experiments/direct_counterfactual_evaluation/run_structured_claim_eval.py
RES=${CHIVE_DATA:-data}/experiments/privileged_access/results
RUNS=${CHIVE_DATA:-data}/runs
QWEN_BANK=$RUNS/opus_investigates_qwen3_8b_5000_thorough_n30/eval_set_conc3_n2000.jsonl
LLAMA_BANK=$RUNS/opus_investigates_llama3_1_8b_5000_thorough_n30/eval_set_conc3_n2000.jsonl
QWEN=Qwen/Qwen3-8B
LLAMA=meta-llama/Llama-3.1-8B-Instruct
CKPT=checkpoints

# index: bank | arm-key | base model | lora ("-" = base, no adapter)
CELLS=(
  "qwen_regular|llama31_8b_base|$LLAMA|-"
  "qwen_regular|qwen3_8b_on_qwen3_8b_sub24k|$QWEN|$CKPT/qwen3_8b_on_qwen3_8b_counterfactual_24k_e1_kl0"
  "qwen_regular|llama31_8b_on_qwen3_8b_sub24k|$LLAMA|$CKPT/llama3_1_8b_on_qwen3_8b_counterfactual_24k_e1_kl0"
  "llama_regular|qwen3_8b_base|$QWEN|-"
  "llama_regular|llama31_8b_base|$LLAMA|-"
  "llama_regular|llama31_8b_on_llama31_8b|$LLAMA|$CKPT/llama3_1_8b_on_llama3_1_8b_counterfactual_24k_e1_kl0"
  "llama_regular|qwen3_8b_on_llama31_8b|$QWEN|$CKPT/qwen3_8b_on_llama3_1_8b_counterfactual_24k_e1_kl0"
)
IFS='|' read -r BANKKEY ARM BASE_MODEL LORA <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
case "$BANKKEY" in
  qwen_regular)  BANKFILE=$QWEN_BANK ;;
  llama_regular) BANKFILE=$LLAMA_BANK ;;
esac

mkdir -p $RES/scored
OUT=$RES/scored/${BANKKEY}__${ARM}.jsonl
echo "=== cell $SLURM_ARRAY_TASK_ID: bank=$BANKKEY arm=$ARM model=$BASE_MODEL lora=$LORA -> $OUT ==="

if [ "$LORA" = "-" ]; then
  srun python $EVAL --backend vllm --base-model "$BASE_MODEL" \
      --eval-set "$BANKFILE" --out "$OUT"
else
  test -f "$LORA/adapter_config.json" || { echo "missing adapter at $LORA"; exit 1; }
  srun python $EVAL --backend vllm --base-model "$BASE_MODEL" --lora-path "$LORA" \
      --eval-set "$BANKFILE" --out "$OUT"
fi
echo "=== DONE cell $SLURM_ARRAY_TASK_ID ($BANKKEY $ARM) ==="
