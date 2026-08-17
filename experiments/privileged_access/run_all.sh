#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# End-to-end reproduction of the traditional privileged-access test (the paper figure).
# See README.md for the full design. Stages:
#   0. CPU preflight (tokenizer/render/length checks; informational — read the log)
#   1. Download the 4 released trained adapters from HF (skipped if already present).
#      To retrain them from scratch instead, see "Retraining the predictors" in README.md.
#   2. Score the 7 eval arms (sbatch, 1 GPU per array task)
#   3. Figure + stdout table (CPU, run manually once the array finishes)
set -euo pipefail
cd /workspace-vast/adamk/chive
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root

PA=experiments/privileged_access
RES=${CHIVE_DATA:-data}/experiments/privileged_access/results

echo "=== Stage 0: preflight checks (CPU) ==="
mkdir -p $RES/scored
python $PA/stage0_checks.py | tee $RES/stage0_checks.log

echo "=== Stage 1: released adapters ==="
for ADAPTER in qwen3_8b_on_qwen3_8b_counterfactual_24k_e1_kl0 \
               llama3_1_8b_on_qwen3_8b_counterfactual_24k_e1_kl0 \
               llama3_1_8b_on_llama3_1_8b_counterfactual_24k_e1_kl0 \
               qwen3_8b_on_llama3_1_8b_counterfactual_24k_e1_kl0; do
    if [ ! -f "checkpoints/$ADAPTER/adapter_config.json" ]; then
        hf download "adamkarvonen/$ADAPTER" --local-dir "checkpoints/$ADAPTER"
    fi
done

echo "=== Stage 2: eval arms (sbatch, GPU array) ==="
EVAL_JOB=$(sbatch --parsable $PA/jobs/eval_traditional.sh)
echo "eval array job: $EVAL_JOB"

echo "=== Stage 3: figure (run manually once the array finishes) ==="
echo "  python $PA/plot_traditional.py"
echo "(waits on Slurm job $EVAL_JOB; check: squeue -u \$USER)"
