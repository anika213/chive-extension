#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Train one predictor LoRA for the traditional privileged-access test. All four predictors
# share hyperparameters (matched to qwen3_8b_binary_counterfactual_e1_kl0); only
# model_name/data/paths differ per config. Run make_sub24k_data.py first (the two
# *_on_qwen3_8b_sub24k configs train on its output).
# Arg $1 = config basename under training_configs/model_understanding/ (no dir, no .json),
# one of: qwen3_8b_on_qwen3_8b_sub24k_binary_counterfactual_e1_kl0,
#         llama31_8b_on_qwen3_8b_sub24k_binary_counterfactual_e1_kl0,
#         llama31_8b_on_llama31_8b_binary_counterfactual_e1_kl0,
#         qwen3_8b_on_llama31_8b_binary_counterfactual_e1_kl0.
# Submit with a matching -J:
#   sbatch -J qwen3_8b_on_qwen3_8b_sub24k_binary_counterfactual_e1_kl0 \
#     experiments/privileged_access/jobs/train_cross_predictor.sh \
#     qwen3_8b_on_qwen3_8b_sub24k_binary_counterfactual_e1_kl0
#SBATCH --job-name=cross_predictor_train
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --output=/workspace-vast/adamk/exp/logs/%x_%j.out

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root
source scripts/gpu_preflight.sh
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

CFG_NAME="$1"
CFG=${CHIVE_DATA:-data}/training_configs/model_understanding/$CFG_NAME.json

echo "=== Phase 1: prepare dataset (.pt) for $CFG_NAME ==="
python -m chive.training.model_understanding --config "$CFG" --prepare-only

echo "=== Phase 2: train (1 GPU, global batch 8, 1 epoch) — $CFG_NAME ==="
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
torchrun --nproc_per_node=1 --master-port="$MASTER_PORT" \
    -m chive.training.model_understanding --config "$CFG"

echo "=== DONE ($CFG_NAME; checkpoint at checkpoints_text_sft/$CFG_NAME/final) ==="  # (24k variants released at adamkarvonen/*_counterfactual_24k_e1_kl0)
