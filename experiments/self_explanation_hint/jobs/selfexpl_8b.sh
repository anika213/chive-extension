#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=cot_se_8b
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/adamk/chive/experiments/self_explanation_hint/logs/%x_%j.out

# Method-B stage 1 (8B): generate self-explanation cache for one cot_hint MCQ arm via vLLM.
# Reuses the shared generic generate_selfexpl.py (reads row["freeform_question"] -> runs on any
# unified Claim bank). Args: $1=eval-set(bank)  $2=out cache  $3="base" | <8B self-expl LoRA dir>
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root
source scripts/gpu_preflight.sh
export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

SE=experiments/self_explanation_hint
EVAL="$1"; OUT="$2"; SPEC="$3"

if [ "$SPEC" = "base" ]; then
    srun python $SE/generate_selfexpl.py --arm base --backend vllm \
        --eval-set "$EVAL" --base-model Qwen/Qwen3-8B --checkpoint-name base_Qwen3-8B \
        --out "$OUT" --max-new-tokens 4096
else
    srun python $SE/generate_selfexpl.py --arm trained --backend vllm \
        --eval-set "$EVAL" --base-model Qwen/Qwen3-8B \
        --checkpoint-name mu_qwen3_8b_50k_structured_answer_e3_kl0_filtered \
        --lora-path "$SPEC" --out "$OUT" --max-new-tokens 2048
fi
echo "[selfexpl_8b] DONE $OUT"
