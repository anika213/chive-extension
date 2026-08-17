#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=selfexpl_base
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out

# Base-model (un-fine-tuned) self-explanation caches for the `selfexpl_base` negative control.
# Zero-shot, no LoRA, same self-explanation question over the full eval bank (the analysis pairs
# base vs trained on their common claim set).
#   - 8B   : vLLM offline (this GPU).
#   - 397B : Tinker (API, no GPU) — runs on this node after the vLLM part; finish_reason
#            unavailable from Tinker, so 397B-base truncation can't be measured (budget set high).
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source scripts/gpu_preflight.sh

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

SE=experiments/self_explanation
DATA=${CHIVE_DATA:-data}
RES=$DATA/experiments/self_explanation/results
RUNS=$DATA/runs
MNT=2048   # = the trained-explanation length the simulators ran with 0 skips, so even a base
           # generation that rambles to this cap still fits the simulator context and is kept
           # (no dropping). >99% of base explanations finish well under it.

# --- 8B base (vLLM offline, GPU) ------------------------------------------------------------
srun python $SE/precompute_selfexpl_cache.py --base --backend vllm \
    --bank $RUNS/opus_investigates_qwen3_8b_5000_thorough_n30/eval_set_conc3_n2000.jsonl \
    --base-model Qwen/Qwen3-8B --checkpoint-name base_Qwen3-8B \
    --out $RES/qwen8b/cache/base_Qwen3-8B.jsonl --max-new-tokens $MNT

# --- 397B base (Tinker, API; q397_5000 + q397_petri) ----------------------------------------
for run in q397_5000:opus_investigates_qwen3p5_397b_5000_thorough_n30 \
           q397_petri:opus_investigates_qwen3p5_397b_petri_thorough_n30; do
    slug="${run%%:*}"; dir="${run##*:}"
    srun python $SE/precompute_selfexpl_cache.py --base --backend tinker \
        --bank $RUNS/$dir/eval_set_conc3_n2000.jsonl \
        --base-model Qwen/Qwen3.5-397B-A17B --checkpoint-name base_Qwen3.5-397B-A17B \
        --out $RES/$slug/cache/base_Qwen3.5-397B-A17B.jsonl \
        --concurrency 30 --max-new-tokens $MNT
done
echo "[precompute_base_selfexpl] DONE"
