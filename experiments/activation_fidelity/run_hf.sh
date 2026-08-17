#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=actfid_hf
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --qos=high
#SBATCH --output=experiments/activation_fidelity/logs/%x_%j.out

# Activation-fidelity check — HF in-process capture (THE FAST PATH). One H200 job:
# loads Qwen3-8B + both LoRA adapters (kl0, kl1) once, runs batched forwards with hooks (activations
# stay in-process, never serialized out of an engine subprocess), computes per-region cosine /
# relative-L2 / raw-MSE / EXACT full-vocab KL, writes results/ + the plot. No vLLM, no HTTP.
#
# Usage (from repo root):  sbatch [--exclude=<bad nodes>] run_hf.sh [N_TRANSCRIPTS]
#   rogue-GPU two-step first:
#     EXCLUDE=$(./jobs/find_zombie_gpu_processes.sh --exclude || true); sbatch $EXCLUDE run_hf.sh
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
source scripts/gpu_preflight.sh

export HF_HOME=/workspace-vast/adamk/hf_cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

N="${1:-1000}"
HERE=experiments/activation_fidelity
RUN=${CHIVE_DATA:-data}/runs/opus_investigates_qwen3_8b_5000_thorough_n30
BANK=$RUN/eval_set_conc3_n2000.jsonl
OUT=${CHIVE_DATA:-data}/experiments/activation_fidelity/results
mkdir -p "$OUT"

echo "[run_hf] capturing fidelity over up to $N transcripts (HF in-process)"
python "$HERE/collect_hf.py" --bank "$BANK" --out-dir "$OUT" --n-transcripts "$N"

echo "[run_hf] plotting"
python "$HERE/make_plot.py" --summary "$OUT/fidelity_summary.json" --out "$OUT/fidelity.png"

echo "[run_hf] done -> $OUT/fidelity_summary.json , $OUT/fidelity.png"
