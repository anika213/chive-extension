#!/bin/bash
# Paper TODO #6 — canonical base vs trained on the direct counterfactual-prediction eval, for both
# the 8B and 397B models, on both the regular (5000) and petri eval sets. 8 runs total, all on the
# SAME prompt: format A (claim as a real follow-up turn on the transcript), no-think, first-token
# Yes/No logprob — base and trained differ only in weights.
#
#   8B  (vLLM)  : 4 runs kicked off as PARALLEL sbatch jobs (1 GPU each).
#   397B (Tinker): 4 runs SEQUENTIAL in this driver (CPU/API only, concurrency 100).
#
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Both arms go through run_structured_claim_eval.py (base = omit the
# adapter flag; tinker base support added in this repo). After the 8B jobs finish, analyze.py writes
# SUMMARY.md. Run from repo root with .venv active and .env sourced. CPU/API driver — no GPU here.
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root

EVAL=experiments/direct_counterfactual_evaluation/run_structured_claim_eval.py
HERE=experiments/direct_counterfactual_evaluation
OUT=${CHIVE_DATA:-data}/experiments/direct_counterfactual_evaluation/results/canonical
RUNS=${CHIVE_DATA:-data}/runs
mkdir -p "$OUT"

# --- eval sets (regular = 5000_thorough, petri = petri_thorough) ---
Q8_REG=$RUNS/opus_investigates_qwen3_8b_5000_thorough_n30/eval_set_conc3_n2000.jsonl
Q8_PET=$RUNS/opus_investigates_qwen3_8b_petri_thorough_n30/eval_set_conc3_n2000.jsonl
Q397_REG=$RUNS/opus_investigates_qwen3p5_397b_5000_thorough_n30/eval_set_conc3_n2000.jsonl
Q397_PET=$RUNS/opus_investigates_qwen3p5_397b_petri_thorough_n30/eval_set_conc3_n2000.jsonl

# --- trained checkpoints (canonical, cf5 e1) ---
# Released 8B adapter — download first:
#   hf download adamkarvonen/qwen3_8b_counterfactual_e1_kl0 --local-dir checkpoints/qwen3_8b_counterfactual_e1_kl0
LORA8=checkpoints/qwen3_8b_counterfactual_e1_kl0
# Tinker sampler weights used for the paper; adapter weights released at adamkarvonen/qwen3p5_397b_counterfactual_e1_kl0 (HF).
TINK397=tinker://7c443149-82fa-52af-94c6-e635f39da1d2:train:0/sampler_weights/final

# ---- 8B: 4 parallel sbatch jobs (vLLM, 1 GPU each) ----
EXCLUDE=$(./jobs/find_zombie_gpu_processes.sh --exclude || true)
J=dcf8b_$USER
echo "[run_all] submitting 4 parallel 8B vLLM jobs ($EXCLUDE)"
sbatch $EXCLUDE -J ${J}_base_reg "$HERE/jobs/eval_8b_vllm.sh" "$Q8_REG" "$OUT/8b_base_regular.jsonl"    base
sbatch $EXCLUDE -J ${J}_base_pet "$HERE/jobs/eval_8b_vllm.sh" "$Q8_PET" "$OUT/8b_base_petri.jsonl"      base
sbatch $EXCLUDE -J ${J}_lora_reg "$HERE/jobs/eval_8b_vllm.sh" "$Q8_REG" "$OUT/8b_trained_regular.jsonl" "$LORA8"
sbatch $EXCLUDE -J ${J}_lora_pet "$HERE/jobs/eval_8b_vllm.sh" "$Q8_PET" "$OUT/8b_trained_petri.jsonl"   "$LORA8"

# ---- 397B: 4 sequential Tinker runs (CPU/API, concurrency 100) ----
echo "[run_all] 397B base — regular"
python "$EVAL" --backend tinker --eval-set "$Q397_REG" --out "$OUT/397b_base_regular.jsonl"  --concurrency 100
echo "[run_all] 397B base — petri"
python "$EVAL" --backend tinker --eval-set "$Q397_PET" --out "$OUT/397b_base_petri.jsonl"    --concurrency 100
echo "[run_all] 397B trained — regular"
python "$EVAL" --backend tinker --eval-set "$Q397_REG" --model-path "$TINK397" --out "$OUT/397b_trained_regular.jsonl" --concurrency 100
echo "[run_all] 397B trained — petri"
python "$EVAL" --backend tinker --eval-set "$Q397_PET" --model-path "$TINK397" --out "$OUT/397b_trained_petri.jsonl"   --concurrency 100

# ---- wait for the 8B jobs, then analyze ----
echo "[run_all] waiting for 8B jobs (${J}_*) to finish..."
while squeue -u "$USER" -h -n ${J}_base_reg,${J}_base_pet,${J}_lora_reg,${J}_lora_pet -o %i | grep -q .; do
    sleep 30
done
echo "[run_all] all 8B jobs done; analyzing"
# analyze.py reads $OUT itself (results/canonical under CHIVE_DATA) and takes no results arg.
python "$HERE/analyze.py"
echo "[run_all] done -> $HERE/SUMMARY.md"
