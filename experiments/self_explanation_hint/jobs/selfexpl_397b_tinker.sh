#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Method-B stage 1 (397B): generate self-explanation caches for the cot_hint MCQ bank via Tinker
# (API, no GPU — run inline/background). base (zero-shot) + trained (self-expl LoRA), for
# mcq_explain. Sycophancy 397B is handled separately (reused + Sonnet-rescored).
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.  # resolve experiments.* / chive from the repo root
export HF_HOME=/workspace-vast/$USER/hf_cache

SE=experiments/self_explanation_hint
DATA=${CHIVE_DATA:-data}
CACHE=$DATA/experiments/self_explanation_hint/results/cache
BASE=Qwen/Qwen3.5-397B-A17B
# Tinker sampler weights used for the paper; the adapter weights are released at
# adamkarvonen/qwen3p5_397b_selfexpl_e3_kl0 (HF).
TINK=tinker://75e43819-7e1e-533a-b811-636a8f325e53:train:0/sampler_weights/final
EXP=$DATA/hint_banks/cot_hint/mcq_explain/cf_eval_qwen35_397b.jsonl

run_arm() {  # $1=eval-set  $2=out  $3=base|trained
    local eval="$1" out="$2" arm="$3"
    if [ "$arm" = "base" ]; then
        python $SE/generate_selfexpl.py --arm base --backend tinker \
            --eval-set "$eval" --base-model $BASE --checkpoint-name base_Qwen3.5-397B-A17B \
            --out "$out" --max-new-tokens 4096 --concurrency 30
    else
        python $SE/generate_selfexpl.py --arm trained --backend tinker \
            --eval-set "$eval" --base-model $BASE --checkpoint-name mu_qwen35_397b_selfexpl \
            --tinker-weights $TINK --out "$out" --max-new-tokens 2048 --concurrency 30
    fi
}

run_arm "$EXP" "$CACHE/mcq_explain_qwen35_397b_base.jsonl"      base
run_arm "$EXP" "$CACHE/mcq_explain_qwen35_397b_trained.jsonl"   trained
echo "[selfexpl_397b_tinker] DONE"
