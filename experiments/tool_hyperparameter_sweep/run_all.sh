#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# run_all.sh — reproduce the tool-hyperparameter sweep end-to-end.
#
# Sweeps the two evaluation hyperparameters of the activation-tool arms on the Gemma-3-27B-IT
# cf7 eval set (the full eval_set_conc3_n2000.jsonl, 1750 claims — same bank as every main-run
# arm; analyze.py restricts all reported numbers to the quality slice, n~1380):
#   - tool calls per claim: 2 and 10   (5 = the existing main-run arms, read back)  x AO/NLA/SAE
#   - token positions per call: 16 and 64  (32 = existing, read back)  x NLA/SAE only
#     (the AO selects transcript segments, not token positions, so the knob doesn't apply)
# The existing main-run arms (plain + AO/NLA/SAE at 5 calls / 32 positions) are NOT re-run:
# analyze.py reads them back from the canonical result files restricted to the same claim_ids.
#
# GPU: only the two AO arms need the live vllm-lens server (serve_gemma_ao.sh, 1x H200);
# NLA and SAE are cache-only and cost API only. The read caches (nla_cache/, sae_cache/ next
# to the bank) are not part of the data release: regenerate them first with
# chive/binary_eval/agent/NLA/precompute_nla_cache.py and agent/SAE/precompute_sae_cache.py
# (GPU), or rely on --resume over the shipped result files, which skips the caches entirely.
#
# Idempotent: every arm runs with --resume; the AO server is reused if already live.
# Run from repo root.
set -uo pipefail
cd /workspace-vast/adamk/chive
source .env
export HF_HOME=/workspace-vast/adamk/hf_cache
export ANTHROPIC_API_KEY_HIGH_PRIORITY="${ANTHROPIC_API_KEY_HIGH_PRIORITY:-$ANTHROPIC_API_KEY}"
export PYTHONPATH=.
U=$USER
PY=python
RB=chive.binary_eval.run_binary_eval
AGENT=chive/binary_eval/agent
HERE=experiments/tool_hyperparameter_sweep
DATA="${CHIVE_DATA:-/workspace-vast/adamk/chive/data}"
EXPDATA=$DATA/experiments/tool_hyperparameter_sweep
LOG=$HERE/logs; mkdir -p "$LOG" "$EXPDATA/results"

RUN=$DATA/runs/opus_investigates_gemma3_27b_it_5000_thorough_n30
BASE=google/gemma-3-27b-it
AO_LORA=adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it
NLA_CACHE=$RUN/nla_cache/gemma3-27b-L41.jsonl
SAE_CACHE=$RUN/sae_cache/gemma-scope-2-27b-it_layer_40_width_65k_l0_medium.jsonl
BANK=$RUN/eval_set_conc3_n2000.jsonl
AOG=$DATA/scratch/ao_server.json

read_url(){ $PY -c "import json,sys;print(json.load(open(sys.argv[1]))['url'])" "$1" 2>/dev/null; }
url_up(){ local u; u=$(read_url "$1") || return 1; [ -n "$u" ] && curl -sf -m 5 "$u/v1/models" >/dev/null 2>&1; }
wait_up(){ local cj="$1"; echo "[wait] $cj"; for i in $(seq 1 200); do url_up "$cj" && { echo "[up] $(read_url "$cj")"; return 0; }; sleep 15; done; echo "[TIMEOUT] $cj"; return 1; }
run_arm(){ local tag="$1"; shift; echo "[arm] $tag  $(date +%H:%M)"; $PY -m "$RB" --bank "$BANK" --answerer agent \
  --base-model "$BASE" --out-dir "$EXPDATA/results" --concurrency 100 --resume "$@" \
  > "$LOG/arm_$tag.log" 2>&1 || echo "[arm] $tag rc=$? (see $LOG/arm_$tag.log)"; }

echo "######## tool_hyperparameter_sweep run_all START $(date) ########"

# ===== Stage 1: AO server (only GPU dependency; reused if already live) =====
if ! url_up "$AOG"; then
  EXCL=$(./jobs/find_zombie_gpu_processes.sh --exclude | grep -o '^--exclude=.*' | tail -1)
  echo "[sbatch] serve_gemma_ao.sh ${EXCL:-<no exclusions>}"
  sbatch ${EXCL:+"$EXCL"} "$AGENT/jobs/serve_gemma_ao.sh"
fi

# ===== Stage 2: cache-only sweep arms first (NLA + SAE; no GPU) =====
# tool-calls sweep (positions stay at the default 32)
run_arm nla_tc2   --tool nla --nla-cache "$NLA_CACHE" --max-tool-calls 2  --label tc2
run_arm nla_tc10  --tool nla --nla-cache "$NLA_CACHE" --max-tool-calls 10 --label tc10
run_arm sae_tc2   --tool sae --sae-cache "$SAE_CACHE" --max-tool-calls 2  --label tc2
run_arm sae_tc10  --tool sae --sae-cache "$SAE_CACHE" --max-tool-calls 10 --label tc10
# positions sweep (tool calls stay at the default 5)
run_arm nla_pos16 --tool nla --nla-cache "$NLA_CACHE" --max-tokens-per-call 16 --label pos16
run_arm nla_pos64 --tool nla --nla-cache "$NLA_CACHE" --max-tokens-per-call 64 --label pos64
run_arm sae_pos16 --tool sae --sae-cache "$SAE_CACHE" --max-tokens-per-call 16 --label pos16
run_arm sae_pos64 --tool sae --sae-cache "$SAE_CACHE" --max-tokens-per-call 64 --label pos64

# ===== Stage 3: AO tool-calls arms (need the live server) =====
wait_up "$AOG"
run_arm ao_tc2  --tool ao --ao-lora "$AO_LORA" --max-tool-calls 2  --label tc2
run_arm ao_tc10 --tool ao --ao-lora "$AO_LORA" --max-tool-calls 10 --label tc10
scancel -u "$U" -n vlens_gemma_ao

# ===== Stage 4: analysis (SUMMARY.md + plots/) =====
echo "[analyze] writing SUMMARY.md + plots"
$PY "$HERE/analyze.py"
echo "######## tool_hyperparameter_sweep run_all DONE $(date) -> $HERE/SUMMARY.md ########"
