#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Cross-model binary counterfactual eval on the GEMMA-3-27B test set.
#
# Re-runs the binary-claim eval (run_binary_eval) with NON-Anthropic answerers — GPT-5.5 and
# Gemini-3.1-Pro via OpenRouter — across the interp tools (plain / NLA / SAE / AO). The canonical
# Opus-4.8 arm already lives in the run dir's binary_evals/ (see README); we write OUR results to
# this experiment's data results/ so the main plotter's claim-id intersection is never perturbed.
#
# Backend support: run_binary_eval now drives both the plain answerer AND the tool-using agent
# through the LLMClient abstraction, so --answerer-backend openrouter works for every arm. Reasoning
# is controlled with --reasoning-effort high (anthropic adaptive effort / openrouter reasoning.effort).
#
# Arms & GPUs:
#   plain : API only (no GPU, no cache)
#   nla   : API + precomputed nla_cache (no GPU)
#   sae   : API + precomputed sae_cache (no GPU)
#   ao    : API answerer + the gemma vllm-lens AO server (1 GPU, serve_gemma_ao.sh)
#   cf    : SKIPPED for now (would need a second gemma vLLM target server; expensive)
#
# The two models run as PARALLEL background drivers (separate OpenRouter rate limits per model);
# arms within a model run SEQUENTIALLY. plain/nla/sae overlap the AO server boot; AO waits for it.
#
# Run from repo root with .venv active and .env sourced. CPU/API driver — no GPU here.
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
export PYTHONPATH=.

HERE=experiments/cross_model
EVAL=chive.binary_eval.run_binary_eval
DATA="${CHIVE_DATA:-/workspace-vast/adamk/chive/data}"
RUN_DIR=$DATA/runs/opus_investigates_gemma3_27b_it_5000_thorough_n30
BANK=$RUN_DIR/eval_set_conc3_n2000.jsonl
OUT=$DATA/experiments/cross_model/results
mkdir -p "$OUT" "$HERE/logs"

BASE_MODEL=google/gemma-3-27b-it
AO_LORA=adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it
LABEL=conc3                          # match the canonical filename tag (different dir = no collision)

# Answerer models as "<backend> <model>" pairs. Opus is re-run on the SAME (hardened) tool as
# GPT/Gemini and dumped into the SAME results/ dir, so all three are directly comparable (the
# canonical binary_evals/ Opus used a different selection-tool schema). The eval names result files by
# model+effort+label, so the three never collide. arm tag (plain/nla_*/sae_*/ao_*) is automatic.
# All three have INDEPENDENT rate limits (openrouter per-provider, anthropic) -> run in parallel.
MODELS=("openrouter openai/gpt-5.5" "openrouter google/gemini-3.1-pro-preview" "anthropic claude-opus-4-8")
ARMS="${ARMS:-plain nla sae ao}"     # override e.g. ARMS="nla" for a smoke run
CONC="${CONC:-100}"                  # API-arm concurrency per model (separate rate limits)
AO_CONC="${AO_CONC:-40}"             # ao hits ONE shared vllm-lens server; 3 models -> ~120 in-flight
MAXCLAIMS="${MAXCLAIMS:-}"           # set e.g. 20 for a smoke run; empty = full bank

mc_flag() { [ -n "$MAXCLAIMS" ] && echo "--max-claims $MAXCLAIMS" || true; }

# --- AO server (only if the ao arm is requested) -------------------------------------------------
AO_CONF=$DATA/scratch/ao_server.json
need_ao=0; for a in $ARMS; do [ "$a" = "ao" ] && need_ao=1; done
if [ "$need_ao" = 1 ]; then
  EXCLUDE=$(./jobs/find_zombie_gpu_processes.sh --exclude || true)
  echo "[run_all] submitting gemma AO server ($EXCLUDE)"
  sbatch $EXCLUDE chive/binary_eval/agent/jobs/serve_gemma_ao.sh
fi

wait_for_ao() {
  echo "[run_all] waiting for gemma AO server (ao_server.json + /v1/models)..."
  while :; do
    if [ -f "$AO_CONF" ]; then
      url=$(python -c "import json;print(json.load(open('$AO_CONF'))['url'])" 2>/dev/null || true)
      if [ -n "$url" ] && curl -sf -m 5 "$url/v1/models" >/dev/null 2>&1; then
        echo "[run_all] AO server ready at $url"; return 0
      fi
    fi
    sleep 20
  done
}

# --- per-model driver: run the requested arms sequentially ---------------------------------------
run_model() {
  local BACKEND="$1" SLUG="$2" SAFE LOG
  SAFE=$(echo "$SLUG" | tr '/' '_')
  LOG=$HERE/logs/${SAFE}.log
  echo "[run_all] driver start: $BACKEND:$SLUG -> $LOG"
  for arm in $ARMS; do
    echo "[$SLUG] === arm=$arm ===" | tee -a "$LOG"
    case "$arm" in
      plain)
        python -m "$EVAL" --bank "$BANK" --answerer plain \
          --answerer-backend "$BACKEND" --answerer-model "$SLUG" --reasoning-effort high \
          --concurrency "$CONC" --out-dir "$OUT" --label "$LABEL" $(mc_flag) >>"$LOG" 2>&1
        ;;
      nla|sae)
        python -m "$EVAL" --bank "$BANK" --answerer agent --tool "$arm" --base-model "$BASE_MODEL" \
          --answerer-backend "$BACKEND" --answerer-model "$SLUG" --reasoning-effort high \
          --concurrency "$CONC" --out-dir "$OUT" --label "$LABEL" $(mc_flag) >>"$LOG" 2>&1
        ;;
      ao)
        wait_for_ao
        python -m "$EVAL" --bank "$BANK" --answerer agent --tool ao --ao-lora "$AO_LORA" \
          --base-model "$BASE_MODEL" \
          --answerer-backend "$BACKEND" --answerer-model "$SLUG" --reasoning-effort high \
          --concurrency "$AO_CONC" --out-dir "$OUT" --label "$LABEL" $(mc_flag) >>"$LOG" 2>&1
        ;;
      *) echo "[$SLUG] unknown arm: $arm" | tee -a "$LOG"; exit 1 ;;
    esac
    echo "[$SLUG] done arm=$arm" | tee -a "$LOG"
  done
  echo "[run_all] driver done: $SLUG"
}

# --- launch all models in parallel, wait, analyze ------------------------------------------------
# run_model is called UNQUOTED so "<backend> <model>" word-splits into its two args.
pids=()
for m in "${MODELS[@]}"; do run_model $m & pids+=($!); done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "[run_all] a model driver failed — see $HERE/logs/"; exit 1; }

echo "[run_all] all drivers done; analyzing"
python "$HERE/analyze.py" --results-dir "$OUT"
echo "[run_all] done -> $HERE/SUMMARY.md"
