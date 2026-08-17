#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# run_all.sh — reproduce the MAIN counterfactual-tool eval end-to-end for all 5 runs.
#
# Assumes each run dir already has the upstream pipeline outputs (completions -> screening ->
# investigations -> verification), produced by chive/pipeline/run_pipeline.py.
# From there this script reproduces: cf7 bank -> mechanism-concreteness -> eval set -> GPU servers
# + NLA/SAE caches -> score every arm -> analyze.py (writes SUMMARY.md + plots/).
#
# Arms per run (answerer = claude-opus-4-8 / high, hardcoded in run_binary_eval; cf-n=30):
#   gemma-3-27b-it            : plain, ao, cf (vLLM target), nla, sae
#   qwen3-8b  (5000 & petri)  : plain, ao(single), ao(multi), cf (vLLM target), nla
#   qwen3.5-397b (5000 & petri): plain, cf (Tinker bf16 target)   [no activation tools for 397B]
#
# Idempotent: skips a stage whose output already exists; every arm runs with --resume; live GPU
# servers are reused. GPU jobs exclude node-13 (honor-system isolation has bricked it before).
# Anthropic batch builds are the long pole (~1h/run); the cf/ao/nla scoring needs GPU servers.
#
# Run from repo root. Total cost: ~5 batch builds + ~5 GPU servers at a time + ~19 scored arms.
set -uo pipefail
cd /workspace-vast/adamk/chive
source .env
export HF_HOME=/workspace-vast/adamk/hf_cache
export ANTHROPIC_API_KEY_HIGH_PRIORITY="${ANTHROPIC_API_KEY_HIGH_PRIORITY:-$ANTHROPIC_API_KEY}"
export PYTHONPATH=.
U=$USER
PY=python
RB=chive.binary_eval.run_binary_eval
BUILD=chive.binary_eval.build_claims
EVALSET=chive.binary_eval.build_eval_set
CONC=chive.binary_eval.concreteness
AGENT=chive/binary_eval/agent
DATA="${CHIVE_DATA:-/workspace-vast/adamk/chive/data}"
RES=$DATA/scratch
RUNS=$DATA/runs
HERE=experiments/main_counterfactual_eval
LOG=$HERE/logs; mkdir -p "$LOG"
EXCL="--exclude=node-13"
ES=eval_set_conc3_n2000.jsonl

GEMMA=$RUNS/opus_investigates_gemma3_27b_it_5000_thorough_n30
QWEN=$RUNS/opus_investigates_qwen3_8b_5000_thorough_n30
QWENP=$RUNS/opus_investigates_qwen3_8b_petri_thorough_n30
Q5000=$RUNS/opus_investigates_qwen3p5_397b_5000_thorough_n30
QPETRI=$RUNS/opus_investigates_qwen3p5_397b_petri_thorough_n30
declare -A BASE=([$GEMMA]=google/gemma-3-27b-it [$QWEN]=Qwen/Qwen3-8B [$QWENP]=Qwen/Qwen3-8B \
                 [$Q5000]=Qwen/Qwen3.5-397B-A17B-FP8 [$QPETRI]=Qwen/Qwen3.5-397B-A17B-FP8)
AO_GEMMA=adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it
AO_QWEN_SINGLE=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
AO_QWEN_MULTI=adamkarvonen/checkpoints_500k_pl_31k_spqav2_199k_sqav3_50k_hb_126k_cls
GNLA_SPEC=gemma3-27b-L41
QNLA_SPEC=qwen3-8b-baseline-nosalvage
GSAE=gemma-scope-2-27b-it_layer_40_width_65k_l0_medium

read_url(){ $PY -c "import json,sys;print(json.load(open(sys.argv[1]))['url'])" "$1" 2>/dev/null; }
url_up(){ local u; u=$(read_url "$1") || return 1; [ -n "$u" ] && curl -sf -m 5 "$u/v1/models" >/dev/null 2>&1; }
wait_up(){ local cj="$1"; echo "[wait] $cj"; for i in $(seq 1 200); do url_up "$cj" && { echo "[up] $(read_url "$cj")"; return 0; }; sleep 15; done; echo "[TIMEOUT] $cj"; return 1; }
wait_job(){ local n="$1"; echo "[wait-job] $n"; sleep 20; while squeue -u "$U" -h -n "$n" -o %i 2>/dev/null | grep -q .; do sleep 30; done; echo "[done] $n"; }
covers(){ $PY - "$1" "$2" <<'PY'
import json,sys,os
need={json.loads(l)["prompt_id"] for l in open(sys.argv[1]) if l.strip()}
have=set()
if os.path.exists(sys.argv[2]): have={json.loads(l)["prompt_id"] for l in open(sys.argv[2]) if l.strip()}
sys.exit(0 if not (need-have) else 1)
PY
}
run_arm(){ local tag="$1"; shift; echo "[arm] $tag"; $PY -m "$RB" "$@" --label conc3 --concurrency 150 --resume > "$LOG/arm_$tag.log" 2>&1 || echo "[arm] $tag rc=$? (see $LOG/arm_$tag.log)"; }
# NLA cache via the 5-shard array, then cat-merge. $1=run $2=spec $3=capture_url
nla_cache(){ local R="$1" spec="$2" cap="$3"; local out="$R/nla_cache/$spec.jsonl"
  covers "$R/$ES" "$out" && { echo "[nla] cache ok: $out"; return 0; }
  echo "[nla] array precompute $spec"
  sbatch -J nlapc_$U $EXCL "$AGENT/NLA/jobs/precompute_nla_array.sh" "$R/$ES" "$cap" "$spec"
  wait_job nlapc_$U
  # precompute_nla_cache.py writes shards to nla_cache/shards/; fall back to the
  # legacy flat layout (same pattern as unfiltered_evaluation/run_all.sh)
  cat "$R/nla_cache/shards/${spec}_shard"*of5.jsonl > "$out" 2>/dev/null || \
    cat "$R/nla_cache/${spec}_shard"*of5.jsonl > "$out"
  covers "$R/$ES" "$out" || echo "[nla] WARN incomplete: $out"
}

echo "######## run_all START $(date) ########"

# ===== Stage A+B: cf7 bank + concreteness + eval set (CPU/Anthropic, no GPU) =====
# Bank uses the batch API (~50% cheaper, ~1h/run); concreteness is async. Both are idempotent.
for R in "$GEMMA" "$QWEN" "$QWENP" "$Q5000" "$QPETRI"; do
  B=${BASE[$R]}; echo "[build] $R (base $B)"
  [ -f "$R/binary_counterfactual_questions.jsonl" ] || \
    $PY -m "$BUILD" --run-dir "$R" --out "$R/binary_counterfactual_questions.jsonl" --min-score 8 --n 100000 --batch
  [ -f "$R/mechanism_concreteness.jsonl" ] || \
    $PY -m "$CONC" --run-dir "$R" --out "$R/mechanism_concreteness.jsonl" --min-score 8 --concurrency 50
  [ -f "$R/$ES" ] || \
    $PY -m "$EVALSET" --run-dir "$R" --base-model "$B" --bank "$R/binary_counterfactual_questions.jsonl" \
        --min-concreteness 3 --n 2000 --seed 0
done

# ===== Stage C/D: caches + scored arms =====

# --- 397B (no GPU): plain + cf via Tinker bf16 ---
for R in "$Q5000" "$QPETRI"; do
  run_arm "$(basename $R)_plain" --bank "$R/$ES" --answerer plain
  run_arm "$(basename $R)_cf" --bank "$R/$ES" --answerer agent --tool cf \
    --base-model Qwen/Qwen3.5-397B-A17B-FP8 --cf-target tinker:Qwen/Qwen3.5-397B-A17B
done

# --- Gemma: capture (nla+sae) + AO + cf servers; 5 arms ---
CAPG=$RES/capture_gemma.json; CFG=$RES/cf_server_gemma.json; AOG=$RES/ao_server.json
url_up "$CAPG" || sbatch -J capg_$U $EXCL "$AGENT/NLA/jobs/serve_capture.sh" google/gemma-3-27b-it "" "$CAPG"
url_up "$CFG"  || sbatch -J cfg_$U  $EXCL "$AGENT/jobs/serve_cf_target.sh" google/gemma-3-27b-it "" "$CFG"
url_up "$AOG"  || sbatch $EXCL "$AGENT/jobs/serve_gemma_ao.sh"
wait_up "$CAPG"; GCAP=$(read_url "$CAPG")
nla_cache "$GEMMA" "$GNLA_SPEC" "$GCAP"
GSAE_OUT=$GEMMA/sae_cache/$GSAE.jsonl
if ! covers "$GEMMA/$ES" "$GSAE_OUT"; then
  echo "[sae] precompute gemma"
  sbatch -J saepc_$U $EXCL "$AGENT/SAE/jobs/precompute_sae_cache.sh" "$GEMMA/$ES" "$GCAP" google/gemma-3-27b-it --capture-layer 40 --resume
  wait_job saepc_$U
fi
wait_up "$CFG"; wait_up "$AOG"; GCF=$(read_url "$CFG"); GAO=$(read_url "$AOG")
run_arm gemma_plain --bank "$GEMMA/$ES" --answerer plain
run_arm gemma_cf    --bank "$GEMMA/$ES" --answerer agent --tool cf --url "$GCF" --base-model google/gemma-3-27b-it
run_arm gemma_ao    --bank "$GEMMA/$ES" --answerer agent --tool ao --url "$GAO" --base-model google/gemma-3-27b-it --ao-lora "$AO_GEMMA"
run_arm gemma_nla   --bank "$GEMMA/$ES" --answerer agent --tool nla --base-model google/gemma-3-27b-it --nla-cache "$GEMMA/nla_cache/$GNLA_SPEC.jsonl"
run_arm gemma_sae   --bank "$GEMMA/$ES" --answerer agent --tool sae --base-model google/gemma-3-27b-it --sae-cache "$GSAE_OUT"
scancel -u "$U" -n capg_$U; scancel -u "$U" -n cfg_$U; scancel -u "$U" -n vlens_gemma_ao

# --- Qwen 8B (5000 + petri): shared AO server (both LoRAs); per-run capture + cf ---
AOQ=$RES/ao_server_qwen.json
url_up "$AOQ" || sbatch $EXCL "$AGENT/jobs/serve_qwen_ao.sh"
wait_up "$AOQ"; QAO=$(read_url "$AOQ")
for R in "$QWEN" "$QWENP"; do
  CAPQ=$RES/capture_qwen.json; CFQ=$RES/cf_server_qwen.json
  scancel -u "$U" -n capq_$U 2>/dev/null; scancel -u "$U" -n cfq_$U 2>/dev/null; sleep 3
  sbatch -J capq_$U $EXCL "$AGENT/NLA/jobs/serve_capture.sh" Qwen/Qwen3-8B "" "$CAPQ"
  sbatch -J cfq_$U  $EXCL "$AGENT/jobs/serve_cf_target.sh" Qwen/Qwen3-8B "" "$CFQ"
  wait_up "$CAPQ"; QCAP=$(read_url "$CAPQ")
  nla_cache "$R" "$QNLA_SPEC" "$QCAP"
  scancel -u "$U" -n capq_$U
  wait_up "$CFQ"; QCF=$(read_url "$CFQ")
  b=$(basename "$R")
  run_arm ${b}_plain    --bank "$R/$ES" --answerer plain
  run_arm ${b}_cf       --bank "$R/$ES" --answerer agent --tool cf --url "$QCF" --base-model Qwen/Qwen3-8B
  run_arm ${b}_ao_single --bank "$R/$ES" --answerer agent --tool ao --url "$QAO" --base-model Qwen/Qwen3-8B --ao-lora "$AO_QWEN_SINGLE"
  run_arm ${b}_ao_multi  --bank "$R/$ES" --answerer agent --tool ao --url "$QAO" --base-model Qwen/Qwen3-8B --ao-lora "$AO_QWEN_MULTI"
  run_arm ${b}_nla      --bank "$R/$ES" --answerer agent --tool nla --base-model Qwen/Qwen3-8B --nla-cache "$R/nla_cache/$QNLA_SPEC.jsonl"
  scancel -u "$U" -n cfq_$U
done
scancel -u "$U" -n vlens_qwen_ao

# ===== Stage E: analysis (SUMMARY.md + plots/) =====
echo "[analyze] writing SUMMARY.md + plots"
$PY "$HERE/analyze.py" --mode both
echo "######## run_all DONE $(date) -> $HERE/SUMMARY.md ########"
