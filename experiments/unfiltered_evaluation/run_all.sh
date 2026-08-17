#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# run_all.sh — reproduce the truly-unfiltered evaluation end-to-end (see README.md).
#
# Assumes the three WildChat run dirs already have the cf7 bank + mechanism_concreteness +
# conc3 eval set + published conc3 arm results (produced by main_counterfactual_eval/run_all.sh
# and direct_counterfactual_evaluation/run_all.sh). From there:
#   build complements -> spot-check file -> GPU servers + NLA/SAE caches (gemma) ->
#   score every arm on the complements -> analyze.py (SUMMARY.md).
#
# Arms (answerer = claude-opus-4-8 / high; ~150 Opus concurrency, ONE Opus arm at a time):
#   gemma  : plain, ao, nla, sae          (tool arms; capture + AO servers, <=6 GPUs infra)
#   qwen8b : plain, base + trained LoRA   (vLLM, 2x 1-GPU sbatch)
#   q397   : plain, base + trained        (Tinker, CPU/API)
#
# Idempotent: every Opus arm uses --resume; precompute stages skip if the cache covers the
# complement. Run from repo root.
set -uo pipefail
cd /workspace-vast/adamk/chive
source .env
export ANTHROPIC_API_KEY_HIGH_PRIORITY="${ANTHROPIC_API_KEY_HIGH_PRIORITY:-$ANTHROPIC_API_KEY}"
export HF_HOME=/workspace-vast/adamk/hf_cache
U=$USER
PY=python
AGENT=chive/binary_eval/agent
HERE=experiments/unfiltered_evaluation
DCF=experiments/direct_counterfactual_evaluation
DATA="${CHIVE_DATA:-/workspace-vast/adamk/chive/data}"
EXPDATA=$DATA/experiments/unfiltered_evaluation
RUNS=$DATA/runs
GEMMA=$RUNS/opus_investigates_gemma3_27b_it_5000_thorough_n30
# Released 8B adapter — download first:
#   hf download adamkarvonen/qwen3_8b_counterfactual_e1_kl0 --local-dir checkpoints/qwen3_8b_counterfactual_e1_kl0
LORA8=checkpoints/qwen3_8b_counterfactual_e1_kl0
# Tinker sampler weights used for the paper; adapter weights released at adamkarvonen/qwen3p5_397b_counterfactual_e1_kl0 (HF).
TINK397=tinker://7c443149-82fa-52af-94c6-e635f39da1d2:train:0/sampler_weights/final
GNLA_SPEC=gemma3-27b-L41
AO_GEMMA=adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it
EXCL="--exclude=node-13"
mkdir -p "$HERE/logs" "$EXPDATA/results"

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
# One Opus arm at a time (150-concurrency cap): run_arm BLOCKS until done.
run_arm(){ local tag="$1" outdir="$2"; shift 2; echo "[arm] $tag"
  PYTHONPATH=. $PY -m chive.binary_eval.run_binary_eval "$@" --label unfiltered --concurrency 150 --resume \
    --out-dir "$outdir" > "$HERE/logs/arm_$tag.log" 2>&1 || echo "[arm] $tag rc=$? (see log)"; }

# ===== Stage A: complements (CPU) =====
[ -f "$EXPDATA/eval_sets/gemma_complement.jsonl" ] || PYTHONPATH=. $PY "$HERE/build_unfiltered_complement.py"

COMPG=$EXPDATA/eval_sets/gemma_complement.jsonl
COMP8=$EXPDATA/eval_sets/qwen8b_complement.jsonl
COMP397=$EXPDATA/eval_sets/q397_complement.jsonl

# ===== Stage B: trained-model arms (no Opus concurrency used) =====
# 8B: two parallel 1-GPU sbatch jobs.
[ -f "$EXPDATA/results/8b_base_unfiltered.jsonl" ] || \
  sbatch $EXCL -J unf8b_base_$U "$DCF/jobs/eval_8b_vllm.sh" "$COMP8" "$EXPDATA/results/8b_base_unfiltered.jsonl" base
[ -f "$EXPDATA/results/8b_trained_unfiltered.jsonl" ] || \
  sbatch $EXCL -J unf8b_lora_$U "$DCF/jobs/eval_8b_vllm.sh" "$COMP8" "$EXPDATA/results/8b_trained_unfiltered.jsonl" "$LORA8"
# 397B: sequential Tinker runs (CPU/API), backgrounded against the Opus arms below.
(
  source .venv/bin/activate
  [ -f "$EXPDATA/results/397b_base_unfiltered.jsonl" ] || \
    python $DCF/run_structured_claim_eval.py --backend tinker \
      --eval-set "$COMP397" --out "$EXPDATA/results/397b_base_unfiltered.jsonl" --concurrency 100 \
      > "$HERE/logs/397b_base.log" 2>&1
  [ -f "$EXPDATA/results/397b_trained_unfiltered.jsonl" ] || \
    python $DCF/run_structured_claim_eval.py --backend tinker \
      --eval-set "$COMP397" --model-path "$TINK397" --out "$EXPDATA/results/397b_trained_unfiltered.jsonl" \
      --concurrency 100 > "$HERE/logs/397b_trained.log" 2>&1
) &

# ===== Stage C: gemma GPU infra + caches (<=6 GPUs: capture + 5-shard NLA array) =====
CAPG=$EXPDATA/results/capture_gemma.json
url_up "$CAPG" || sbatch -J unfcap_$U $EXCL "$AGENT/NLA/jobs/serve_capture.sh" google/gemma-3-27b-it "" "$CAPG"
wait_up "$CAPG"; GCAP=$(read_url "$CAPG")
GNLA_CACHE=$EXPDATA/eval_sets/nla_cache/$GNLA_SPEC.jsonl
if ! covers "$COMPG" "$GNLA_CACHE"; then
  sbatch -J unfnla_$U $EXCL "$AGENT/NLA/jobs/precompute_nla_array.sh" "$COMPG" "$GCAP" "$GNLA_SPEC"
  wait_job unfnla_$U
  cat "$EXPDATA/eval_sets/nla_cache/shards/${GNLA_SPEC}"*of5.jsonl > "$GNLA_CACHE" 2>/dev/null || \
    cat "$EXPDATA/eval_sets/nla_cache/${GNLA_SPEC}_shard"*of5.jsonl > "$GNLA_CACHE"
  covers "$COMPG" "$GNLA_CACHE" || echo "[nla] WARN incomplete cache: $GNLA_CACHE"
fi
GSAE_CACHE=$(ls "$EXPDATA"/eval_sets/sae_cache/gemma-scope-2-27b-it_layer_40*.jsonl 2>/dev/null | head -1 || true)
if [ -z "$GSAE_CACHE" ] || ! covers "$COMPG" "$GSAE_CACHE"; then
  sbatch -J unfsae_$U $EXCL "$AGENT/SAE/jobs/precompute_sae_cache.sh" "$COMPG" "$GCAP" google/gemma-3-27b-it --capture-layer 40 --resume
  wait_job unfsae_$U
  GSAE_CACHE=$(ls "$EXPDATA"/eval_sets/sae_cache/gemma-scope-2-27b-it_layer_40*.jsonl | head -1)
fi
AOG=$EXPDATA/results/ao_server.json
url_up "$AOG" || sbatch $EXCL "$AGENT/jobs/serve_gemma_ao.sh"   # writes $DATA/scratch/ao_server.json
[ -f "$AOG" ] || AOG=$DATA/scratch/ao_server.json
wait_up "$AOG"; GAO=$(read_url "$AOG")

# ===== Stage D: Opus arms, STRICTLY SEQUENTIAL (150 concurrency each) =====
run_arm gemma_plain  "$EXPDATA/results/gemma"  --bank "$COMPG" --answerer plain
run_arm qwen8b_plain "$EXPDATA/results/qwen8b" --bank "$COMP8" --answerer plain
run_arm q397_plain   "$EXPDATA/results/q397"   --bank "$COMP397" --answerer plain
run_arm gemma_ao  "$EXPDATA/results/gemma" --bank "$COMPG" --answerer agent --tool ao --url "$GAO" \
  --base-model google/gemma-3-27b-it --ao-lora "$AO_GEMMA"
run_arm gemma_nla "$EXPDATA/results/gemma" --bank "$COMPG" --answerer agent --tool nla \
  --base-model google/gemma-3-27b-it --nla-cache "$GNLA_CACHE"
run_arm gemma_sae "$EXPDATA/results/gemma" --bank "$COMPG" --answerer agent --tool sae \
  --base-model google/gemma-3-27b-it --sae-cache "$GSAE_CACHE"
scancel -u "$U" -n unfcap_$U 2>/dev/null; scancel -u "$U" -n vlens_gemma_ao 2>/dev/null

# ===== Stage E: wait for trained arms, then analyze =====
wait   # 397B tinker subshell
wait_job unf8b_base_$U; wait_job unf8b_lora_$U
PYTHONPATH=. $PY "$HERE/analyze.py"
echo "######## unfiltered_evaluation run_all DONE -> $HERE/SUMMARY.md ########"
