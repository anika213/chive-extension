#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
# Score the GPU/agent arms (cf, ao_single, ao_multi, nla) for the 5000 thinking run, then print the
# full AUROC table. Mirrors run_all.sh's qwen block for a single run. NO SAE (no Qwen SAE).
# Idempotent: run_binary_eval --resume per arm; reuses live servers; rebuilds NLA cache only if missing.
set -uo pipefail
cd /workspace-vast/adamk/chive
source .env
export HF_HOME=/workspace-vast/$USER/hf_cache
export ANTHROPIC_API_KEY_HIGH_PRIORITY="${ANTHROPIC_API_KEY_HIGH_PRIORITY:-$ANTHROPIC_API_KEY}"
U=$USER
PY=python
EXCL="--exclude=node-13"
AGENT=chive/binary_eval/agent
RB=chive.binary_eval.run_binary_eval
RES=${CHIVE_DATA:-data}/scratch
NEW=${CHIVE_DATA:-data}/runs/opus_investigates_qwen3_8b_think1024_5000_thorough_n30
ES=$NEW/eval_set_conc3_n2000.jsonl
OUT=$NEW/binary_evals
LOG=$NEW/eval_arms_logs; mkdir -p "$LOG" "$RES"
AO_SINGLE=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
AO_MULTI=adamkarvonen/checkpoints_500k_pl_31k_spqav2_199k_sqav3_50k_hb_126k_cls
QNLA_SPEC=qwen3-8b-baseline-nosalvage
NLA_OUT=$NEW/nla_cache/$QNLA_SPEC.jsonl
mkdir -p "$NEW/nla_cache"
exec > >(tee -a "$NEW/eval_arms.log") 2>&1
echo "================ eval GPU arms START $(date) ================"

read_url(){ $PY -c "import json,sys;print(json.load(open(sys.argv[1]))['url'])" "$1" 2>/dev/null; }
url_up(){ local u; u=$(read_url "$1") || return 1; [ -n "$u" ] && curl -sf -m 5 "$u/v1/models" >/dev/null 2>&1; }
wait_up(){ local cj="$1"; echo "[wait] $cj"; for i in $(seq 1 240); do url_up "$cj" && { echo "[up] $(read_url "$cj")"; return 0; }; sleep 15; done; echo "[TIMEOUT] $cj"; return 1; }
wait_job(){ local n="$1"; echo "[wait-job] $n"; sleep 20; while squeue -u "$U" -h -n "$n" -o %i 2>/dev/null | grep -q .; do sleep 30; done; echo "[done] $n"; }
covers(){ $PY - "$1" "$2" <<'PYEOF'
import json,sys,os
need={json.loads(l)["prompt_id"] for l in open(sys.argv[1]) if l.strip()}
have=set()
if os.path.exists(sys.argv[2]): have={json.loads(l)["prompt_id"] for l in open(sys.argv[2]) if l.strip()}
sys.exit(0 if not (need-have) else 1)
PYEOF
}
run_arm(){ local tag="$1"; shift; echo "[arm] $tag START $(date '+%H:%M')";
  $PY -m "$RB" "$@" --bank "$ES" --label conc3 --concurrency 150 --resume --out-dir "$OUT" > "$LOG/arm_$tag.log" 2>&1 \
    || echo "[arm] $tag rc=$? (see $LOG/arm_$tag.log)"; echo "[arm] $tag DONE $(date '+%H:%M')"; }

# --- servers (gpu:1 each; generation pool already torn down; under 16-GPU cap) ---
AOQ=$RES/ao_server_qwen.json; CFQ=$RES/cf_server_qwen.json; CAPQ=$RES/capture_qwen.json
url_up "$AOQ" || { echo "[srv] launching AO server"; sbatch $EXCL "$AGENT/jobs/serve_qwen_ao.sh"; }
scancel -u "$U" -n cf_target 2>/dev/null; scancel -u "$U" -n nla_capture 2>/dev/null; sleep 3
echo "[srv] launching cf target + NLA capture"
sbatch -J cf_target  $EXCL "$AGENT/jobs/serve_cf_target.sh" Qwen/Qwen3-8B "" "$CFQ" "--reasoning-parser qwen3 --reasoning-config {}"
sbatch -J nla_capture $EXCL "$AGENT/NLA/jobs/serve_capture.sh" Qwen/Qwen3-8B "" "$CAPQ"

# --- NLA cache (needs the capture server; 5-shard array) ---
if ! covers "$ES" "$NLA_OUT"; then
  wait_up "$CAPQ"; QCAP=$(read_url "$CAPQ")
  echo "[nla] precompute array over $ES"
  sbatch -J nlapc_$U $EXCL "$AGENT/NLA/jobs/precompute_nla_array.sh" "$ES" "$QCAP" "$QNLA_SPEC"
  wait_job nlapc_$U
  cat "$NEW/nla_cache/shards/${QNLA_SPEC}_shard"*of5.jsonl > "$NLA_OUT" 2>/dev/null || true
  covers "$ES" "$NLA_OUT" && echo "[nla] cache OK ($(wc -l < $NLA_OUT) rows)" || echo "[nla] WARN incomplete cache"
else
  echo "[nla] cache already covers eval set"
fi
scancel -u "$U" -n nla_capture 2>/dev/null

# --- score arms (servers stay up across arms) ---
wait_up "$AOQ"; QAO=$(read_url "$AOQ")
wait_up "$CFQ"; QCF=$(read_url "$CFQ")
run_arm cf        --answerer agent --tool cf  --url "$QCF" --base-model Qwen/Qwen3-8B --cf-thinking-budget 1024 --cf-max-tokens 2048
run_arm ao_single --answerer agent --tool ao  --url "$QAO" --base-model Qwen/Qwen3-8B --ao-lora "$AO_SINGLE"
run_arm ao_multi  --answerer agent --tool ao  --url "$QAO" --base-model Qwen/Qwen3-8B --ao-lora "$AO_MULTI"
run_arm nla       --answerer agent --tool nla --base-model Qwen/Qwen3-8B --nla-cache "$NLA_OUT"

# --- teardown eval servers ---
scancel -u "$U" -n cf_target 2>/dev/null
scancel -u "$U" -n vlens_qwen_ao 2>/dev/null

# --- AUROC table over all arms ---
echo "[table] AUROC over all arms:"
$PY - <<'PYEOF'
import glob, json, os
def auroc(ps, ys):
    pos=[p for p,y in zip(ps,ys) if y]; neg=[p for p,y in zip(ps,ys) if not y]
    if not pos or not neg: return float('nan')
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
OUT=os.path.join(os.environ.get("CHIVE_DATA", "data"), "runs/opus_investigates_qwen3_8b_think1024_5000_thorough_n30/binary_evals")
print(f"  {'arm':48s} {'n':>5} {'AUROC':>7} {'acc':>6} {'Brier':>6}")
for f in sorted(glob.glob(f"{OUT}/*conc3*.jsonl")):
    rows=[json.loads(l) for l in open(f)]
    ps=[r["prob_true"] for r in rows]; ys=[r["is_true"] for r in rows]
    acc=sum((p>=0.5)==y for p,y in zip(ps,ys))/len(rows)
    brier=sum((p-y)**2 for p,y in zip(ps,ys))/len(rows)
    print(f"  {os.path.basename(f)[:48]:48s} {len(rows):5d} {auroc(ps,ys):7.3f} {acc:6.3f} {brier:6.3f}")
print("\n  no-think qwen3-8b ref (main_counterfactual_eval SUMMARY.md): plain 0.798/73.3% | cf 0.949/89.1% | ao/nla ~0.79")
PYEOF
echo "================ eval GPU arms DONE $(date) ================"
