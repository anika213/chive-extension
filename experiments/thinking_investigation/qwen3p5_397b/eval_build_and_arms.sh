#!/bin/bash
# cluster-specific; adapt paths to your setup.
# 397B THINKING run: build the binary-eval set, then score the two approved arms —
# transcript-only (plain) and counterfactual-with-thinking (cf, against the live 397B pool proxy).
# CPU/Anthropic only except cf (which needs the pool up on :9210). Idempotent: skips stages whose
# outputs exist; arms run with --resume. Tees to $NEW/eval_build.log.
set -uo pipefail
cd /workspace-vast/adamk/chive
source .env
export HF_HOME=/workspace-vast/$USER/hf_cache
export ANTHROPIC_API_KEY_HIGH_PRIORITY="${ANTHROPIC_API_KEY_HIGH_PRIORITY:-$ANTHROPIC_API_KEY}"
PY=python
NEW=${CHIVE_DATA:-data}/runs/opus_investigates_qwen3p5_397b_think1024_5000_thorough_n30
ES=$NEW/eval_set_conc3_n2000.jsonl
BASE=Qwen/Qwen3.5-397B-A17B-FP8
PROXY=http://localhost:9210   # thinking-capable 397B pool (NOT Tinker — Tinker can't do thinking)
mkdir -p "$NEW/binary_evals"
exec > >(tee -a "$NEW/eval_build.log") 2>&1
echo "================ 397B-think eval build+arms START $(date) ================"

# 1. concreteness (async Opus) in the background, parallel with the batch bank build
if [ ! -f "$NEW/mechanism_concreteness.jsonl" ]; then
  echo "[conc] launching mechanism-concreteness (min-score 8)"
  $PY -m chive.binary_eval.concreteness \
    --run-dir "$NEW" --out "$NEW/mechanism_concreteness.jsonl" --min-score 8 --concurrency 50 &
  CONC_PID=$!
else
  echo "[conc] already done"; CONC_PID=""
fi

# 2. binary counterfactual bank via the batch API (~1h; ~50% cheaper for a 5000 run)
if [ ! -f "$NEW/binary_counterfactual_questions.jsonl" ]; then
  echo "[bank] building binary counterfactual bank (--batch, min-score 8)"
  $PY -m chive.binary_eval.build_claims --run-dir "$NEW" \
    --out "$NEW/binary_counterfactual_questions.jsonl" --min-score 8 --n 100000 --batch
else
  echo "[bank] already done"
fi
[ -n "$CONC_PID" ] && { echo "[conc] waiting for concreteness to finish"; wait "$CONC_PID" || true; }
echo "[bank] claims: $(wc -l < $NEW/binary_counterfactual_questions.jsonl)"
echo "[conc] rows:   $(wc -l < $NEW/mechanism_concreteness.jsonl)"

# 3. assemble the eval set (concreteness>=3, up to 2000 claims, 50/50)
if [ ! -f "$ES" ]; then
  echo "[eval_set] building $ES"
  $PY -m chive.binary_eval.build_eval_set --run-dir "$NEW" --base-model "$BASE" \
    --bank "$NEW/binary_counterfactual_questions.jsonl" --min-concreteness 3 --n 2000 --seed 0 \
    --max-unsegmentable-rate 0.01
fi
echo "[eval_set] size: $(wc -l < $ES)"

# 4. plain arm — no GPU, the headline number (claude-opus-4-8 high, hardcoded in run_binary_eval)
echo "[plain] scoring transcript-only arm"
$PY -m chive.binary_eval.run_binary_eval --bank "$ES" --answerer plain \
  --label conc3 --concurrency 150 --resume --out-dir "$NEW/binary_evals"

# 5. cf arm — against the thinking-capable pool; re-runs reproduce reasoning-mode behavior
echo "[cf] scoring counterfactual arm against $PROXY (thinking 1024/2048)"
curl -sf -m 5 "$PROXY/v1/models" >/dev/null || { echo "[cf] FATAL: pool proxy not up"; exit 1; }
$PY -m chive.binary_eval.run_binary_eval --bank "$ES" --answerer agent --tool cf \
  --url "$PROXY" --base-model "$BASE" --cf-thinking-budget 1024 --cf-max-tokens 2048 \
  --label conc3 --concurrency 150 --resume --out-dir "$NEW/binary_evals"

# 6. AUROC/acc/Brier readout
echo "[table] AUROC over arms:"
$PY - <<'PYEOF'
import glob, json, os
def auroc(ps, ys):
    pos=[p for p,y in zip(ps,ys) if y]; neg=[p for p,y in zip(ps,ys) if not y]
    if not pos or not neg: return float('nan')
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
OUT=os.path.join(os.environ.get("CHIVE_DATA", "data"), "runs/opus_investigates_qwen3p5_397b_think1024_5000_thorough_n30/binary_evals")
print(f"  {'arm':48s} {'n':>5} {'AUROC':>7} {'acc':>6} {'Brier':>6}")
for f in sorted(glob.glob(f"{OUT}/*conc3*.jsonl")):
    if "toolcalls" in os.path.basename(f): continue
    rows=[json.loads(l) for l in open(f)]
    ps=[r["prob_true"] for r in rows]; ys=[r["is_true"] for r in rows]
    acc=sum((p>=0.5)==y for p,y in zip(ps,ys))/len(rows)
    brier=sum((p-y)**2 for p,y in zip(ps,ys))/len(rows)
    print(f"  {os.path.basename(f)[:48]:48s} {len(rows):5d} {auroc(ps,ys):7.3f} {acc:6.3f} {brier:6.3f}")
print("\n  refs: 8B-think plain 0.835 / cf 0.963; no-think 397B: see its binary_evals/")
PYEOF
echo "================ 397B-think eval build+arms DONE $(date) ================"
