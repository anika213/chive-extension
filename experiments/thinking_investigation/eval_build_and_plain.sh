#!/bin/bash
# cluster-specific; adapt paths to your setup.
# Build the binary-eval set for the 5000 thinking run + score the transcript-only (plain) arm.
# CPU/Anthropic only (no GPU). Idempotent (skips stages whose outputs exist). Tees to eval_build.log.
set -uo pipefail
cd /workspace-vast/adamk/chive
source .env
export HF_HOME=/workspace-vast/$USER/hf_cache
export ANTHROPIC_API_KEY_HIGH_PRIORITY="${ANTHROPIC_API_KEY_HIGH_PRIORITY:-$ANTHROPIC_API_KEY}"
PY=python
NEW=${CHIVE_DATA:-data}/runs/opus_investigates_qwen3_8b_think1024_5000_thorough_n30
ES=$NEW/eval_set_conc3_n2000.jsonl
mkdir -p "$NEW/binary_evals"
exec > >(tee -a "$NEW/eval_build.log") 2>&1
echo "================ eval build+plain START $(date) ================"

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
  $PY -m chive.binary_eval.build_eval_set --run-dir "$NEW" --base-model Qwen/Qwen3-8B \
    --bank "$NEW/binary_counterfactual_questions.jsonl" --min-concreteness 3 --n 2000 --seed 0 \
    --max-unsegmentable-rate 0.01
fi
echo "[eval_set] size: $(wc -l < $ES)"

# 4. score the transcript-only (plain) arm — no GPU, the headline number
echo "[plain] scoring transcript-only arm (claude-opus-4-8, high)"
$PY -m chive.binary_eval.run_binary_eval --bank "$ES" --answerer plain \
  --label conc3 --concurrency 150 --resume --out-dir "$NEW/binary_evals"

# 5. quick AUROC/acc readout for plain
echo "[plain] AUROC/acc:"
$PY - <<'PYEOF'
import glob, json, os
def auroc(ps, ys):
    pos=[p for p,y in zip(ps,ys) if y]; neg=[p for p,y in zip(ps,ys) if not y]
    if not pos or not neg: return float('nan')
    r=sum((a>b)+0.5*(a==b) for a in pos for b in neg)
    return r/(len(pos)*len(neg))
NEW=os.path.join(os.environ.get("CHIVE_DATA", "data"), "runs/opus_investigates_qwen3_8b_think1024_5000_thorough_n30")
for f in sorted(glob.glob(f"{NEW}/binary_evals/*plain*conc3*.jsonl") or glob.glob(f"{NEW}/binary_evals/*conc3*.jsonl")):
    rows=[json.loads(l) for l in open(f)]
    ps=[r["prob_true"] for r in rows]; ys=[r["is_true"] for r in rows]
    acc=sum((p>=0.5)==y for p,y in zip(ps,ys))/len(rows)
    print(f"  {os.path.basename(f):50s} n={len(rows)} AUROC={auroc(ps,ys):.3f} acc={acc:.3f}  (no-think baseline 0.798/0.733)")
PYEOF
echo "================ eval build+plain DONE $(date) ================"
