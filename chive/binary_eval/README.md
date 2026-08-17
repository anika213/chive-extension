# binary_eval — binary causal-claim eval: end-to-end pipeline

This is the runbook that ties the stages together: the **order of operations** + exact
commands. Each stage's details live in its script docstring / sibling README (linked below).

The eval measures whether an interp tool (Activation Oracle / NLA) helps an agent judge *why*
a target model behaved a certain way. From a verified investigation we generate true/false
**claims**; the answerer sees only the transcript + question + one claim and outputs
P(claim true). Scored by AUROC / Brier / acc@0.5.

**Claim shape (cf7).** Each claim names one factor via a single-factor intervention and asks one
**unsigned** thing: does this factor move the behavior rate by ≥30pp, in *either* direction? Rates
are generic `k/n` counts (n=30). There is **no mechanism field** — a claim is just BEHAVIOR +
BASELINE + INTERVENTION + PREDICTION. The label is **computed in code** from the measured rates,
never asserted by a model:

```
is_true = (factor_status == "supported")        # effect_pp = |k1/n1 - k0/n0|*100
  supported (effect ≥ MOVED_PP=50pp) -> TRUE     inert (effect ≤ INERT_PP=15pp) -> FALSE
```

A factor is `supported` iff its measured effect is ≥50pp (`MOVED_PP`) and `inert` iff ≤15pp
(`INERT_PP`); the (15, 50)pp band is a **dead zone** — no claim is built on it. The claim text
asserts the magnitude at `THRESHOLD_PP`=30pp (which sits in the dead zone, so supported answers
"≥30pp" yes and inert answers it no); `THRESHOLD_PP` never enters the label, which is purely the
gate. The wide 15→50 gap deliberately absorbs n=30 measurement noise, so there is **no statistical
certification step**. Every bank row
still stores the raw counts (`k_baseline/n_baseline/k_counterfactual/n_counterfactual` + `effect_pp`)
so any post-hoc analysis stays possible. The thresholds are constants at the top of
`build_claims.py`.

> **Claim format.** Every bank row carries `gen_version: "cf7"`: one unsigned claim per factor —
> no increase/decrease direction and no changes/stays prediction, since signing the claim would
> grade an often-ambiguous sub-skill and AUROC (the primary metric) is bias-independent.
> Earlier-generation `cf5` eval sets still appear in some run dirs as development data.

**Two-stage, blinded generation.** To remove the "did the wording leak the label?" confound,
no outcome-aware model writes claim prose. Code classifies each experiment as supported/inert from
the measured rates; an Opus **selector** (outcome-aware) picks the key supported + plausible inert
single-factor experiments, emitting *only* structured selections; code assigns the label from the
measured effect; a Sonnet **blind writer** (never shown any outcome) describes each intervention from
the original-vs-counterfactual conversation diff and rates `edit_cleanliness` (minor_targeted /
large_single_factor / mixed / confounded — metadata for optional downstream slicing, not a filter).
Because code owns every rate, there is nothing for a model to mis-transcribe, so no
rate-grounding guards or bank-audit step are needed.

Because the claim is unsigned and the prose is blind-written, a TRUE claim (supported factor) and a
FALSE claim (plausible-but-inert factor) are **identical in form** — only the intervention differs —
so there is no polarity, direction, or wording tell for the answerer to exploit.

All paths are relative to the repo root; commands run from there with `.venv` active and
`.env` sourced. `data/scratch/` holds local run outputs (dev banks, server-discovery
jsons); `data/` is gitignored.

---

## The stages

Every stage consumes the previous stage's output **directly** — no hand-built caches, no prep
steps. Each consumer reads the run dir (`investigations.json` + `verification.json`) in memory.

```
(0) investigations run     ── upstream: run_pipeline.py -> a run dir with investigations.json +
        │                      verification.json (verified investigations + experiment records)
        │
(1) build claims           ── build_claims.py --run-dir <run>: TWO-STAGE, blinded.
        │                      Code buckets each experiment by measured effect; an Opus SELECTOR picks
        │                      key supported + plausible inert single-factor experiments
        │                      (structured output only); code assigns the label (supported=TRUE,
        │                      inert=FALSE); a Sonnet BLIND WRITER describes each intervention
        │                      from the conversation diff + rates edit_cleanliness. Per verified>=8
        │                      investigation, read straight from the run dir (in memory). Streams the
        │                      full bank; a .build_meta.json sidecar guards --resume.  [API]
        │   -> <run>/binary_counterfactual_questions.jsonl  (+ _review / _dropped / .build_meta)
        │
(2) score concreteness     ── concreteness.py --run-dir <run>: Opus rates each
        │                      verified>=8 finding 1–4 (mechanism legibility), straight from the run dir.  [API]
        │   -> <run>/mechanism_concreteness.jsonl
        │
(3) build eval set         ── build_eval_set.py --run-dir <run> --base-model <target>: join bank +
        │                      concreteness, filter concreteness>=3, drop unsegmentable transcripts
        │                      (so all arms score the SAME set), balance 50/50 true/false,
        │                      cap at n, seeded.  [CPU]
        │   -> <run>/eval_set_conc3_n1000.jsonl   (the ONE claim set every arm scores)
        │
(4) run eval               ── run_binary_eval.py --bank <eval_set>: plain / ao answerers, scored.  [GPU for ao server; API]
            -> <run>/binary_evals/<tag>_<model>_<effort>_<label>.jsonl   (tag = plain | ao_<lora> | nla_<av> | sae_<slug>)
```

Optional side-paths (not on the critical path): the **NLA/SAE cache precomputes** (only for
`--tool nla` / `--tool sae`): `agent/NLA/jobs/precompute_nla_array.sh` (args: `<bank>
<capture_url> <spec>`, e.g. spec `gemma3-27b-L41`) / `agent/SAE/jobs/precompute_sae_cache.sh`
(args: `<bank> <capture_url> <base_model> --capture-layer 40` — **`--capture-layer` is
required**, it asserts == the SAE's hook layer; see `agent/SAE/README.md`) [GPU] — both read
the bank directly and write checkpoint-named caches into `<run>/nla_cache/` / `<run>/sae_cache/`.
Both capture from a vllm-lens server; **pass `""` for its port to avoid SO_REUSEPORT
collisions** (see `agent/NLA/README.md` capture-server gotcha).

Stages 1, 2, 4 are API-bound (Opus); stage 3 is instant CPU. Run stages 1 and 2 in parallel — both
just read the run dir independently.

---

## Runbook — adding a new target model (e.g. Qwen3-8B)

```bash
RUN=data/runs/<your_run>           # from run_pipeline.py (stage 0)
source .env
# (optional speed-up) default straight to the high-priority Anthropic key + bump concurrency:
export ANTHROPIC_API_KEY_LOW_PRIORITY="$ANTHROPIC_API_KEY_HIGH_PRIORITY"

# 1 + 2 (parallel) — both read the run dir directly, no prep:
python -m chive.binary_eval.build_claims \
    --run-dir $RUN --out $RUN/binary_counterfactual_questions.jsonl --min-score 8 --n 100000 &
python -m chive.binary_eval.concreteness \
    --run-dir $RUN --min-score 8 --out $RUN/mechanism_concreteness.jsonl --concurrency 100 &
wait

# 3 — materialize the canonical eval set (concreteness>=3, segmentable, balanced, capped):
python -m chive.binary_eval.build_eval_set \
    --run-dir $RUN --base-model <target model> --bank $RUN/binary_counterfactual_questions.jsonl \
    --min-concreteness 3 --n 1000 --seed 0
#   -> $RUN/eval_set_conc3_n1000.jsonl

# 4 — start the AO server for the target, then run both arms on the ONE eval set:
sbatch chive/binary_eval/agent/jobs/serve_<model>_ao.sh   # writes ao_server*.json (url/base/ao_loras)
ES=$RUN/eval_set_conc3_n1000.jsonl
python -m chive.binary_eval.run_binary_eval --bank $ES --answerer plain --label conc3
python -m chive.binary_eval.run_binary_eval --bank $ES --answerer agent --tool ao --label conc3 \
    --url http://<node>:<port> --base-model <base> --ao-lora <full HF repo id>
```

Per target you need: a vLLM server for stage-0 generation (`tools/vllm_pool/submit_<model>.sh` — any plain vLLM OpenAI server works),
and an AO server for stage 4 (`agent/jobs/serve_<model>_ao.sh`; can host SEVERAL AO LoRAs at once —
`--ao-lora` picks one per eval run; capture layers are derived from that LoRA's `ao_config.json`,
see `agent/ao_tool.resolve_capture_layers`). The AO is base-model-specific — its base MUST match
the run's target model.

---

## Stage 1 — build the cf7 claim bank  (CPU / API)

`build_claims.py --run-dir <run>` loads the `verification≥8` investigations
**straight from the run dir** (`investigations.json` + `verification.json`, in memory — no cache),
each carrying its experiment record (`experiment_summaries` rates + `experiment_log` full
conversations) and `classifier_question`. It is **two-stage and blinded** (see "Two-stage, blinded
generation" above): code buckets each experiment by measured effect (supported ≥50pp / inert ≤15pp,
dead zone excluded); an Opus **selector** picks the key single-factor experiments (structured output
only — it never reports a rate or assigns a label); code assigns the label from the measured effect
(supported=TRUE, inert=FALSE), one unsigned claim per factor; a Sonnet **blind
writer** writes each intervention from the original-vs-counterfactual conversation diff plus an
`edit_cleanliness` rating. Because code reads every rate straight from the record, there are no
rate-grounding guards. Each investigation's claims are **streamed** to the bank as they complete
(crash-safe; resumable — a `.build_meta.json` sidecar records run_dir/model/writer_model/min_score/
n/seed and `--resume` refuses unless they match exactly). The build emits only the full bank —
balance/cap is a downstream concern (Stage 3).

**Full scale-up run** (all `verification≥8`, output into the run dir):
```bash
python -m chive.binary_eval.build_claims \
    --run-dir data/runs/<run> \
    --out data/runs/<run>/binary_counterfactual_questions.jsonl \
    --min-score 8 --n 100000 --concurrency 100
```

**Dev run** (small sample → `data/scratch/binary/<bank>/`):
```bash
python -m chive.binary_eval.build_claims --run-dir <run> --n 10 --seed 0 --bank-version cf7_dev
```

**Target a fixed investigation set** (e.g. a gold set): `--prompt-ids-file <dict.json>` (keys
= prompt_ids; bypasses `--n/--seed/--min-*`).

Key flags:
- `--run-dir` and `--n` are REQUIRED (no defaults — a hardcoded default run dir + a silent
  `--n 10` each caused real near-misses; pass `--n 100000` for "all").
- `--out PATH` (canonical/full) **xor** `--bank-version NAME` (→ `data/scratch/binary/<NAME>/`).
- `--min-score 8` gates which investigations are used; claims are built for all `verified≥8` investigations (no interestingness/concreteness pre-filter — mechanism concreteness is scored downstream, Stage 2).
- `--resume` — skip prompt_ids already in the output bank and append the rest (recover API/rate-limit failures or continue after a crash). Guarded by the `.build_meta.json` sidecar: settings must match the original build exactly.
- **No balance knob.** Balance + concreteness-filter + cap are a downstream/eval-time concern,
  done by `build_eval_set.py` (Stage 3); the build keeps the full bank.
- **Concurrency:** with the high-priority key as default (`export
  ANTHROPIC_API_KEY_LOW_PRIORITY="$ANTHROPIC_API_KEY_HIGH_PRIORITY"`), 100 runs clean (0×529).
  On the low-priority key default, keep ~40 (sustained high concurrency triggers 529s →
  "no gen tool call"; re-run with `--resume` to recover).

Outputs (for `--out <stem>.jsonl`; `--bank-version` writes the same names under `data/scratch/binary/<bank>/`):
- `<stem>.jsonl` — **the full bank**, one self-contained claim per line (transcript, question,
  behavior_question, intervention, statement, is_true, claim_type=`{factor_status}`,
  edit_cleanliness, rates + integer counts, grounding=experiment summary, raw messages +
  chosen_completion). `run_binary_eval.load_bank`-compatible. The NLA/SAE precomputes read this file
  directly (deduped by prompt_id) — no separate precompute input.
- `<stem>.build_meta.json` — build-settings sidecar; asserted on `--resume`.
- `<stem>_review.md` / `<stem>_dropped.md` — manual-review doc + log of the rare dropped selector
  picks. Written on fresh runs only (a `--resume` doesn't clobber them).

Design rationale: **`build_claims.py`** docstring. **Eyeball `_review.md` / a few
bank rows before spending GPU on Stage 3.**

> **Removed:** earlier banks had a Stage 1b statistical certification (Beta-posterior
> moved/inert/band, `certify_labels.py`). With the wide cf7 dead zone (15–50pp) at n=30 it is no
> longer needed; that machinery has been deleted and the bank's labels are final.

---

## Stage 2 — mechanism-concreteness scoring  (downstream, CPU/API)

`concreteness.py --run-dir <run>` (Opus 4.8) scores each
`verification≥8` finding on a single 1–4 **mechanism_concreteness** axis (is the causal story
human-legible vs statistical/lexical): 1 = purely statistical (no interpretable mechanism),
2 = plausible but not crisp, 3 = understandable conceptual mechanism, 4 = concrete and
unambiguous. It emits multi-label category tags + `is_mistake` in the same call (the tag
taxonomy is defined in `concreteness.py`'s prompt; nothing downstream consumes the tags). It
reads the run dir **directly** (same in-memory entry load as Stage 1) → `<run>/mechanism_concreteness.jsonl`.
Runs independently of Stage 1 (do them in parallel). Stage 3 (`build_eval_set.py`) joins it on `prompt_id`.

```bash
python -m chive.binary_eval.concreteness \
    --run-dir <run> --min-score 8 --out <run>/mechanism_concreteness.jsonl --concurrency 100
```

The script's other mode, `--gold/--cache <dict.json>`, is the scorer's **validation** harness —
not used for scoring a run. The scorer was validated against a held-out set of hand-endorsed
labels (not distributed); the classifier prompt contains no held-out finding — the rubric is
general principles only, and every few-shot example was verified to be absent from the gold
set (see the leakage-discipline note in `concreteness.py`'s docstring).

## Stage 3 — build the eval set  (CPU)

`build_eval_set.py --run-dir <run> --base-model <target> --min-concreteness 3 --n 1000`
joins the full bank with `mechanism_concreteness.jsonl`, filters to concreteness≥3 (verification≥8
already guaranteed by the build), drops claims whose transcript fails `segment_transcript` with the
target tokenizer (the agent arms could not score them — filtering here keeps every arm on the SAME
set), balances 50/50 true/false (supported vs inert — cf7 claims are unsigned and identical in form,
so there is no polarity to balance within), caps at `--n` (n//2 per label), and writes
`<run>/eval_set_conc<min>_n<n>.jsonl`. This is the **single, deterministic, auditable** definition of
which claims get scored — every arm points `--bank` at this one file, so they score the identical
set (no per-arm on-the-fly balancing that could drift).

---

## Optional — precompute the NLA cache  (GPU, parallelized; only for `--tool nla`)

NLA verbalization is **unconditional and per-token** (claim-independent), so it's computed
once for every token position in every transcript and cached; the agent's NLA tool is then
a dict lookup. Mechanism + the `async_scheduling=False` requirement (root-caused vLLM < 0.22
`<<<<` corruption — no more decode-batch capping):
**`agent/NLA/precompute_nla_cache.py`** docstring and **`agent/NLA/README.md`**.

The precompute reads the **full bank directly** (rows deduped by prompt_id — no separate
input file) and decodes SAMPLED at temp 1.0 (the in-distribution setting; recorded per row).
Caches are run-local and checkpoint-named: `<bank_dir>/nla_cache/<av_repo_basename>.jsonl`
(run identity = the directory, checkpoint identity = the filename).

**Architecture:** ONE shared AO capture server (base model + AO LoRA, vllm-lens; cheap
`max_tokens=1` residual-stream read) + N decode workers (each loads the AV model
**offline**, decodes its shard in batches of 100). Capture is ~30× cheaper than decode, so
one server feeds many workers; the offline+batch-100 recipe avoids the `<<<<` bug.

### 3a. Launch the shared AO capture server, wait until it serves
```bash
sbatch chive/binary_eval/agent/jobs/serve_gemma_ao.sh   # -> node-XX:8414
curl -s http://node-XX:8414/v1/models     # ready when it returns JSON with "id"
```
On startup the job writes its connection info (url / base model / hosted LoRAs) to
`data/scratch/ao_server.json`.

### 3b. Launch the decode workers as a Slurm array (sharding is internal)
```bash
sbatch chive/binary_eval/agent/NLA/jobs/precompute_nla_array.sh \
    <run>/binary_counterfactual_questions.jsonl http://node-XX:8414 \
    <base model> <av repo, e.g. kitft/nla-gemma3-27b-L41-av>
```
5 tasks (`--array=0-4`, must match the script's `NUM_SHARDS`), 1 GPU/task, `--time=6:00:00`,
≈10–16 decodes/s/worker. Each task writes `<run>/nla_cache/shards/<av>_shard{i}of5.jsonl`.
Pre-submit hygiene (see `docs/infra.md`): run `jobs/find_zombie_gpu_processes.sh
--exclude` and pass `--exclude=` to `sbatch`.

### 3c. Combine the shard caches
```bash
cat <run>/nla_cache/shards/<av>_shard*of5.jsonl > <run>/nla_cache/<av>.jsonl
wc -l <run>/nla_cache/<av>.jsonl     # = number of bank investigations (minus any logged skips)
scancel -u $USER --name=vlens_gemma_ao
```
Then load it once (`nla_tool.load_nla_cache`) — it asserts full coverage / no duplicate
prompt_ids and warns if the degenerate-decode rate (truncated / missing tags) exceeds 0.1%.

---

## Stage 4 — run the eval  (API for agents; GPU only for the AO server)

`run_binary_eval.py` runs one answerer over the **Stage-3 eval set** and writes scored results to
`<run>/binary_evals/` (next to the bank). Both arms point `--bank` at the same
`eval_set_conc3_n1000.jsonl` so they score identical claims. Streaming + `--resume` are supported
(skip already-scored claim_ids).

```bash
ES=<run>/eval_set_conc3_n1000.jsonl

# plain (transcript-only ceiling; no tools, no GPU)
python -m chive.binary_eval.run_binary_eval --bank $ES --answerer plain --label conc3

# AO agent — --ao-lora (full HF repo id) is ALWAYS required: which AO checkpoint to query is
# the experiment variable, and the server may host several (e.g. the qwen server hosts the
# single-layer AND multi-layer Qwen3-8B AOs). Capture layers are derived from the LoRA's
# ao_config.json. Pass explicit --url/--base-model from the serve job's ao_server*.json so
# you never depend on / clobber another eval's shared ao_server.json:
python -m chive.binary_eval.run_binary_eval --bank $ES --answerer agent --tool ao --label conc3 \
    --url http://node-XX:8413 --base-model Qwen/Qwen3-8B \
    --ao-lora adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
# multi-layer AO: --ao-lora adamkarvonen/checkpoints_500k_pl_31k_spqav2_199k_sqav3_50k_hb_126k_cls
# (without --url/--base-model it falls back to data/scratch/ao_server.json, written by serve_gemma_ao.sh.)

# NLA agent (cache-only; no GPU at eval time; requires the optional NLA precompute above).
# --nla-cache defaults to the single cache in <bank_dir>/nla_cache/ (the eval set lives next
# to the bank, so the bank's cache is found automatically; several caches -> flag required).
python -m chive.binary_eval.run_binary_eval --bank $ES --answerer agent --tool nla --base-model <base>

# SAE agent (cache-only; no GPU at eval time). Cache + Neuronpedia labels are
# precomputed by agent/SAE/ (see agent/SAE/README.md); --sae-cache defaults like
# --nla-cache (<bank_dir>/sae_cache/), --sae-labels is derived from the SAE in the cache.
python -m chive.binary_eval.run_binary_eval --bank $ES \
    --answerer agent --tool sae --base-model google/gemma-3-27b-it
```

The answerer model + reasoning effort are hardcoded in `run_binary_eval.py`
(`ANSWERER_MODEL` / `ANSWERER_EFFORT`, shared by all answerers so the tool comparison stays
clean) and baked into the output name, along with the AO/NLA checkpoint identity:
`plain_<model>_<effort>[_<label>].jsonl`, `ao_<lora_name>_<model>_<effort>[_<label>].jsonl`,
`nla_<av_repo_name>_<model>_<effort>[_<label>].jsonl`,
`sae_<sae_slug>_<model>_<effort>[_<label>].jsonl`, written to `<bank_dir>/binary_evals/`
(alongside the run logs) — so changing model/effort/checkpoint can never silently
`--resume`-mix results from different settings into one file. Each row records the exact prompt sent, prob_true, correctness, full
reasoning, stop_reason, + the agent tool-call transcript for ao/nla. Metrics print to stdout
(AUROC / Brier / acc@0.5, per-(claim_type, label) calibration, and the forced-submit rate —
nonzero means the agent ran out of tool budget before answering). **Cost note:** ao/nla are
agentic (up to 5 interp reads each) — confirm the spend before launching; for a quick read,
cap the bank with `--max-claims`. Answerer/scoring details: **`run_binary_eval.py`** docstring.

---

## Reference: per-stage docs
| stage | code | doc |
|---|---|---|
| 1 build | `build_claims.py` | its docstring |
| 2 concreteness | `concreteness.py` | its docstring + prompt (scale + category taxonomy) |
| 3 eval set | `binary_eval/build_eval_set.py` | its docstring |
| 4 eval | `binary_eval/run_binary_eval.py` | its docstring |
| NLA precompute (optional) | `agent/NLA/precompute_nla_cache.py` | `agent/NLA/README.md` |

Canonical banks (all **cf7**, n=30 runs):
- gemma: `data/runs/opus_investigates_gemma3_27b_it_5000_thorough_n30/`
- Qwen3-8B: `data/runs/opus_investigates_qwen3_8b_5000_thorough_n30/`
- Qwen3.5-397B: `data/runs/opus_investigates_qwen3p5_397b_5000_thorough_n30/` and `..._petri_thorough_n30/`
  (plain + cf arms only; cf runs the 397B target via Tinker — `run_binary_eval --tool cf
  --cf-target tinker:Qwen/Qwen3.5-397B-A17B`)

> **History.** Each format change archives the prior bank in-place (`archive_cf4/`, `archive_cf5/`,
> `archive_cf6/`) and is regenerated. cf7 (current) = one unsigned claim per factor; cf6 added the
> verbatim behavior_question + percentage baseline but kept a changes/stays prediction + direction;
> cf5 was two-stage blinded; cf4 had a mechanism field + statistical certification. The older
> non-`_n30` dirs are superseded.
