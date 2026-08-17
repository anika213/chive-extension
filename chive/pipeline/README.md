# Model Understanding Pipeline Guide

## What this pipeline does

We find and explain surprising behaviors in a target LLM. The end product is a dataset of verified causal explanations: "the model does X because of prompt feature Y, and here are the counterfactual experiments proving it." These can be used as training data for self-explanation and as evaluations for interpretability methods.

**Backends are fully pluggable.** Every model-facing stage (1 = generation,
2 = screening, 3 = investigation agent + counterfactual target, 4 = verification)
takes an `llm_client` spec, so any role can be:
- a **cloud model** — `anthropic:claude-opus-4-6`
- a **vLLM server** — `vllm:Qwen/Qwen3.5-397B-A17B-FP8@http://node:8000`
- an **OpenRouter model** — `openrouter:Qwen/Qwen3.5-397B-A17B`

Mix freely: e.g. Opus as the investigator/judge with a Qwen vLLM target (the
Opus→Qwen pipeline), or the same Qwen3.5-397B spec for *all* roles
(self-investigation). Use the **HF-canonical model name** in the spec — it drives
both the tokenizer and the OpenRouter/vLLM endpoint, so there's one source of
truth and no name drift. **Models run so far:** Qwen3-8B, Qwen3-32B,
Qwen3.5-397B-A17B (FP8).

Source data is WildChat-4.8M — real user conversations with LLMs. We use both single-turn prompts and multi-turn conversations (the model sees prior turns as context and generates a response to the final user message). Because WildChat's parquet shards are time-ordered and ~half the corpus is automated/template API traffic (Midjourney generators, data-labeling/recruiter bots), Stage 0 samples across evenly-spaced shards (time-spread) and applies a per-prefix template cap.

## Pipeline stages in detail

### Stage 0: Prompt preparation

**Canonical prompt pool (use this going forward):**
`data/runs/wildchat_plus_agentic/prompts.jsonl` — **189,542** prompts: the 172,332
cleaned WildChat prompts **plus ~9.1% (17,210) agentic / system-prompt prompts**
shuffled in. This is the single source new runs should draw from: index into the
first N lines for a run (the file is shuffled, so the first N is a representative
random sample) and feed those prompts into Stage 1.

- One JSON object per line (head-able), each
  `{id, source, conversation_hash, user_message, messages, n_turns, total_chars}`.
  `source` is `wildchat`, `hermes`, `toolace`, or `systemchat2`.
- **Untokenized by design** (`prompt_ids` omitted) so the *same* pool works for
  any model — Stage 1 sends `messages` straight to a chat-completions backend
  (OpenRouter / vLLM server), which tokenizes server-side. See the
  HF-canonical-name convention: one model string drives both tokenizer and endpoint.

**The agentic / system-prompt slice (~10%).** Real chat-UI logs (WildChat, LMSYS)
carry **no** system prompts, so to make the model attend to system prompts and
tool definitions we mix in three curated sources, each as a record of the form
`messages = [{role: system}, {role: user}]` (ends at the user turn, exactly like a
WildChat single-turn prompt). The system message carries the persona / behavioral
instructions and, for the tool datasets, the **tool definitions as TEXT** inside
the system prompt.

- **Why tools-as-text and not structured `tools=`:** verified against the live
  vLLM server — if you pass structured `tools=` with the parser on, the model's
  tool call comes back with **empty `content`** (Stage 1 drops empty completions
  and fails past 2%). With the tool spec as text in the system message, the model
  emits its `<tool_call>{...}</tool_call>` as **plain text in the completion**, so
  the whole pipeline handles these with **no code changes** (Stage 1 forwards
  `messages`; stages 2-4 already render any role as `[ROLE]:`).
- **Sources** (all length-filtered to 400–8000 total chars, like WildChat):
  - `hermes` (2,236): NousResearch/hermes-function-calling-v1 (`func_calling_singleturn`
    + `func_calling`) — rich, realistic multi-step tool tasks.
  - `toolace` (10,547): Team-ACE/ToolACE — kept **only** when the system prompt's
    tool list is a parseable JSON array (HTML-`<table>` / bespoke-format rows skipped).
  - `systemchat2` (4,427): cognitivecomputations/SystemChat-2.0 — diverse
    behavioral / persona system prompts (often *conflicting* with the request,
    which surfaces interesting behavior).

**The WildChat base pool** (`data/runs/wildchat48m_clean_172k/prompts.jsonl`, 172,332; not shipped in this release — rebuild with `build_prompt_pool.py` below, or use the canonical merged pool above):
built by draining all 86 WildChat-4.8M shards with shard-spread (time-spread)
sampling + a global cap-5-per-100-char-prefix template dedup, which strips the
bot/template duplication (Midjourney generators, labeling/recruiter bots) that
dominated earlier pools. Filters: English, non-redacted, 400–8000 total chars,
≤5 turns. 172k is the measured clean ceiling at these settings (535,744 dropped by
the prefix cap across 137,781 distinct prefixes); going higher means loosening the
char filters or supplementing with another source rather than raising the cap.

**Per-run Stage 0 — slice the pool (`run_pipeline.run_stage0`):** reads the first
`stage1_n_prompts` rows (after `stage1_offset`) from `prompt_pool`, optionally
drops any whose `conversation_hash`/content-hash appear in `exclude_files` (prior
runs), and writes `run_dir/prompts.jsonl`. No streaming, no tokenization, no
model. Resumable.

**Rebuilding the canonical pool — two steps (CPU-only, run rarely).** The build
is split into two focused scripts that share one record schema + JSONL I/O
(`pool_common.py`, the single source of truth for both). Run them in order:

1. `build_prompt_pool.py` → the WildChat base (`wildchat48m_clean_172k/prompts.jsonl`).
   Heavy/slow (streams all 86 shards); the 172k base is stable, so this is rare.
2. `build_agentic_pool.py --wildchat <base> --out-dir <canonical>` → builds the
   agentic per-source JSONLs and shuffles them with the base into the canonical
   `wildchat_plus_agentic/prompts.jsonl`. Fast; re-run whenever refreshing the mix.

**Step 1 — `build_prompt_pool.py`:**
streams WildChat-4.8M, applies all filters + template dedup, writes one JSONL.

- Streams conversations from `allenai/WildChat-4.8M` (English, non-redacted)
- **Shard-spread sampling, target-driven:** pulls a per-shard quota across evenly-spaced parquet shards (default: ALL shards) sized so the quotas sum to ~`--n-prompts` (x an overshoot). Spans the full time range AND stops at the target. Lower `--n-sample-shards` only for fast dev runs.
- **Template dedup:** global cap of `--prefix-cap` (default 5) per normalized `--prefix-chars` (default 100) prefix, collapsing bot/template families
- Filters by total character count across all turns; includes single- and multi-turn (truncated to end with a user message); drops NAME_ placeholders
- Stores `messages` only — **no tokenization** (model-agnostic). The char filter bounds length, so there's no token cap.
- **`--exclude`** one or more completions JSON / prompts JSONL files to dedup against (by `conversation_hash` + content hash).

```bash
.venv/bin/python chive/pipeline/build_prompt_pool.py \
    --n-prompts 200000 --min-chars 400 --max-chars 8000 --max-turns 5 --prefix-cap 5 \
    --jsonl-out data/runs/wildchat48m_clean_172k/prompts.jsonl
```

**Output (pool):** `prompts.jsonl`, one record per line: `{id, conversation_hash, user_message, messages, n_turns, total_chars}`.

**Step 2 — `build_agentic_pool.py`** (CPU-only):
builds the three per-source agentic JSONLs (`hermes.jsonl`, `toolace.jsonl`,
`systemchat2.jsonl`) and shuffles them together with the WildChat base pool into
the canonical `data/runs/wildchat_plus_agentic/prompts.jsonl`. ToolACE is filtered to
rows whose tool list is a parseable JSON array. Set `HF_HOME` first.

```bash
export HF_HOME=/workspace-vast/$USER/hf_cache
.venv/bin/python chive/pipeline/build_agentic_pool.py \
    --wildchat data/runs/wildchat48m_clean_172k/prompts.jsonl \
    --out-dir  data/runs/wildchat_plus_agentic
```

> Note: pool JSONLs are read with `pool_common.read_jsonl` (`split("\n")`, not
> `str.splitlines()`), because some prompt text contains literal Unicode line
> separators (U+2028/U+2029, NEL, etc.) that `splitlines()` would split
> mid-record. Use `read_jsonl`/`write_jsonl` from `pool_common` for any new reader.

### Stage 1: Generate completions

**Script:** `generate_completions.py` (also called in-process by `run_pipeline.run_stage1`).
**What it does:** Reads `run_dir/prompts.jsonl` and generates **`n_completions` independent temperature-1 completions of 500 tokens** per prompt (count: the single `n_completions` config knob, shared with stage 3's `run_prompt` so the two cannot drift — 10 for training-data runs, 30 for held-out test runs; length/temperature: fixed constants `MAX_RESPONSE_TOKENS` / `COMPLETION_TEMPERATURE` in `generate_completions.py`, deliberately NOT config knobs since they define the behavior distribution every downstream stage reads), thinking disabled, via an `llm_client` backend selected by `stage1_llm` — the **same spec form as stages 2-4**. No offline-vLLM/GPU jobs, no shards.

- **Backend-aware fan-out:** vLLM HTTP servers use one request with `n=n_completions` (shared prefill); other backends (e.g. OpenRouter, which returns a single choice for the Qwen route) fan out independent `n=1` calls.
- A bare `vllm:MODEL` (no `@url`) is resolved against the running vLLM server found by `vllm_job_name`.
- Per-completion `finish_reasons` are recorded and the run reports `truncated_rate` (`finish_reason == "length"`).
- For multi-turn prompts the full conversation history is sent; completions are for the final assistant turn.

**Output:** `completions.jsonl` — one `PromptCompletions` record per line (`{id, user_message, messages, n_turns, total_chars, completions: [str], finish_reasons, ...}`), written incrementally (it is also the resume file). Run metadata (model spec, knobs) lives in the run's `config.json`; counts and `truncated_rate` in `status.json`.

### Stage 2: Screening

**Script:** `screen_completions.py`
**What it does:** the judge model (`judge_llm`; Opus with extended thinking in our canonical runs) reads each prompt + all completions and judges whether the behavior is interesting enough to investigate.

For each prompt, the judge outputs:
- `interest_score` (1-5): 1=routine, 3=non-obvious with 2+ hypotheses, 5=deeply puzzling
- `behavior_summary`: what the model does across completions (e.g., "7/10 completions claim X despite evidence of Y")
- `question`: an investigable question about why ("Why does the model do X?")
- `second_person_question`: same question in second person ("Why did you do X?") for training data
- `classifier_question`: a binary yes/no question operationalizing the behavior, answerable from a single completion. This is the investigation's FIXED measurement instrument: the stage-3 grader applies it to every completion of every experiment, so the screener must resolve edge cases (do variants count?) and avoid hard-coding incidental entity names in it
- `hypotheses`: 2-4 plausible competing explanations
- `counterfactual`: a concrete experiment to distinguish hypotheses
- `justification`: why this score was given

Only prompts scoring >= 3 proceed to investigation.

Two API modes:
- **Async:** Concurrent API calls (configurable concurrency). Good for small/medium runs.
- **Batch:** Anthropic batch API. Cheaper, higher throughput for large runs. Chunks into batch_chunk_size requests.

Both modes use the SAME judge configuration (extended thinking per
`judge_thinking_budget`, no forced tool choice), so scores are comparable
across modes.

For multi-turn prompts, the screening prompt shows the full conversation history labeled as `[USER]`/`[ASSISTANT]` turns, then the final user message, then all completions.

**What gets filtered out (score 1-2):**
- Hallucinations (small model, expected)
- Uniform safety refusals (10/10 refuse) — split behavior IS interesting
- Single obvious keyword triggers
- Generic sycophancy/politeness
- Straightforward task execution with no variation

**Bottleneck:** API call latency × number of prompts. Batch API handles large runs efficiently.

**Output:** `screening.jsonl` — one `ScreeningResult` per line, written incrementally (also the resume file) — plus a human-readable `top_results` .txt dump of the score >= 4 entries.

**Throughput:** Batch API processes 10k+ prompts in a few hours. Async at concurrency 40: ~200 prompts screened per minute.

**Pass rate:** ~55% of prompts score >= 3 (measured at 40k scale).

### Stage 3: Investigation (THE BOTTLENECK)

**Script:** `investigate.py`
**What it does:** An Opus agent runs counterfactual experiments to answer the screening question. It has two tools:

1. `run_prompt(user_message=..., turns=..., n=n_completions, temperature=1.0)`: Sends a modified prompt to the vLLM server and gets back `n_completions` completions. The agent sees the first 10 in full; an independent **grader** (`grader_llm`, Sonnet 4.6) answers the investigation's fixed `classifier_question` against ALL of them (per-completion verdicts via tool call, counted in code) and the count is appended to the tool result. The agent is instructed to use grader counts for every rate it reports. For multi-turn, accepts a `turns` list so the agent can modify any conversation turn.

2. `submit_findings(question, baseline_behavior, structured_answer, experiment_summaries, commentary, confidence)`: Submits structured results at the end.

The agent loop:
1. Agent sees the original prompt + the first 10 completions + the grader's baseline count over all of them (the stage-1 completions are graded with the same fixed question at investigation startup) + the screening question + hypotheses
2. Agent designs and runs counterfactual experiments (typically 8-15 per investigation)
3. Each experiment: modify one aspect of the prompt, send to vLLM, observe how behavior changes
4. Agent identifies the causal factor(s) and submits structured findings
5. Max `stage3_max_agent_turns` turns (default 10 — each turn cycle adds ~10k
   tokens of conversation history, so higher caps risk context-window overflow.
   For held-out / evaluation-data runs, bump it (e.g. 30) together with
   `stage3_thorough: true`, which switches the methodology from "efficient,
   5-8 experiments" to "rule out competing hypotheses, ~10-15 experiments")

For multi-turn prompts, the system prompt instructs the agent to modify any turn — user messages, assistant responses, or both. The agent uses the `messages` parameter (list of {role, content} objects) to construct full modified conversations.

The `answer` field contains only claims verifiable from the experiments. Speculation about training data, RLHF effects, etc. goes in the separate `commentary` field.

**Key output fields:**
- `structured_answer`: first-person causal report (Observed behavior / Core causes / Negative evidence / Final answer), percentages for rates. Only experiment-verified claims.
- `experiment_summaries`: one entry per experiment with what changed, baseline rate, counterfactual rate (copied from grader counts; a loud warning fires if the prose diverges from the log), and `is_key` for the decisive ones
- `commentary`: Speculation beyond what experiments verify (training data hypotheses, etc.)
- `experiment_log`: Every vLLM call with exact prompt/messages + all completions + `grader_verdicts`/`grader_yes` (the numeric source of truth for downstream significance filtering)
- `classifier_question`, `baseline_grader_verdicts`, `baseline_grader_yes`: the fixed measurement instrument and the baseline measured with it

**Error handling:** If an investigation fails (malformed tool call, API error), it gets one retry; a second failure is skipped with a count and the full traceback is appended to `stage3_quarantine.jsonl` in the run dir. The pipeline continues with remaining investigations. Skip count is tracked in progress output and status.json.

**Bottleneck:** This is where ~90% of the pipeline time goes. Each investigation involves multiple serial Opus API calls (each waiting for the previous response), plus vLLM calls. At concurrency N, N investigations run in parallel.

**Output:** `investigations.jsonl` — one `InvestigationResult` per line, written incrementally (also the resume file) — plus `transcripts/` (full agent message histories) and `stage3_quarantine.jsonl` (skipped items with tracebacks).

**Throughput:** Scales sublinearly with concurrency — at high concurrency, the vLLM server or API latency can become the bottleneck. With 3 GPUs (data-parallel, FP8, max-model-len 6000) and concurrency 250-500, measured ~1,200-1,300 investigations/hr. The pipeline logs `[rate]` lines every 60s showing input/output tokens per minute.

**Token usage per investigation (measured, 22k investigations):**
- Non-cached input: ~80k tokens (input + cache_create)
- Cache read: ~182k tokens (~70% cache hit rate)
- Output: ~8.5k tokens (includes thinking)
- vLLM calls: ~7.6 per investigation

### Stage 4: Verification

**Script:** `verify_investigations.py`
**What it does:** An independent Opus judge scores the investigation quality. It sees:
- The claimed question, the grader's fixed `classifier_question`, baseline behavior, and causal answer
- The original prompt + the first 10 completions with the grader's per-completion verdicts + the grader's baseline count
- 3-5 key experiments with the actual prompts sent, the first 10 completions (verdict-marked), and the grader's count over all completions

The judge checks (rates come from the automated grader and are cross-checked mechanically, so it is told NOT to re-count completions):
0. **Fair operationalization:** Is the fixed `classifier_question` a fair measure of what the report claims — not broader, not narrower? (the one measurement check that matters)
1. **Change description accuracy:** Does "what changed" match the actual prompt difference?
2. **Relevance to causal claim:** Does this experiment actually test the stated cause, isolating one factor (no confounds)?

For multi-turn experiments, the judge sees the full conversation that was sent (all `[USER]`/`[ASSISTANT]` turns), not just the last user message.

Score guide:
- 1-2: Fundamentally unsupported — experiments irrelevant or contradict the answer
- 3-4: Major gaps — key experiments don't isolate claimed factors, or the fixed question measures something materially different from the report's claims
- 5-6: Moderate issues — most experiments relevant but gaps remain
- 7-8: Minor issues — all experiments relevant
- 9-10: Fully verified — fair instrument, experiments directly test the causal explanation

Two API modes: async (concurrency) and batch (for large runs). Both use the
SAME judge configuration — extended thinking per `judge_thinking_budget`, no
forced tool choice — so scores are comparable across modes.

**Bottleneck:** Not significant. Batch API handles thousands of verifications in hours.

**Output:** `verification.jsonl` — one `VerificationResult` per line, written incrementally (also the resume file).

**Score distribution (from qwen3_32b_run, 22,020 verifications):**
- Mean: 7.36/10
- Score >= 7: 78%
- Score >= 8: 44%

## End-to-end pipeline yield

Starting from raw WildChat prompts (measured from qwen3_32b_run, 40k prompts):

| Step | What happens | Pass rate | Cumulative |
|------|-------------|-----------|------------|
| Char filter | 500-5000 total chars, <= 5 turns | ~30-40% of WildChat | ~35% |
| Stage 2 | Opus scores behavior 1-5, need >= 3 | ~55% of screened | ~19% |
| Stage 3 | Opus investigates, produces findings | ~99.9% succeed (few skips) | ~19% |
| Stage 4 >= 8 | High quality, verified | ~44% of investigated | ~8% |
| Stage 4 >= 7 | Usable quality | ~78% of investigated | ~15% |

**From 40,000 stage 1 prompts, expect ~9,500 high-quality data points (score >= 8) or ~17,000 usable data points (score >= 7).**

## Backends

Every model-facing stage picks its backend with a **spec string**
(`make_llm_client`), one per role:

| Config key | Role |
|---|---|
| `stage1_llm` | Stage 1 completion generation (the *target* model) |
| `target_llm` | Stage 3 `run_prompt` counterfactual target (the model being investigated) |
| `stage3_llm` | Stage 3 investigator *agent* |
| `judge_llm` | Stage 2 screening **and** Stage 4 verification judge |
| `grader_llm` | Stage 3 behavior grader (applies `classifier_question` per completion; default `anthropic:claude-sonnet-4-6`) |
| `stage5_llm` | Stage 5 synthesis |

These are all **required and explicit** (no bare-model-name default that could
silently decide which model judges/investigates). Spec forms:

- `anthropic:claude-opus-4-6` — native Anthropic API. Real thinking-budget cap.
  The only backend that supports **batch mode** (`stage2_mode`/`stage4_mode`/
  `stage5_mode: "batch"`).
- `openrouter:qwen/qwen3.5-397b-a17b`, `openrouter:anthropic/claude-haiku-4.5` —
  OpenRouter. Good for iteration. `thinking_budget` maps to `reasoning.max_tokens`.
  The Qwen-FP8 provider filter (spread across FP8 stacks) is applied **only to
  Qwen** models, so `openrouter:anthropic/claude-*` routes normally — you can mix
  Qwen + Claude on this one backend (used by judge/investigator sweeps). Routes
  across several third-party stacks, so not canonical for training data.
- `vllm:Qwen/Qwen3.5-397B-A17B-FP8@http://node-26:8210` — a vLLM HTTP server
  (`MODEL@BASE_URL`). Native `n>1` with a single shared prefill (Stage 1 + Stage 3
  target generate all `n_completions` completions in one request). Server-side prefix caching
  if launched with `--enable-prefix-caching`. `thinking_budget != None` enables
  Qwen thinking (the *value* isn't a hard cap on vLLM — `max_tokens` bounds it;
  `None` disables thinking). Set the server's `max_model_len` high enough for
  Stage 4 (verification prompts are large, ~tens of k tokens).

**Concurrency** is one knob, `concurrency`: the per-endpoint in-flight cap,
enforced at the leaf (`LLMClient.call`) and **shared by every stage hitting the
same base URL** (so the agent + target + verifier on one endpoint share it, and
nested fan-out never multiplies past it). vLLM: set it high (e.g. 1000) and let
the server queue. OpenRouter: ~100-200.

Common setups:
- **Self-investigation (Qwen explains Qwen):** point every `*_llm` at the same
  Qwen spec (`vllm:` for canonical runs, `openrouter:qwen/...` for iteration).
- **Opus → Qwen:** `anthropic:claude-opus-4-6` for `judge_llm`/`stage3_llm`, a
  Qwen `vllm:`/`openrouter:` spec for `stage1_llm`/`target_llm`.
- **Sweeps:** vary one role (e.g. `stage3_llm`) across models while holding the
  rest constant — see *Sharing stages across runs* below.

## How to run

```bash
# 1. Make sure .env has the right ANTHROPIC_API_KEY / OPENROUTER_API_KEY
source .env

# 2. Make sure a vLLM server is running (needed for stage 3)
#    Check: squeue -u $USER
#    Start if needed: sbatch --time=72:00:00 chive/pipeline/slurm_scripts/submit_vllm_server.sh

# 3. Run the pipeline
.venv/bin/python chive/pipeline/run_pipeline.py \
    --config data/runs/<run_name>/config.json
```

### Config format

The config JSON overrides values from the `DEFAULTS` dict in
`chive/pipeline/config.py` — the single source of truth for every knob (the
standalone stage CLIs take their argparse defaults from it too). Unknown keys
fail loudly at load time, so a misspelled knob can't silently fall back to its
default; prefix free-text comments with `_` (e.g. `"_comment"`). Example:

All stages select their backend with the same `*_llm` spec form
(`make_llm_client`): `vllm:MODEL@http://node:8000`, `openrouter:MODEL`, or
`anthropic:MODEL`. For self-investigation, point them all at the same Qwen spec;
for the Opus→Qwen pipeline, use `anthropic:claude-opus-4-6` for the judge/agent
and a `vllm:`/`openrouter:` Qwen spec for `stage1_llm`/`target_llm`.

```json
{
  "run_name": "qwen3p5_397b_self_run",
  "concurrency": 1000,
  "prompt_pool": "data/runs/wildchat_plus_agentic/prompts.jsonl",
  "stage1_llm": "vllm:Qwen/Qwen3.5-397B-A17B-FP8@http://node:8000",
  "stage1_n_prompts": 40000,
  "stage1_offset": 0,
  "n_completions": 10,
  "grader_llm": "anthropic:claude-sonnet-4-6",
  "stage2_n": 40000,
  "judge_llm": "vllm:Qwen/Qwen3.5-397B-A17B-FP8@http://node:8000",
  "judge_max_tokens": 8192,
  "judge_thinking_budget": 4096,
  "stage3_min_score": 3,
  "stage3_llm": "vllm:Qwen/Qwen3.5-397B-A17B-FP8@http://node:8000",
  "target_llm": "vllm:Qwen/Qwen3.5-397B-A17B-FP8@http://node:8000",
  "stage3_max_tokens": 16384,
  "stage3_thinking_budget": 10000,
  "stage3_max_agent_turns": 10,
  "stage5_llm": "vllm:Qwen/Qwen3.5-397B-A17B-FP8@http://node:8000",
  "skip_stage5_synthetic": false
}
```

`concurrency` is the single per-endpoint in-flight cap (see *Backends*).
For held-out / evaluation-data runs, set `stage3_thorough: true` and raise
`stage3_max_agent_turns` (e.g. 30) and `n_completions` (e.g. 30).

### Sharing stages across runs (sweeps)

To compare models on one axis — e.g. *which model investigates best*, or *how
well does Qwen/Haiku judge vs Opus* — you want the upstream stages shared and only
the varied stage repeated. Rather than a config entry point that points across run
dirs (a mis-wiring hazard), do this with the **standalone stage scripts**: every
stage script reads `--input` and writes `--output`, so you keep one self-contained
run dir and the wiring is visible in each command.

Each stage is its own script: `generate_completions.py` (stage 1),
`screen_completions.py` (2), `investigate.py` (3), `verify_investigations.py` (4),
`generate_synthetic_data.py` (5 — works, but synthetic data was not used for the
paper's results) — all take `--input` / `--output` / `--llm`.
`investigate.py --output` also relocates its `.jsonl` resume file and
`transcripts/` next to the output, so multiple investigators can write into
separate subdirs of one run dir without clobbering.

```bash
RUN=data/runs/sweep_200      # one self-contained dir

# Stage 2: three screeners on the same completions (judge-quality axis)
for s in qwen:openrouter:qwen/qwen3.5-397b-a17b haiku:anthropic:claude-haiku-4-5 opus:anthropic:claude-opus-4-6; do
  python chive/pipeline/screen_completions.py \
    --input $RUN/completions.jsonl --llm "${s#*:}" \
    --output $RUN/screening_${s%%:*}.jsonl --concurrency 100 --thinking-budget 4096
done

# Stage 3: a model investigates a screening (target = the model whose behavior we explain).
# Point --input at whichever screening you want (its own, or a shared one); --output to a subdir.
python chive/pipeline/investigate.py \
  --input $RUN/chain_opus/screening.jsonl --llm anthropic:claude-opus-4-6 \
  --target-llm openrouter:qwen/qwen3.5-397b-a17b --min-score 3 \
  --output $RUN/chain_opus/investigations.jsonl --concurrency 50

# Stage 4: verify a fixed investigations.jsonl with a second model to measure agreement
python chive/pipeline/verify_investigations.py \
  --input $RUN/chain_qwen/investigations.jsonl --llm anthropic:claude-opus-4-6 \
  --output $RUN/chain_qwen/verification_opus.jsonl --concurrency 50 --thinking-budget 4096
```

Use the native `anthropic:` backend for Claude judges/investigators (we have the
API key — cheaper and more reliable than routing Claude through OpenRouter);
`openrouter:` (or `vllm:`) for Qwen.

### Resuming

The pipeline is fully resumable. Each stage checks `status.json` and skips if "done". Within a stage, completed items are tracked in JSONL files. To re-run a specific stage:
1. Edit `status.json` in the run directory — set the stage's status to "pending"
2. Delete any output files for that stage if you want a clean re-run
3. Re-run the pipeline command

To change concurrency mid-stage-3: kill the pipeline, change `concurrency` in config.json, re-run. It resumes from the JSONL.

### Monitoring

```bash
# Check overall progress
cat data/runs/<run_name>/status.json

# Watch stage 3 progress
tail -f <pipeline_output_log> | grep "Investigation progress"

# Count completed investigations
wc -l data/runs/<run_name>/investigations.jsonl
```

### API key

The pipeline uses whatever `ANTHROPIC_API_KEY` is set in `.env`. To use the low-priority key, comment out the high-priority line and set ANTHROPIC_API_KEY to the low-priority value. Stage 2/4 batch API uses `ANTHROPIC_API_KEY_BATCH_API` (separate key).

## Key files

| File | Purpose |
|---|---|
| `run_pipeline.py` | Orchestrator — runs all stages (incl. Stage 0 slice), config-driven |
| `config.py` | `DEFAULTS` (single source of truth for every knob, incl. the stage CLIs' defaults) + `load_config` (merges config.json, rejects unknown keys) |
| `generate_completions.py` | Stage 1: completion generation via an llm_client backend (vLLM server / OpenRouter) |
| `build_prompt_pool.py` | Builds the WildChat base pool (`wildchat48m_clean_172k/prompts.jsonl`); run once |
| `build_agentic_pool.py` | Builds the agentic per-source JSONLs + merges with WildChat into the canonical `wildchat_plus_agentic/prompts.jsonl`; run once |
| `pool_common.py` | Shared pool record schema + JSONL I/O (`read_jsonl`/`write_jsonl`/`content_hash`) used by both pool builders |
| `screen_completions.py` | Stage 2: LLM-judge screening (`judge_llm`) |
| `investigate.py` | Stage 3: investigator agent (`stage3_llm`) + shared stage-3 driver |
| `verify_investigations.py` | Stage 4: verification judge (`judge_llm`) |
| `few_shot_question_pool.json` | 185 good question examples for screening |
| `INVESTIGATION_QUALITY_GUIDE.md` | What makes investigations interesting |
| `slurm_scripts/submit_vllm_server.sh` | Generic Slurm job for a vLLM server — `[MODEL] [PORT] [DP] [extra vllm args]`, GPU count overridden at sbatch time (see its header) |
| `slurm_scripts/submit_vllm_server_gemma3_27b_dp2.sh` | Slurm job for the gemma-3-27b-it vLLM server (needs `--language-model-only` + NCCL env) |

## After the pipeline completes

### 1. Train/test split

Split the pipeline output into train and test sets. Two conventions were used (both
simple to implement):

- **By verification score buckets** (older convention) — sample N examples per score bucket for the test set, rest goes to train.
- **By first-N prompts** (Opus-held-out convention, used for the 35B and 397B runs) — take the first N lines of `completions.jsonl` as the test set, everything else as train. The first-N rule preserves the held-out set defined when an Opus comparison run was seeded from the same prompts.

### 2. Training (Text SFT)

Train a LoRA on the pipeline output so the model can explain its own behavior in text.

- **Training script (local GPUs):** `chive/training/model_understanding.py`
- **Training script (Tinker API):** `chive/training/model_understanding_tinker.py` — same data
  semantics and loss mask, runs server-side on Tinker; used for the 397B LoRAs. Set
  `TINKER_PROJECT_ID` / `TINKER_OWNER` in `.env`. Configs: `data/training_configs/tinker/`
- **Training configs:** `data/training_configs/model_understanding/`
- **Slurm job template:** `jobs/slurm_job_template.sh` (see `docs/infra.md`)
- **Data assembly:** read `chive/training/model_understanding.py` (the code is the source of truth)
- **KL-regularized runs (`*_kl1` configs):** build the KL reference data first with
  `chive/training/prepare_kl_data.py` (produces the `.pt` file `kl_data_path` points at)

### 3. Evaluation

Evaluate trained LoRAs, base model text baselines, or activation oracle verbalizers.

- **Eval script (archived):** `chive/pipeline/archive/eval_model_understanding.py` — the
  GT-matching LLM-judge eval; not used for the paper (the paper evaluates open-ended
  explanations via hint cue-attribution and counterfactual simulatability, see
  `experiments/self_explanation/`)
- **Three modes:** `ao` (activation oracle), `text_baseline` (base model), `text_sft` (trained LoRA)
- **AO mode details:** the script's docstring

## Monitoring and operations

### Monitoring checklist

```bash
# Pipeline alive?
ps -p $(cat data/runs/<run_name>/pipeline.pid)

# Stage progress
cat data/runs/<run_name>/status.json

# vLLM server running? (needed for stage 3)
squeue -u $USER -o "%.10i %.12j %.2t %.10M %.20R %.10b"

# Investigation count (stage 3)
wc -l data/runs/<run_name>/investigations.jsonl

# Token rates (stage 3)
grep "\[rate\]" data/runs/<run_name>/pipeline.log | tail -3

# Errors
grep -i "SKIPPED\|Traceback\|rate_limit" data/runs/<run_name>/pipeline.log
```

### Resuming after a crash

```bash
source .env
nohup .venv/bin/python chive/pipeline/run_pipeline.py \
    --config data/runs/<run_name>/config.json \
    > data/runs/<run_name>/pipeline.log 2>&1 &
echo $! > data/runs/<run_name>/pipeline.pid
```

The pipeline resumes from JSONL checkpoints automatically.

### Changing concurrency mid-run

1. `kill $(cat data/runs/<run_name>/pipeline.pid)`
2. Edit `concurrency` in the run's `config.json`
3. Restart with the command above

### Expected timeline (40k prompts, measured from qwen3_32b_run)

- Stage 1 (vLLM server on 4 GPUs): ~1-2 hours
- Stage 2 (batch API, 40k prompts): ~30 min
- Stage 3 (concurrency 250-500, 3 GPUs): ~15 hours (the bottleneck)
- Stage 4 (batch API, ~22k verifications): ~30 min

### Tips

- **Check ~10 min after each stage starts** — failures surface early (bad GPU node, OOM, API errors)
- **Stage 1 backend health**: Stage 1 hits the `stage1_llm` endpoint (vLLM server / OpenRouter). If the empty-completion rate trips the 2% assert, the backend is likely overloaded — lower `concurrency` or check the server.
- **Stage 4 batch chunk size**: use `stage4_batch_chunk_size: 500` in config (verification prompts are ~330KB each, default 10k exceeds the 256MB batch API limit)
- **Always set `max_model_len`** on vLLM servers — reduces KV cache allocation and increases throughput significantly (see `tools/vllm_pool/SERVER_TUNING.md`)
