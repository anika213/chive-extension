# NLA (activation verbalizer) — production cache precompute + validation

NLA ([kitft/nla-inference](https://github.com/kitft/nla-inference)) reads a single
residual-stream vector from a subject model and describes it in words:

```
hidden-state vector v ─► AV verbalizer ─► explanation text ─► AR critic ─► v̂
                                                                          │
                                     cos(v, v̂) / FVE  (verification only — see below)
```

**AV** (activation verbalizer): a full fine-tune of the subject model. You build its
canonical prompt (template ships in the checkpoint's `nla_meta.yaml`), overwrite the
single marker token's input embedding with the activation (L2-normalized to the trained
`injection_scale`), and decode `<explanation>…</explanation>`. **AR** (activation
reconstructor): maps explanation text back to a vector; only used to *validate* wiring
and quality — production never reconstructs.

All production code lives in [`nla_inference.py`](nla_inference.py):
- `NLASpec` registry — one entry per checkpoint (base model, AV/AR paths, subject
  extraction layer). **The single source of truth**; everything else (injection scale,
  marker ids, template, d_model) is read from the checkpoint's own `nla_meta.yaml`.
- `AVInjector` — builds the prompt embeds, injects one vector per call. The one
  model-family behavior (Gemma scales token embeddings by `√d`, which vLLM's
  `prompt_embeds` path omits) is an explicit `model_type -> scale` map that fails loud
  on unknown architectures.
- `capture_residual` — HTTP capture from a vllm-lens base server (token ids in,
  `[T, d]` out, length-asserted).
- `AVDecoder` — offline `vllm.LLM(enable_prompt_embeds=True, async_scheduling=False)`;
  vectors in, `DecodeResult`s out.

Current specs: `qwen3-8b-baseline-nosalvage` (Qwen3-8B L24, Marco's checkpoint),
`qwen2.5-7b-L20` (kitft, regression target), `gemma3-27b-L41` (kitft, the original
binary-eval tool).

---

## Precomputing the NLA cache (the default production path)

The binary eval (`../../binary_eval/README.md`) uses NLA as an agent tool. NLA is
unconditional and per-token, so the tool is **fully precomputed**: every token position
of every transcript in the eval bank is verbalized once, and the agent tool
([`../nla_tool.py`](../nla_tool.py)) is a pure cache lookup — no GPU or server at eval
time.

**Architecture: ONE shared capture server + N offline decode workers.** Capture is one
cheap forward pass per transcript (yields every position's activation at once); decode
pays up to 256 generated tokens *per position*. The asymmetry is orders of magnitude,
so a single vllm-lens capture server saturates several decode workers. Decode runs
offline (`vllm.LLM` inside each Slurm array task, no HTTP AV server) for throughput and
fewer moving parts.

### Step 1 — start the capture server (1 GPU)

```bash
sbatch chive/binary_eval/agent/NLA/jobs/serve_capture.sh \
    Qwen/Qwen3-8B "" \
    data/scratch/capture_qwen3.json
```

Generic vllm-lens server for any base model. Writes `{"url", "base_model",
"slurm_job"}` to the conf JSON on startup; **always read the URL from there** (do not
assume the port you passed). Capture API: `POST /v1/completions` with
`vllm_xargs={"output_residual_stream": "[24]"}` (JSON-encoded string — vLLM rejects
lists in `vllm_xargs`).

> **⚠️ Port collisions (SO_REUSEPORT) — pass `""` for the port.** vLLM binds with
> `SO_REUSEPORT`, so two servers given the **same port** that land on the **same node**
> silently *share* it: the kernel load-balances connections between them and ~half the
> requests hit the wrong server → `404`/cross-model (e.g. a gemma + a qwen capture server
> both on `:8422`, same node — this bit us, manifesting as intermittent precompute 404s
> while `/v1/completions` worked from elsewhere). `serve_capture.sh` now auto-picks a free
> port when you pass `""`, and auto-switches if an explicit port is already serving on the
> assigned node. After startup, sanity-check there's no flap:
> `for i in $(seq 5); do curl -s $URL/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])'; done`
> — it must return the **same single model** every time.

### Step 2 — run the decode workers (5-task Slurm array, 1 GPU each)

```bash
sbatch chive/binary_eval/agent/NLA/jobs/precompute_nla_array.sh \
    <run_dir>/eval_set.jsonl  http://node-XX:8422  qwen3-8b-baseline-nosalvage
```

The bank is any jsonl with `prompt_id`/`messages`/`chosen_completion` rows (deduped by
`prompt_id`). Each task shards internally (`--num-shards/--shard-idx`), captures its
transcripts over HTTP, decodes every position offline (temp 1.0 — the in-distribution
RL setting — `max_tokens=256`, the trained budget), and writes
`<bank_dir>/nla_cache/shards/<spec.name>_shard{i}of{N}.jsonl`. For small banks /
smoke tests there is a single-job variant: `jobs/precompute_nla_cache.sh` (same args,
supports e.g. `--max-questions 50`).

### Step 3 — combine shards + verify

```bash
cat <bank_dir>/nla_cache/shards/<spec.name>_shard*of5.jsonl \
    > <bank_dir>/nla_cache/<spec.name>.jsonl
```

The final filename is the eval's default lookup (run identity = directory, checkpoint
identity = filename). Check the per-shard reports: `truncated (hit max_tokens)` and
`missing <explanation> tags` should both be ~0 (<0.1%); `nla_tool.load_nla_cache`
re-asserts full coverage, no duplicates, and the degeneration rate on load.

### Step 4 — run the eval arm

```bash
python -m chive.binary_eval.run_binary_eval \
    --tool nla --base-model Qwen/Qwen3-8B ...
```

---

## Validation: the round trip

[`nla_roundtrip.py`](nla_roundtrip.py) (TEST harness, not production — it imports the
production primitives and adds the torch AR critic + scoring). Three capture modes:

- **`fineweb`** — the quality/wiring gate. Training-matched capture: raw fineweb
  sample-10BT docs (no chat template), random positions ≥50 skipping special tokens,
  replicating the author's `stage0_extract.py`. Has a reference number (below); a
  wiring bug (wrong layer / embed scale / marker / injection scale) tanks it.
- **`chat`** — the production-format diagnostic. The fixed prompt set through the base
  chat template, ALL positions, each tagged `content` vs `scaffold` (exact char-offset
  match against the rendered template); the summary reports the two subgroups.
- **`parquet`** — decode+AR only on vectors from one of the author's extraction
  parquets (no capture server needed). Use to onboard a NEW checkpoint fast and to
  isolate AV+AR quality from our capture path.

```bash
sbatch chive/binary_eval/agent/NLA/jobs/roundtrip.sh \
    qwen3-8b-baseline-nosalvage <conf.json> --capture-mode fineweb
```

Each run prints the first 8 full generations and writes a complete per-sample JSONL
(context tail, raw output, per-sample cos/mse/FVE) to
`results/roundtrip_<spec>_<mode>_<jobid>.jsonl`.

### FVE: denominators are computed on the eval set, never hardcoded

`fve = 1 - mse / baseline`, where both predict-the-mean baselines (raw-variance
`fve_nrm`, the author's training metric; normalized-mean `fve_meannorm`, his
eval-script metric) are computed **on the gold vectors of the run itself** —
replicating the author's `compute_predict_mean_baselines`. A hardcoded denominator
from another distribution (e.g. 0.7335, Qwen2.5's training variance) would make the
FVEs meaningless; they are only comparable when the underlying eval distribution matches.
Note `mse = 2(1-cos)` on the normalized vectors, so cos and FVE carry the same
per-sample signal; FVE additionally normalizes by how concentrated the eval set is
(a low-variance set — e.g. chat scaffold tokens — can have high cos and *negative*
FVE at the same time).

### Validation results — Qwen3-8B baseline (`rl_qwen3_baseline_nosalvage`, iter 400)

All through this folder's pipeline (vllm-lens capture + offline AV decode + torch AR),
2026-06-12, N≈100-700 per row, temp 1.0:

| mode | what it measures | mean cos | fve_nrm |
|---|---|---|---|
| author's training log (reference) | in-distribution ceiling | ≈0.93 | ≈0.75 |
| `parquet` on his RL training rows | AV+AR only, seen data | 0.932 | 0.755 |
| `fineweb` (our capture, held-out docs) | full pipeline, unseen | 0.934 | 0.753 |
| `parquet` on his held-out fineweb | his capture, unseen | 0.927 | — |
| `parquet` on his held-out **wildchat** | chat-style content, unseen | **0.886** | — |
| `chat` **content** tokens (our capture, n=303) | content inside the chat template | **0.927** | 0.699 |
| `chat` **scaffold** tokens (n=175) | `<|im_end|>/<|im_start|>/assistant/\n` | 0.746 | low/negative |

(Scaffold per-token cos ranges 0.50 on the trailing `\n` to 0.87 on `<|im_start|>`;
on a scaffold-only eval set the FVE is negative — those vectors cluster around a
shared mean, so cosine flatters them while per-sample information is ~nil.)

Takeaways: (1) the pipeline is at exact parity with the author's numbers — capture
path, injection, decode, and scoring are all verified; (2) **content tokens inside
the chat template are at in-distribution quality** (0.927 vs 0.934 — chat formatting
costs ~nothing on content); (3) **chat scaffold tokens are near-noise** (the AV was
trained on raw fineweb only — scaffold activations are OOD, and their verbalizations
describe generic "AI assistant dialogue" rather than the transcript). The cache
includes scaffold positions for coverage, but treat their verbalizations as
uninformative (the agent tool prompt warns about this).

The Qwen2.5/kitft AV scores 0.90 on the same chat scaffold positions because it was
RL-trained ~40× longer *including WildChat* (chat data); Marco's Qwen3 baseline
(400 rollouts, fineweb-only, 1-act no-source critic) is the weakest variant of the
family by design.

### Onboarding a new NLA checkpoint

1. Add an `NLASpec` to `nla_inference.py` (base model, AV/AR paths, extraction layer —
   get the layer from the training-data sidecar, e.g.
   `<rl.parquet>.nla_meta.yaml::extraction.layer_index`, not from word of mouth).
2. Run `--capture-mode parquet` against a held-out extraction parquet if one exists
   (no capture server needed), then `--capture-mode fineweb` against a capture server
   for the full-pipeline number. Expect ≈ the checkpoint's training-log
   `train/fve_nrm`; large gaps mean wiring (check the embed-scale map and the layer).
3. Eyeball the printed generations + the JSONL before building any cache.

---

## Gotchas

- **⚠️ `async_scheduling=False` is REQUIRED for `prompt_embeds` decoding on vLLM < 0.22
  (root-caused 2026-06-11).** vLLM 0.19.1's
  async scheduling (ON by default, offline and server alike) corrupts a fraction of
  `prompt_embeds` decodes into never-closing `<explanation><<<<<<...` loops: on an
  all-decode step whose batch was reordered (a request finished → `condense()`),
  `gpu_model_runner._prepare_input_ids()` takes a scatter path that updates
  `input_ids.gpu` but never refreshes the `is_token_ids.gpu` mask, so `_preprocess()`
  feeds **stale `inputs_embeds.gpu` rows (another request's injected AV prompt
  embedding) as decode inputs**. One bad step can corrupt ~100 in-flight requests —
  hence bursty rates (~21% under streaming HTTP load; ~0.5% offline).
  A/B-verified fix: `async_scheduling=False` → 0 engine-caused degenerations at <10%
  throughput cost (`AVDecoder` sets it). Fixed upstream in vLLM v0.22
  ([vllm#42347](https://github.com/vllm-project/vllm/pull/42347)). Things that do NOT
  fix it (tested): disabling prefix caching, capping `max_num_seqs`, temperature.
  `repetition_penalty` separately **crashes** the vLLM < 0.21 prompt_embeds engine
  (sampler penalties index placeholder ids; fixed by vllm#40720 in v0.21).
- **Injection scale matters.** The AV expects `||v|| = injection_scale` exactly (150
  for the Qwen checkpoints, 60000 for gemma — read from yaml, never hardcoded).
  Raw-magnitude vectors are OOD → CJK-flavoured garbage.
- **Marker neighbor check.** The injection char is rare but not unique; `AVInjector`
  verifies the ±1 neighbor token ids from the yaml before overwriting.
- **AV prompt thinking mode.** Build the AV prompt with the tokenizer-DEFAULT chat
  template (no `enable_thinking` kwarg) — verified identical to the author's RL
  rollout templating. Forcing `enable_thinking=False` inserts an empty think block the
  model never saw.
- **Layer convention.** "Extraction layer L" = output of decoder block L
  (`hidden_states[L+1]` in HF terms). vllm-lens matches this (for fused-residual
  models like Qwen3 it reconstructs `hidden_states + residual`). Verified at parity
  with the author's HF-hook extraction (fineweb row of the results table).
- **AR critic requires `add_special_tokens=True`** (no-op for Qwen, load-bearing for
  Gemma/Llama critics).
- **`vllm_xargs` lists are rejected by vLLM** — JSON-encode:
  `{"output_residual_stream": "[24]"}`. See `../VLLM_LENS_PATCH.md`.

## File index

- [`nla_inference.py`](nla_inference.py) — production module: spec registry, injector,
  capture client, offline decoder
- [`precompute_nla_cache.py`](precompute_nla_cache.py) — cache builder (one shard)
- [`nla_roundtrip.py`](nla_roundtrip.py) — validation harness (fineweb / chat /
  parquet modes; writes per-sample JSONLs to `results/`)
- [`jobs/serve_capture.sh`](jobs/serve_capture.sh) — generic vllm-lens capture server
- [`jobs/precompute_nla_array.sh`](jobs/precompute_nla_array.sh) /
  [`jobs/precompute_nla_cache.sh`](jobs/precompute_nla_cache.sh) — decode workers
- [`jobs/roundtrip.sh`](jobs/roundtrip.sh) — validation job (extra args pass through)
- [`../nla_tool.py`](../nla_tool.py) — the agent tool (pure cache lookup)
- [`vllm_ar/`](vllm_ar/) — optional HTTP-served AR (not used by the eval path; the
  round trip uses torch AR)

Historical note: SGLang `input_embeds` ↔ vLLM `prompt_embeds` equivalence was
validated 2026-06 (mean cos 0.888 vs 0.885 on Qwen2.5; per-vector capture parity vs HF
≥ 0.9998). Artifacts in `/workspace-vast/adamk/nla_verification/` (external archive, not
included in this release).
