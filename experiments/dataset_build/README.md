# Held-out eval dataset builds (Qwen3.5-397B + PETRI)

**Purpose.** Provenance record for three held-out investigation datasets built for the paper's
training evaluations: a WildChat run on Qwen3.5-397B and two PETRI-sourced runs (397B and
Qwen3-8B). All three use the same "thorough / n30" recipe as the Gemma reference run
(`opus_investigates_gemma3_27b_it_5000_thorough_n30`).

**These are held-out test sets — do NOT train on them.**

**Paper artifact.** The held-out evaluation datasets of §`sec:training_heldout`
(Figure `fig:heldout_direct`, `fig:heldout_selfexpl`) and the PETRI prompt-source description in
`app:prompt_sources`.

All eval data excludes the conversations in `data/eval_blocklist.json` (prompts that also
appear in training data); `chive/pipeline/build_agentic_pool.py` applies this exclusion as its
final build step, so a pool rebuild reproduces the shipped pool exactly.

## The datasets (in `data/runs/`)

| Run | Source prompts | Target model | Prompts → investigated | Stage-4 mean |
|---|---|---|---|---|
| `opus_investigates_qwen3p5_397b_5000_thorough_n30` | WildChat (canonical pool, first 5000 minus blocklist = 4714) | Qwen3.5-397B-A17B-FP8 | 4714 → 1616 | 8.46 |
| `opus_investigates_qwen3p5_397b_petri_thorough_n30` | PETRI audit of the 397B (344 transcripts → 1732 prompts) | Qwen3.5-397B-A17B-FP8 | 1732 → 549 | 8.34 |
| `opus_investigates_qwen3_8b_petri_thorough_n30` | PETRI audit of Qwen3-8B (343 transcripts → 2061 prompts) | Qwen3-8B | 2061 → 880 | 8.26 |

## Recipe ("thorough / n30")

Plain config keys consumed by `chive/pipeline/run_pipeline.py` and `chive/pipeline/investigate.py`
(each run's exact config is its `config.json`):

- `n_completions: 30` — Stage 1 samples 30 completions/prompt; the investigator sees only the
  first 10, the grader counts behavior rates over all 30.
- `stage3_thorough: true` + `stage3_max_agent_turns: 16` — the investigator confirms the cause
  AND rules out 1–2 salient competitors (~10–15 experiments per behavior).
- Investigator + judge: Opus 4.6; grader: Sonnet 4.6.
- Target generation with thinking OFF (verified — see gates below).
- `skip_stage5_synthetic: true` — held-out eval, no synthetic training data.

## PETRI audits (prompt source for the two petri runs)

Transcripts were generated with PETRI, Sonnet 4.6 as auditor (original variant: system prompts
and tools kept), against each target served from the vLLM pool (`tools/vllm_pool/`). The audit
transcripts were flattened into prompt shards ending at a user turn, with a 4096-token cap
(~33% of the deepest petri prefixes dropped, expected for long audit contexts); each run's
`prompts.jsonl` is the resulting input.

**Qwen3-8B tool-call parser note (for re-runs).** No single vLLM tool-call parser serves both
uses of the dense Qwen3-8B: `hermes` parses the 8B's `<tool_call>{json}` output (required for the
*audit*, which needs structured tool calls) but errors on truncated JSON and crashes the
*investigation* pipeline; `qwen3_coder` passes the 8B's tool calls through as plain text (safe for
investigation, useless for auditing). The 397B has no such split — its native format *is* the
`qwen3_coder` format, so one parser covers both. Consequently `tools/vllm_pool/submit_qwen3_8b.sh`
defaults to parser-OFF (correct for its main job: investigations on WildChat/PETRI prompts), and
a PETRI audit of the 8B needs a one-off hermes-enabled server.

## Verification gates (all three runs passed)

- **Stage-4 judge score:** means in the table above (≥7 on 96–97% of investigations).
- **Thinking OFF** — `check_thinking_off.py --run-dir <run>` (or `--petri-transcripts <json>`)
  scans target completions for leaked `<think>` markers; empirical ground truth, not a config
  re-read. Zero leaks on the WildChat run. On PETRI-sourced runs, interpret hits
  **qualitatively**: auditors probe for reasoning extraction, so the target *discussing or
  refusing* `<think>` tags is content, not a leak — all flagged hits on the petri runs were this.
- **Qualitative review:** ≥2 investigations per dataset read end-to-end before acceptance.
