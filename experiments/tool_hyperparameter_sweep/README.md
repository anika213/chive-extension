# Tool-hyperparameter sweep

**Purpose.** Test whether the eval's tool-budget hyperparameters change the outcome, by sweeping
the two knobs of the activation-tool arms on the Gemma-3-27B-IT claims: **tool calls per claim
∈ {2, 5, 10}** (AO/NLA/SAE) and **token positions per call ∈ {16, 32, 64}** (NLA/SAE).

**Paper artifact.** The table and figure of Appendix `app:toolcalls` (cited from
§`sec:interp_eval_results`). All numbers live in the paper and the auto-generated `SUMMARY.md`.

## Design

- 10 fresh sweep arms, all `claude-opus-4-8` at effort `high` — identical answerer settings to
  the main run. The canonical main-run arms (5 calls / 32 positions) and `plain` are **read
  back** from their existing result files as the center points and baseline, not re-run; pairing
  is automatic via claim-id intersection.
- The AO is excluded from the positions sweep: it selects transcript *segments* to ask questions
  about, not token positions — the knob doesn't exist for it.
- The arms run on the full `eval_set_conc3_n2000.jsonl` (~1,750 claims); `analyze.py` restricts
  every reported number to the quality slice ∩ all arms (n=1,380), the population of the paper's
  main figures.
- Output: `data/experiments/tool_hyperparameter_sweep/results/` with labels
  `tc2`/`tc10`/`pos16`/`pos64` — kept out of the run dir's canonical `binary_evals/` so the main
  plotter never globs them.

## Reproduce

    bash experiments/tool_hyperparameter_sweep/run_all.sh   # idempotent; every arm --resume

The 8 NLA/SAE arms are cache-only and cost API only. The NLA/SAE read caches (`nla_cache/`,
`sae_cache/` next to the bank) are not part of the data release — regenerate them with
`chive/binary_eval/agent/NLA/precompute_nla_cache.py` / `agent/SAE/precompute_sae_cache.py` (GPU),
or rely on `--resume` over the shipped result files, which never touches the caches. The 2 AO
arms need the live Gemma AO server, which the script submits
(`chive/binary_eval/agent/jobs/serve_gemma_ao.sh`, 1 GPU) and tears down when done. The final step
runs `analyze.py`, which writes `SUMMARY.md` (all numbers; auto-generated, do not hand-edit) and
`plots/`.

## Related archived result

[NLA_FULL_DUMP_RESULTS.md](NLA_FULL_DUMP_RESULTS.md) — the NLA full-dump / no-selection ceiling
test cited by `SUMMARY.md` and `app:toolcalls` ("additionally tried"). Kept as a write-up only:
it ran on an older claim bank and its raw data was not retained.
