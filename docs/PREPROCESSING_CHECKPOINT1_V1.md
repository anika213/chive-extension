# Observation-only preprocessing: Checkpoint 1

Run from the project root:

```bash
clustering/.venv/bin/python clustering/scripts/preprocessing.py
```

This pipeline uses no LLM and computes no embeddings or clusters. Its only
external dependency is `pyarrow==25.0.1` for Parquet output. It stops at Checkpoint
1. Embedding, near-duplicate reporting, and clustering require subsequent user
approval.

## Review and output files

Open [the preprocessing log](../data/checkpoint1/preprocessing_log.md) for 15 random behaviour rows
and 15 random cause rows **per source**, with original and preprocessed text side
by side. Read [the preprocessing log](../data/checkpoint1/preprocessing_log.md) for
the schema mapping, counts after each step, applied rules, and flags.

Files are separated into `data/checkpoint1/{petri,wildchat_eval,wildchat_thinking,wildchat_train}/`:

- `behaviours.parquet`: one retained row per investigation, verification >=7,
  using `behavior_summary` or `structured_findings.baseline_behavior`, with
  within-source exact text duplicates removed.
- `causes.parquet`: one retained row per experiment with absolute measured change
  >=50 pp, using only its edit description from `experiment_summaries[].summary`.
  These are intervention observations, not proven causal mechanisms.
- `behaviours_unfiltered.parquet` and `causes_unfiltered.parquet`: before row
  filters and deduplication; preserve incomplete rows with nullable fields. Cause
  text preprocessing has already been applied, with the full original summary
  retained separately.
- `flags.jsonl`: rate disagreements, annotated/reliability-qualified rate strings,
  missing fields, and other review flags.
- `exclusions.jsonl`: every excluded row and the reason. Duplicate records identify
  the retained row.
- `text_edits.jsonl`: removed text spans and rule categories.
- `counts.json` and `review_samples.json`: machine-readable counts and samples.

`input_observations.jsonl` is an allowed-field audit copy of the original records,
including all labels, original summaries, and source-file line numbers. It does
not include Core causes, Final answer, or any explanatory report sections.
`input_provenance.json` records source hashes, revision, and the cache hash.

## Measurement and text rules

Stored classifier labels take priority over reported rates. Exact rational
arithmetic determines threshold inclusion; output rates are proportions in [0,1]
and `delta_pp` is signed percentage points. If labels are absent, the leading
`x/n` string is the only permitted fallback. Annotation prose is never treated as
a replacement measurement. Missing descriptions or unusable rates are excluded
from filtered cause tables and preserved in the audit outputs, as authorized.

Text stripping is deterministic and deliberately does not paraphrase. It separates
sentences and outcome/inference clauses while protecting quoted and inline-code
edit payloads. Numeric literals in an actual edit are retained. Ambiguous or
unrecognized segments are removed; rows with no extractable edit text are flagged
and excluded pending review. The side-by-side checkpoint exists to assess these
heuristics before embeddings are computed. Both facets get only Markdown and
whitespace cleanup beyond outcome stripping for causes.

Edit type is tagged from the original description's leading verb exactly as
requested. `unknown` is retained, especially for passive descriptions such as
“The edit removed…”. `is_key` remains metadata and is not a filter.

To iterate deterministic text rules without reparsing the full raw files:

```bash
clustering/.venv/bin/python clustering/scripts/preprocessing.py --reuse-extracted-inputs
```

Historical claim-based preprocessing is preserved as `preprocessing_legacy.py`
and documented in `PREPROCESSING_LEGACY.md`. It is not part of this workflow; the
current script never imports or executes it. Historical outputs elsewhere are
not inputs to Checkpoint 1.
