# Observation-only preprocessing

Latest revision: [edit location and seed audit](../results/location_seed/review.html),
with [updated cause clusters](../results/checkpoint2_locations/report.md).
`location_seed_analysis.py` joins raw contexts against verified hashes and records
changed roles/message indices, including explicitly wrapped tool outputs. It removes
unquoted role/turn locator spans from cause text, preserving originals and removed
spans. Location-only remnants and new exact duplicates are flagged and excluded
from embedding; all retained rows stay in `data/checkpoint1_locations/*/causes.parquet`.
Quoted intervention payloads remain verbatim. Behaviour embeddings are unchanged.
See [revision methods and reproduction](LOCATION_SEED.md).

Current outputs: [approved checkpoint 1 amendments](../data/checkpoint1_approved/preprocessing_log.md)
and [completed checkpoint 2](../results/checkpoint2/report.md), for PETRI and WildChat eval only.
The amended cause tables add `control`, `sanity check`, `qualitatively`, and `grader`
to artifact matching, plus `frac_changed` and `context_swap_relative` (> 0.5).
Clustering excludes either size flag or any measurement-artifact flag; flagged rows
remain in `causes.parquet`. All cause-diversity denominators exclude zero-eligible-cause
investigations and report their counts. HDBSCAN also reports and excludes investigations
with only noise from assigned-cluster diversity statistics.
See [reproduction instructions](CHECKPOINT2.md). Stopped before semantic labelling.

## Preserved Checkpoint 1, revision 2

Historical review: [revision 2 preprocessing notes](../data/checkpoint1_v2/preprocessing_log.md) and
[methods, counts, edit-size distributions, and edit types](../data/checkpoint1_v2/preprocessing_log.md).

```bash
clustering/.venv/bin/python clustering/scripts/preprocessing.py
# Reuse already-computed context diffs:
clustering/.venv/bin/python clustering/scripts/preprocessing.py --reuse-context-diffs
# Validate outputs and recompute raw-context diff samples:
clustering/.venv/bin/python clustering/scripts/validate_context_preprocessing.py
```

`preprocessing_context.py` applies the five requested revisions to the preserved
revision-1 tables. It computes exact character and token insertion/deletion counts
from stored contexts. Legacy `prompt`/`system_prompt` records use CHIVE's documented
conversation reader mapping. Tokens come from the local MPNet tokenizer without
truncation; no model weights or inference are used. Dependencies include PyArrow,
NumPy, tokenizers, and RapidFuzz (versions in requirements.lock.txt).

Within each source, `context_swap` means either changed characters or changed
tokens strictly exceeds its 90th percentile across all context-comparable,
verification-qualified experiments. Artifact phrases are matched against the
original summary with case-insensitive word/marker boundaries. Character diffs
determine edit type; zero-edit controls are recorded as `no_change`.

Revision 2 files live under `data/checkpoint1_v2/<source>/`. **Flagged cause rows
stay in `causes.parquet`.** Use `clustering_eligible == true` or the equivalent
`causes_for_clustering.parquet` for any future embedding. Existing cause-text
deduplication remains before these new flags; no discarded duplicate is silently
substituted for its prior representative. `causes_unfiltered.parquet` preserves
every logged experiment, including prior exclusions. Behaviour prefixes are
stripped and behaviour deduplication is re-run.

The review page includes source-specific distributions, edit-type counts, causes
per investigation (including zero-cause investigations), 10 artifact examples and
10 behaviour before/after examples per source where available, and the original
15-row samples for both eligible facets. Revision 2 stopped at Checkpoint 1;
subsequent approved amendments and clustering are linked above.

## Revision 1 reference

The remaining documentation describes the preserved revision-1 intermediate
tables. Use `--base-only` with the commands below to regenerate that intermediate
checkpoint; it does not include revision-2 context controls.

Run from the project root:

```bash
clustering/.venv/bin/python clustering/scripts/preprocessing.py --base-only
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
clustering/.venv/bin/python clustering/scripts/preprocessing.py --base-only --reuse-extracted-inputs
```

Historical claim-based preprocessing is preserved as `preprocessing_legacy.py`
and documented in `PREPROCESSING_LEGACY.md`. It is not part of this workflow; the
current script never imports or executes it. Historical outputs elsewhere are
not inputs to Checkpoint 1.
