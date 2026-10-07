# CHIVE — Checkpoint 1, revision 2

**STOPPED before embedding or clustering. No LLM calls or model inference.**

## Methods and schema changes

- Original schema mapping, verified rate calculations, effect threshold (absolute delta_pp >=50), within-source exact cause-text deduplication, and safe edit-text extraction are inherited from [revision 1](../checkpoint1/preprocessing_log.md). Revision 1 is preserved.
- Cause rows retained at revision 1 stay in causes.parquet, including newly flagged rows. New flags are evaluated AFTER the existing exact cause-text deduplication. causes_for_clustering.parquet is the exact clustering_eligible subset; no embeddings have been computed.
- Contexts come directly from investigations.jsonl: messages versus experiment_log[].messages, joined by source, prompt_id, and exp_num. For legacy experiments with no messages field, use the stored prompt as the full user message and prepend system_prompt when present, exactly as CHIVE’s documented experiment_to_messages reader specifies (chive/chive/pipeline/utils.py:676–699). No context is inferred from investigator prose. Missing/unsupported contexts are flagged and excluded from clustering, not guessed.
- Each context is rendered in stored message order, with explicit role/end-message headers and verbatim content. Character counts are Unicode code points, including these structural markers; token counts use the local all-mpnet-base-v2 tokenizer with no truncation, padding or special tokens. Token counts are NOT Qwen token counts. No model weights or inference are used.
- Exact LCS diffs are computed independently for characters and token IDs. Added = edited length minus LCS length; deleted = baseline length minus LCS length; changed = added + deleted. Replacements contribute both deleted and added units. Sizes are not merely differences in total context length.
- context_swap = characters changed > source p90 OR tokens changed > source p90 (strictly greater). Percentiles use linear interpolation across ALL context-comparable experiments with verification >=7 in that source, before the effect threshold and cause-text deduplication. This is a size proxy, not a semantic claim that the context was replaced.
- edit_type is now based only on the character diff: delete-only → remove, add-only → add, both → modify; identical contexts → no_change; unavailable context → unknown. Tokenizer normalization cannot hide character-only edits. no_change is reported but is not an additional exclusion requested by this revision.
- measurement_artifact checks the ORIGINAL summary, case-insensitively, for N/A, not applicable, not reach, as expected, trivially, cannot be graded. Whitespace is normalized; phrases/markers have word boundaries (so region/add and medication/anxiety do not match N/A), and not reach includes reached/reaching. Quoted occurrences count. This is a requested lexical flag, not proof that the measurement is invalid.
- Behaviour cleanup removes leading numeric count/percentage phrases and all-N-completions phrases only. It preserves the remaining description verbatim apart from existing light Markdown/whitespace cleanup; it does not delete the observed behaviour sentence. Exact behaviour-text deduplication is re-run after this cleanup.
- New cause columns: verif_score; baseline/experiment context chars/tokens; chars_deleted/added/changed; tokens_deleted/added/changed; context_diff_available/error; context_swap and its source char/token p90 cutoffs; measurement_artifact and matched terms; clustering_eligible and clustering_exclusion_reasons. edit_type replaces the old verb-based value.
- Raw/unfiltered copies and the diff measurements for every experiment remain available per source. Character/token counts and hashes are in context_diffs.parquet. Previous rate disagreement flags remain in revision 1; their measurements are unchanged.

## Updated row counts

| Source | Behaviours | Causes kept in Parquet | Context swap | Artifact | Both flags | Missing diff | Clustering-eligible causes |
|---|---:|---:|---:|---:|---:|---:|---:|
| petri | 801 | 4,429 | 502 | 25 | 4 | 0 | 3,906 |
| wildchat_eval | 1,641 | 7,864 | 882 | 72 | 7 | 0 | 6,917 |
| wildchat_thinking | 1,348 | 5,478 | 700 | 41 | 8 | 0 | 4,745 |
| wildchat_train | 21,780 | 61,543 | 7,861 | 31 | 9 | 0 | 53,660 |

## Edit-size distributions and source-specific p90 cutoffs

Population: all verification-qualified experiments with available contexts in each source. Units are deleted + added. The two p90 columns independently define the OR exclusion rule.

| Source | Unit | N | Mean | Min | P25 | Median | P75 | P90 cutoff | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| petri | chars_changed | 11,680 | 2,259.39 | 0.00 | 104.00 | 304.50 | 2,602.25 | 8,123.00 | 11,719.20 | 16,041.94 | 20,123.00 |
| petri | tokens_changed | 11,680 | 602.20 | 0.00 | 27.00 | 85.00 | 723.00 | 2,183.20 | 3,056.10 | 4,070.05 | 5,841.00 |
| wildchat_eval | chars_changed | 23,577 | 770.95 | 0.00 | 35.00 | 117.00 | 765.00 | 2,636.00 | 3,960.20 | 6,586.72 | 9,326.00 |
| wildchat_eval | tokens_changed | 23,577 | 209.24 | 0.00 | 10.00 | 33.00 | 208.00 | 682.00 | 1,047.20 | 1,797.24 | 5,002.00 |
| wildchat_thinking | chars_changed | 16,444 | 667.33 | 0.00 | 34.00 | 109.00 | 546.00 | 2,382.00 | 3,649.65 | 6,098.99 | 7,922.00 |
| wildchat_thinking | tokens_changed | 16,444 | 177.27 | 0.00 | 10.00 | 30.00 | 156.00 | 587.00 | 915.55 | 1,589.00 | 3,071.00 |
| wildchat_train | chars_changed | 167,133 | 594.17 | 0.00 | 55.00 | 137.00 | 749.00 | 1,815.00 | 3,011.00 | 4,088.68 | 7,028.00 |
| wildchat_train | tokens_changed | 167,133 | 145.81 | 0.00 | 10.00 | 32.00 | 184.00 | 447.00 | 714.00 | 1,059.00 | 3,248.00 |

## Diff-derived edit types

| Source | Population | Remove | Add | Modify | No change | Unknown |
|---|---|---:|---:|---:|---:|---:|
| petri | all_size_reference_experiments | 2,312 | 1,168 | 8,059 | 141 | 0 |
| petri | retained_cause_rows | 599 | 531 | 3,299 | 0 | 0 |
| petri | eligible_cause_rows | 402 | 531 | 2,973 | 0 | 0 |
| wildchat_eval | all_size_reference_experiments | 3,048 | 2,231 | 18,054 | 244 | 0 |
| wildchat_eval | retained_cause_rows | 741 | 937 | 6,186 | 0 | 0 |
| wildchat_eval | eligible_cause_rows | 571 | 932 | 5,414 | 0 | 0 |
| wildchat_thinking | all_size_reference_experiments | 2,222 | 1,531 | 12,555 | 136 | 0 |
| wildchat_thinking | retained_cause_rows | 516 | 611 | 4,350 | 1 | 0 |
| wildchat_thinking | eligible_cause_rows | 360 | 606 | 3,778 | 1 | 0 |
| wildchat_train | all_size_reference_experiments | 21,141 | 6,764 | 135,891 | 3,337 | 0 |
| wildchat_train | retained_cause_rows | 6,756 | 2,480 | 52,294 | 13 | 0 |
| wildchat_train | eligible_cause_rows | 5,630 | 2,479 | 45,538 | 13 | 0 |

## Causes per investigation

Main denominator: every verification-qualified investigation in the source, INCLUDING those with zero eligible causes. Statistics are after existing exact cause-text deduplication; the conditional positive-only statistics are also saved in counts.json.

| Source | Population | Investigations | Mean | Median | Max | Zero-cause investigations |
|---|---|---:|---:|---:|---:|---:|
| petri | Before new exclusions | 801 | 5.529 | 6 | 18 | 111 |
| petri | Eligible after exclusions | 801 | 4.876 | 4 | 17 | 134 |
| wildchat_eval | Before new exclusions | 1,641 | 4.792 | 5 | 19 | 364 |
| wildchat_eval | Eligible after exclusions | 1,641 | 4.215 | 3 | 19 | 433 |
| wildchat_thinking | Before new exclusions | 1,348 | 4.064 | 4 | 14 | 296 |
| wildchat_thinking | Eligible after exclusions | 1,348 | 3.520 | 3 | 14 | 350 |
| wildchat_train | Before new exclusions | 21,780 | 2.826 | 3 | 14 | 5,906 |
| wildchat_train | Eligible after exclusions | 21,780 | 2.464 | 2 | 13 | 6,707 |

## Examples and files

[Open retained report](preprocessing_log.md) for 10 flagged artifact examples and 10 behaviour before/after examples per source where available, plus 15 random rows of each eligible facet per source. Random seed 20261004 + source index; no similarity-based selection.

Per source: behaviours.parquet, behaviours_unfiltered.parquet, causes.parquet (flagged rows retained), causes_unfiltered.parquet, causes_for_clustering.parquet, context_diffs.parquet, counts.json, review_samples.json, behaviour_prefix_edits.jsonl, behaviour_exclusions.jsonl, clustering_exclusions.jsonl.

Reproduce: `clustering/.venv/bin/python clustering/preprocessing.py`. Reuse completed exact context diffs with `--reuse-context-diffs`. Original checkpoint at /Users/sharm51155/Desktop/chive/clustering/data/checkpoint1.

**Awaiting Checkpoint 1 approval. Near-duplicate embedding checks, embeddings, and clustering have not run.**
