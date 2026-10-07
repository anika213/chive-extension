# CHIVE observation preprocessing — CHECKPOINT 1

**STOPPED for user review. No LLM calls, embeddings, clustering, or cluster labels.**

## Schema mapping

| Requested | Stored field |
|---|---|
| prompt_id | investigations.jsonl: prompt_id |
| source | Source directory mapping below; no cross-source joins |
| behaviour_text | behavior_summary; fallback structured_findings.baseline_behavior |
| verif_score | verification.jsonl: score, joined by source and prompt_id |
| cause_text_original | structured_findings.experiment_summaries[].summary |
| Experiment metadata | experiment_number, baseline_rate, counterfactual_rate, is_key in the same summary |
| Baseline labels | baseline_grader_verdicts[] |
| Counterfactual labels | experiment_log[].grader_verdicts[] |
| Experiment join | summary experiment_number = log exp_num within source/prompt_id |

## Rules applied

- Qwen3-8B only, selected from the four explicitly mapped downloaded runs. Raw inputs stay intact.
- Verification >=7. Missing-score investigations are excluded from filtered tables.
- Baseline and counterfactual rates use stored boolean labels. If labels are absent, use the leading x/n in the summary; never substitute qualitative estimates. Annotated rate strings are flagged. Invalid labels are not guessed.
- Parsed versus recomputed rate disagreements are flagged; measured labels take priority. Grader reliability notes remain flags, not corrected observations.
- Signed delta_pp = 100 × (counterfactual − baseline). The absolute >=50 threshold uses exact rational arithmetic, avoiding floating-point boundary errors.
- Missing descriptions/rates are excluded, as approved. Incomplete rows remain in unfiltered audit tables, with null fields where unavailable.
- Cause text uses only experiment summary edit descriptions. No Core causes, Final answer, structured_answer, or investigator report is read into the preprocessing cache.
- Deterministic quote-aware sentence/clause rules remove result and interpretation clauses. Quotes and inline code protect literal edited strings (including fractions and percentages). Unrecognized/ambiguous sentences are removed rather than paraphrased. Empty results are excluded and flagged.
- Removed spans are recorded in text_edits.jsonl. This is heuristic extraction; review the side-by-side samples for over-stripping or missed outcomes before approving.
- Edit type follows the requested leading verbs exactly: Removed/Deleted → remove; Added/Inserted → add; Changed/Replaced/Rewrote and Kept … but → modify; otherwise unknown. Unknown does not imply missing data.
- Both facets get whitespace and Markdown-formatting cleanup; original wording, capitalization, names, negation and identifiers are preserved. Link labels and targets are retained.
- Exact duplicate cleaned texts are removed within each facet/source, keeping the first eligible source-file occurrence. Exclusions log retained IDs. No across-source deduplication.
- Unfiltered Parquet copies are before score/effect/completeness/dedup filtering. Unfiltered cause rows include every logged experiment, with nulls for missing fields.
- Near-duplicate cosine >0.95 checks are deferred to the embedding phase after approval, and will not delete rows.

## Source mapping

- `petri`: `petri/qwen3_8b_petri_eval`
- `wildchat_eval`: `wildchat/qwen3_8b_wildchat_eval`
- `wildchat_thinking`: `wildchat/qwen3_8b_wildchat_thinking_eval`
- `wildchat_train`: `wildchat/qwen3_8b_wildchat_train`

## Filter-count log

| Step | petri | wildchat_eval | wildchat_thinking | wildchat_train |
|---|---:|---:|---:|---:|
| Qwen3-8B investigations / unfiltered behaviour rows | 868 | 1,736 | 1,442 | 21,780 |
| Missing verification score | 0 | 1 | 1 | 0 |
| Score below 7 | 67 | 94 | 93 | 0 |
| Investigations after score >=7 | 801 | 1,641 | 1,348 | 21,780 |
| Behaviour rows after nonempty text + cleanup | 801 | 1,641 | 1,348 | 21,780 |
| Behaviour rows after exact dedup (FINAL) | 801 | 1,641 | 1,348 | 21,780 |
| Logged experiments / unfiltered cause rows | 12,605 | 24,940 | 17,534 | 167,133 |
| Experiments after score >=7 | 11,680 | 23,577 | 16,444 | 167,133 |
| Cause rows after complete summary + usable rates | 11,672 | 23,576 | 16,444 | 167,079 |
| Cause rows after absolute delta_pp >=50 | 4,490 | 7,937 | 5,517 | 64,310 |
| Cause rows after nonempty stripped edit text | 4,438 | 7,883 | 5,485 | 64,260 |
| Cause rows after exact dedup (FINAL) | 4,429 | 7,864 | 5,478 | 61,543 |

## Flags and exclusions

All flags include source, prompt_id, and experiment_number where applicable. Flags are counted as events; one row can carry multiple flags. Disagreement flags do not automatically exclude a row because labels take priority. Full counts and per-row logs are stored separately for each source.

### petri

| Flag | Count |
|---|---:|
| baseline_parsed_vs_labels_disagree | 48 |
| counterfactual_parsed_vs_labels_disagree | 9 |
| missing_edit_summary | 8 |
| no_safe_edit_text | 52 |

| Exclusion reason | Count |
|---|---:|
| excluded_behaviour_score_below7 | 67 |
| excluded_cause_effect_below50pp | 7,182 |
| excluded_cause_exact_duplicate | 9 |
| excluded_cause_incomplete | 8 |
| excluded_cause_no_safe_edit_text | 52 |
| excluded_cause_score_below7 | 925 |

Final edit types: `{"add": 407, "modify": 1775, "remove": 406, "unknown": 1841}`.

### wildchat_eval

| Flag | Count |
|---|---:|
| baseline_parsed_vs_labels_disagree | 19 |
| counterfactual_annotated_rate | 53 |
| counterfactual_parsed_vs_labels_disagree | 20 |
| experiment_n_disagrees_with_label_count | 5 |
| missing_edit_summary | 2 |
| no_safe_edit_text | 54 |

| Exclusion reason | Count |
|---|---:|
| excluded_behaviour_missing_score | 1 |
| excluded_behaviour_score_below7 | 94 |
| excluded_cause_effect_below50pp | 15,639 |
| excluded_cause_exact_duplicate | 19 |
| excluded_cause_incomplete | 1 |
| excluded_cause_no_safe_edit_text | 54 |
| excluded_cause_score_below7 | 1,363 |

Final edit types: `{"add": 864, "modify": 3274, "remove": 626, "unknown": 3100}`.

### wildchat_thinking

| Flag | Count |
|---|---:|
| baseline_parsed_vs_labels_disagree | 48 |
| counterfactual_annotated_rate | 12 |
| counterfactual_parsed_vs_labels_disagree | 15 |
| missing_edit_summary | 2 |
| no_safe_edit_text | 32 |

| Exclusion reason | Count |
|---|---:|
| excluded_behaviour_missing_score | 1 |
| excluded_behaviour_score_below7 | 93 |
| excluded_cause_effect_below50pp | 10,927 |
| excluded_cause_exact_duplicate | 7 |
| excluded_cause_missing_score | 2 |
| excluded_cause_no_safe_edit_text | 32 |
| excluded_cause_score_below7 | 1,088 |

Final edit types: `{"add": 619, "modify": 2512, "remove": 508, "unknown": 1839}`.

### wildchat_train

| Flag | Count |
|---|---:|
| missing_edit_summary | 54 |
| no_safe_edit_text | 50 |
| unusable_rates | 54 |

| Exclusion reason | Count |
|---|---:|
| excluded_cause_effect_below50pp | 102,769 |
| excluded_cause_exact_duplicate | 2,717 |
| excluded_cause_incomplete | 54 |
| excluded_cause_no_safe_edit_text | 50 |

Final edit types: `{"unknown": 61543}`.

## Files and review

Each source directory contains behaviours.parquet, causes.parquet, behaviours_unfiltered.parquet, causes_unfiltered.parquet, exclusions.jsonl, flags.jsonl, text_edits.jsonl, counts.json, and review_samples.json.

Filtered behaviours.parquet columns: prompt_id, source, verif_score, behaviour_text.

Filtered causes.parquet columns: prompt_id, source, experiment_number, is_key, edit_type, baseline_rate, counterfactual_rate, delta_pp, cause_text, cause_text_original.

Rates are proportions in [0,1]; delta_pp is signed percentage points. is_key is investigator metadata, not an inclusion filter or proof.

[Open review.html](review.html): 15 random behaviour rows and 15 random cause rows **per source**, with original versus cleaned/stripped texts side by side. Seeds start at 20261004, increasing by source index; no similarity-based selection or rerolling.

Reproduce: `clustering/.venv/bin/python clustering/preprocessing.py`. Requires PyArrow 25.0.1. Source provenance and hashes are in input_provenance.json. Old schema_audit.json and schema_missing_experiments.jsonl document the earlier preflight; counts.json describes this completed preprocessing run.

**Checkpoint 1: awaiting user review. Do not embed or cluster until approved.**
