# Approved checkpoint 1 amendments

Only PETRI and WildChat eval are included. WildChat train and thinking are deferred.

Added artifact markers: control, sanity check, qualitatively, grader. Matching is case-insensitive, against original summaries, with the same word/marker boundaries used at revision 2. This is a lexical screen, not a determination of validity.

frac_changed = (characters deleted + characters added) / baseline_context_chars. It can exceed 1 when the edit adds substantial text. The new exclusion is frac_changed > 0.5 OR the existing source-specific p90 flag. Flagged rows remain in causes.parquet; only causes_for_clustering.parquet is embedded.

All cause-diversity statistics use positive-only investigation denominators; zero-cause investigation counts are always reported separately. No semantic cluster labels are assigned.

| Source | Artifact old | Artifact new | P90 | Relative >0.5 | Both | Either size rule | Eligible causes |
|---|---:|---:|---:|---:|---:|---:|---:|
| petri | 25 | 247 | 502 | 1224 | 501 | 1225 | 3039 |
| wildchat_eval | 72 | 700 | 882 | 2243 | 834 | 2291 | 5074 |

| Source | Fraction population | N | Mean | P25 | Median | P75 | P90 | P95 | P99 | Max | Above 0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| petri | all score-qualified experiments | 11680 | 0.2929 | 0.0186 | 0.0838 | 0.5623 | 0.9325 | 0.9648 | 0.9873 | 3.3517 | 3123 |
| petri | retained cause rows | 4429 | 0.3013 | 0.0208 | 0.0857 | 0.6021 | 0.9363 | 0.9669 | 0.9898 | 3.3517 | 1224 |
| wildchat_eval | all score-qualified experiments | 23577 | 0.2986 | 0.0231 | 0.0905 | 0.6033 | 0.9234 | 0.9592 | 0.9843 | 1.5379 | 6536 |
| wildchat_eval | retained cause rows | 7864 | 0.3079 | 0.0259 | 0.1016 | 0.6306 | 0.9243 | 0.9596 | 0.9842 | 1.3615 | 2243 |

| Source | Investigations with causes (denominator) | Zero-cause investigations (excluded) | Mean | Median | Max |
|---|---:|---:|---:|---:|---:|
| petri | 632 | 169 | 4.809 | 4 | 16 |
| wildchat_eval | 1113 | 528 | 4.559 | 4 | 18 |
