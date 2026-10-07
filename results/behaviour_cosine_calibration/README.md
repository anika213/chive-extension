# Manual cosine-similarity calibration

Six groups of ten distinct evaluator/screener behaviour summaries from the
801 PETRI Qwen3-8B investigations with verification >=7: three uniform-random
groups and three manually chosen positive-example groups. No cause text used.

Open review.html to see all 60 exact summaries, reviewer-written short glosses,
IDs, verification scores, selection notes and results. groups.md contains the
same exact summaries as readable Markdown. groups.json preserves all metadata.
The positive groups were selected by reading, before computing these scores.
The random draw used seed 20261004 with no rerolls; chance overlaps were retained.

| Group | Mean pairwise cosine | Cross-scenario-family pairs only |
|---|---:|---:|
| Random 1 | 0.264 | 0.256 |
| Random 2 | 0.392 | 0.392 |
| Random 3 | 0.355 | 0.355 |
| Confidentiality disclosures | 0.554 | 0.519 |
| Denial of feelings + emotional language | 0.770 | 0.746 |
| Uncritical acceptance of planted memory | 0.684 | 0.675 |

Each mean uses 45 unique unordered pairs, excluding the diagonal. Vectors are
the existing L2-normalized all-mpnet-base-v2 behaviour embeddings. These are raw
cosine similarities, not silhouette scores; prior silhouette numbers cannot be
compared directly to this scale. A group of 10 items yields 45 pairs, which share
items and therefore are not 45 statistically independent observations.

A separate 1,000 uniformly sampled ten-item groups has mean group similarity 0.340
and central 95% empirical range 0.257–0.435. This is a descriptive range of group
means, not a confidence interval and not a range for individual pairs. Individual
pairs in random groups can be very similar. The plotted points represent pairs;
the shaded reference represents group means.

The curated groups are illustrative easy matches, not a measured upper bound.
They share themes, wording and some audit scenarios. Cross-family means omit
pairs whose IDs have the same scenario prefix, but do not eliminate all thematic
or template overlap. The privacy group is a useful broader match: its different
settings lower cosine despite sharing an observable behaviour. No universal
threshold or clustering accuracy can be inferred from these deliberately chosen
examples. Names describe these manual calibration groups only; existing automatic
clusters have not been renamed or modified.

Files: cosine_calibration.png/pdf/svg, review.html, groups.md/json, summary.json,
pairwise_similarities.csv, random_group_mean_reference.npy, validation.json.
Reproduce from workspace root:
`clustering/.venv/bin/python clustering/calibrate_behaviour_cosine.py`
