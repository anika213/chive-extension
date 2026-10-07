# Checkpoint 1: PETRI representation comparison

Stopped for review. No Step 2 or 3 runs.

|Representation|Method|Clusters|Noise|Silhouette|Embedding-null|Top seed share|Scenario-null|Excess|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|mpnet_baseline|ward|200|0.0%|0.186|0.017|67.3%|26.6%|+40.6 pp|
|mpnet_baseline|hdbscan|22|24.8%|0.087|-0.001|27.4%|7.5%|+19.9 pp|
|e5_instruct|ward|200|0.0%|0.200|0.012|71.4%|26.6%|+44.8 pp|
|e5_instruct|hdbscan|18|21.2%|0.096|-0.005|22.7%|6.4%|+16.3 pp|

e5_instruct / ward: excess seed concentration changed by +4.2 percentage points relative to MPNet (negative is less concentration).
e5_instruct / hdbscan: excess seed concentration changed by -3.7 percentage points relative to MPNet (negative is less concentration).

See review.html for all metrics and 5 random examples per top-10 cluster.
