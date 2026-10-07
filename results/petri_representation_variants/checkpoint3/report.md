# PETRI behaviour comparison — checkpoint 3

Stopped for review. Ward excluded; Qwen deferred; no generative LLM or new embedding calls.

|Variant|Clusters|Noise|Largest: size / assigned share|Cosine silhouette / null|Seed share / null|Seed excess|After >15% exclusion: share / null|Trimmed excess|Rows retained after exclusion|
|---|---|---|---|---|---|---|---|---|---|
|mpnet_hdbscan|22|24.8%|74 / 12.3%|0.087 / -0.001|27.4% / 7.5%|+19.9 pp|27.4% / 7.5%|+19.9 pp|602|
|e5_hdbscan|18|21.2%|196 / 31.1%|0.096 / -0.005|22.7% / 6.4%|+16.3 pp|30.8% / 7.9%|+22.9 pp|435|
|e5_leiden|9|0.0%|171 / 21.3%|0.073 / -0.001|11.0% / 4.2%|+6.8 pp|13.5% / 4.8%|+8.7 pp|483|
|e5_gmm|20|0.0%|75 / 9.4%|0.082 / -0.004|19.7% / 6.2%|+13.5 pp|19.7% / 6.2%|+13.5 pp|801|
|mpnet_residual_hdbscan|10|48.8%|126 / 30.7%|0.042 / -0.002|6.8% / 5.9%|+0.9 pp|9.8% / 7.7%|+2.0 pp|205|
|e5_residual_hdbscan|2|0.0%|789 / 98.5%|0.011 / -0.002|2.1% / 2.1%|-0.0 pp|8.3% / 12.2%|-3.8 pp|12|

Largest shares use assigned rows. Excess values are percentage-point differences. See review.html for full diagnostics and random examples.
