# CHIVE — Checkpoint 2

**Stopped before semantic cluster labelling.**

Only PETRI and WildChat eval; separate behaviour and cause embeddings/clusters. Train and thinking deferred. Inputs: [approved preprocessing](../location_seed/review.html).

## Fixed protocol

- all-mpnet-base-v2, local CPU inference, L2 normalization. No truncation or chunking. No generative LLM calls.
- Ward uses Euclidean distance on unit vectors. Every k=10..200 is scored with full-data cosine silhouette; highest silhouette selects the reported cut. The full curve is retained. A peak at k=200 is a search boundary, not an established optimum.
- UMAP: cosine metric, 10 dimensions, n_neighbors=30, min_dist=0, seed 20261004. HDBSCAN: Euclidean in reduced space, min_cluster_size=10, min_samples=5, EOM. Preset, not tuned to silhouette.
- Main HDBSCAN silhouette is cosine in the original embedding space among assigned rows; noise is excluded and coverage reported. UMAP-space silhouette is only a secondary diagnostic.
- Ten null replicates independently shuffle each embedding coordinate across rows, then L2 normalize and refit BOTH pipelines. Simply shuffling whole rows would preserve geometry and is not used. Ward nulls undergo the same k sweep, accounting for cut selection. Ten replicates give a rough reference range, not a precise significance estimate.
- ARI compares Ward’s selected cut with HDBSCAN on non-noise rows. An all-row ARI treating noise as one label is separately provided, not interpreted as a meaningful noise cluster.
- Cosine >0.95 near-duplicate pairs are reported, never dropped. Source and facet cluster IDs are independent. No semantic labels are assigned.
- Cause-diversity means/medians exclude investigations with zero assigned cause clusters; preprocessing zero-eligible-cause counts and HDBSCAN-only-noise investigations are reported separately.

## Results

| Source | Facet | N | Ward k | Ward cosine | Null max-over-k mean | HDBSCAN clusters | HDB cosine | HDB noise | ARI assigned |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| petri | behaviours | 801 | 200 | 0.1860 | 0.0169 | 22 | 0.0870 | 24.8% | 0.1676 |
| petri | causes | 3030 | 197 | 0.0913 | 0.0012 | 75 | 0.1092 | 32.6% | 0.3725 |
| wildchat_eval | behaviours | 1641 | 200 | 0.0447 | 0.0072 | 30 | 0.0729 | 48.2% | 0.1635 |
| wildchat_eval | causes | 5068 | 200 | 0.0604 | -0.0020 | 132 | 0.1209 | 39.3% | 0.4449 |

Full cluster sizes, both null summaries, positive-only diversity statistics, membership assignments, all cuts, and reading samples are saved per source/facet. Review [review.html](review.html) for 5 random members from each of the 10 largest clusters for both methods (all clusters if fewer than 10).

**Checkpoint 2: awaiting review; no labelling performed.**

## Additional diagnostics

Null ranges below are the minimum and maximum of 10 coordinate-shuffled embedding replicates. HDBSCAN silhouette is always computed in the original embedding space on assigned rows; compare it alongside noise fraction, not alone.

| Source | Facet | HDBSCAN cosine | Shuffled HDB cosine mean | Null range | Valid null runs | Observed noise | Null noise mean |
|---|---|---:|---:|---|---:|---:|---:|
| petri | behaviours | 0.0870 | -0.0012 | -0.0045 to 0.0074 | 10 | 24.8% | 45.5% |
| petri | causes | 0.1092 | -0.0096 | -0.0115 to -0.0010 | 10 | 32.6% | 50.4% |
| wildchat_eval | behaviours | 0.0729 | -0.0064 | -0.0099 to 0.0030 | 10 | 48.2% | 51.4% |
| wildchat_eval | causes | 0.1209 | -0.0109 | -0.0132 to 0.0067 | 10 | 39.3% | 52.7% |

### Cluster sizes

| Source | Facet | Method | Clusters | Smallest | Median | Largest | Singletons |
|---|---|---|---:|---:|---:|---:|---:|
| petri | behaviours | ward | 200 | 1 | 4.0 | 15 | 16 |
| petri | behaviours | hdbscan | 22 | 10 | 22.0 | 74 | 0 |
| petri | causes | ward | 197 | 4 | 14 | 53 | 0 |
| petri | causes | hdbscan | 75 | 10 | 21 | 135 | 0 |
| wildchat_eval | behaviours | ward | 200 | 2 | 7.0 | 30 | 0 |
| wildchat_eval | behaviours | hdbscan | 30 | 10 | 17.0 | 152 | 0 |
| wildchat_eval | causes | ward | 200 | 5 | 21.0 | 93 | 0 |
| wildchat_eval | causes | hdbscan | 132 | 10 | 14.5 | 138 | 0 |

### Near duplicates

All pairs above cosine 0.95 are saved in the per-run near_duplicates.parquet files. No near duplicates were removed.

| Source | Facet | Pairs | Rows involved |
|---|---|---:|---:|
| petri | behaviours | 0 | 0 |
| petri | causes | 17 | 34 |
| wildchat_eval | behaviours | 0 | 0 |
| wildchat_eval | causes | 90 | 137 |

### Distinct cause clusters per investigation — positive-only denominators

Zero-eligible-cause investigations are excluded from every diversity mean/median. For HDBSCAN, investigations whose eligible rows are all noise also have no assigned cause clusters; they are counted separately and excluded from the cluster-diversity denominator.

| Source | Method | Positive denominator | Zero eligible causes | Eligible but only noise | Mean clusters | Median clusters | Max clusters |
|---|---|---:|---:|---:|---:|---:|---:|
| petri | ward | 632 | 169 | 0 | 2.3718 | 2.0000 | 9 |
| petri | hdbscan | 533 | 169 | 99 | 1.6417 | 1.0000 | 7 |
| wildchat_eval | ward | 1113 | 528 | 0 | 1.8913 | 2.0000 | 8 |
| wildchat_eval | hdbscan | 826 | 528 | 287 | 1.4274 | 1.0000 | 6 |

### Reading the results

Ward cuts at the upper search boundary: petri/behaviours, wildchat_eval/behaviours, wildchat_eval/causes. These runs do not establish an optimal k or a plateau beyond 200.

Agreement between the methods reflects different cluster counts and HDBSCAN coverage as well as similarities in grouping. Higher-than-null silhouette supports structure in these embeddings; it does not validate a causal explanation or establish semantically coherent mechanisms.

Cause rows are repeated experiments within investigations, so they are not independent investigations. Read the sampled members and near-duplicate report before interpreting fine clusters.

The 30-row manual review samples are original investigator summaries. Only stripped edit descriptions were embedded. No semantic cluster labels have been assigned.
