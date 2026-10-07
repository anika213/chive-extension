# CHIVE — Checkpoint 2

**Stopped before semantic cluster labelling.**

PETRI only: latest qualifying checkpoint per scenario/run, 269 investigations across 147 scenario families. Behaviours and causes clustered separately. WildChat unchanged. Inputs: [selection rules and counts](selection.md).

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
| petri | behaviours | 269 | 107 | 0.1460 | 0.0255 | 6 | 0.0988 | 36.1% | 0.0861 |
| petri | causes | 1052 | 200 | 0.2093 | 0.0128 | 28 | 0.1356 | 38.9% | 0.2296 |

Full cluster sizes, both null summaries, positive-only diversity statistics, membership assignments, all cuts, and reading samples are saved per source/facet. Review [review.html](review.html) for 5 random members from each of the 10 largest clusters for both methods (all clusters if fewer than 10).

**Checkpoint 2: awaiting review; no labelling performed.**

## Cluster sizes and cause coverage

| Facet | Method | Clusters | Minimum size | Median size | Maximum size | Singletons |
|---|---|---:|---:|---:|---:|---:|
| behaviours | ward | 107 | 1 | 2 | 7 | 25 |
| behaviours | hdbscan | 6 | 10 | 28.5 | 56 | 0 |
| causes | ward | 200 | 2 | 5.0 | 13 | 0 |
| causes | hdbscan | 28 | 10 | 20.0 | 50 | 0 |

Cause-diversity counts exclude 56 selected investigations with zero eligible causes. HDBSCAN additionally excludes all-noise investigations from assigned-cluster diversity; these counts are in the cause summary JSON.

With only 269 behaviour rows, high k cuts contain many small clusters. Silhouette selection does not establish semantic usefulness; inspect sizes and sampled members. The two runs per scenario remain related observations.

All sample rows below show unchanged embedding text. Cause originals and measured outcomes are supplied for review. No semantic labels were assigned.
