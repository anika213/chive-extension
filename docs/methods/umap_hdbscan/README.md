# UMAP + HDBSCAN

UMAP reduces normalized embeddings to 10 dimensions; HDBSCAN finds density-connected groups and can reject points as noise (`-1`). The checkpoint protocol uses cosine UMAP, 30 neighbours, min_dist=0 and seed 20261004, followed by Euclidean HDBSCAN with min_cluster_size=10, min_samples=5 and EOM selection. The historical cause evaluation instead uses seed 42 and includes a 90-run parameter grid; read its saved design rather than treating every run as identical.

The main score is cosine silhouette in the original embedding space, restricted to assigned rows. Always read it with assignment coverage and cluster sizes. UMAP-space silhouette measures a different geometry. Noise is not a coherent cluster; agreement primarily uses jointly assigned rows, with overlap reported separately.

The latest-qualifying-checkpoint run uses a 269-record subset, with 172 records assigned to six behaviour clusters and 97 noise rows. Its cause facet is fitted separately. Synthetic embedding nulls show what the pipeline can create from shuffled geometry, not whether clusters represent verified mechanisms.

## Reviews

- [results/petri_latest_qualified/review.html](../../../results/petri_latest_qualified/review.html)
- [results/checkpoint2_locations/review.html](../../../results/checkpoint2_locations/review.html)
- [results/cause_evaluation/reading/review.html](../../../results/cause_evaluation/reading/review.html)

## Implementation and reproduction

- [run_checkpoint2.py](../../../scripts/run_checkpoint2.py)
- [recluster_latest_petri.py](../../../scripts/recluster_latest_petri.py)
- [evaluate_cause_hdbscan.py](../../../scripts/evaluate_cause_hdbscan.py)

Run scripts from the repository root with `.venv/bin/python scripts/<script>.py`. See [reproduction and dependencies](../../REPRODUCING.md) before rerunning: most producers can overwrite their output folders. Saved output metadata describes the original execution.
