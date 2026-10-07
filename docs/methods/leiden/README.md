# Leiden graph communities

The checkpoint-3 Leiden result uses the same 801 PETRI behaviour texts and saved instruction-conditioned E5 embeddings as the other comparisons. It constructs a cosine-nearest-neighbour graph with k=15, symmetrizes by undirected union, and uses cosine edge weights. Leiden uses RBConfigurationVertexPartition, resolution=1.0, seed=20261004 and optimization to convergence.

This is one observed-data setting, not a graph-parameter sweep or a stability study. Communities receive numeric IDs. Read original-space cosine silhouette, size distribution, scenario concentration, 200 scenario-label permutations and 10 refitted embedding-shuffle nulls together.

The >15% diagnostic removes whole oversized clusters once using the original assigned denominator, then recomputes concentration. It does not change the saved clustering. Repeated checkpoints and scenario families are dependent observations. Lower concentration or greater graph modularity alone does not validate behavioural categories.

## Reviews

- [results/petri_representation_variants/checkpoint3/review.html](../../../results/petri_representation_variants/checkpoint3/review.html)

## Implementation and reproduction

- [run_petri_checkpoint3.py](../../../scripts/run_petri_checkpoint3.py)

Run scripts from the repository root with `.venv/bin/python scripts/<script>.py`. See [reproduction and dependencies](../../REPRODUCING.md) before rerunning: most producers can overwrite their output folders. Saved output metadata describes the original execution.
