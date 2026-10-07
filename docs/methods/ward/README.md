# Ward hierarchical clustering

Ward builds a hierarchy by minimizing the increase in within-cluster squared Euclidean dispersion. Here it operates on normalized sentence embeddings. Cosine silhouette evaluates candidate cuts; it is not Ward's linkage metric.

The early final-explanation run uses MiniLM, token-weighted averages of 220-token chunks, and cuts at 5, 10, 15, 20, 30, 40 and 50. Early behaviour and claimed-cause runs use MPNet. Checkpoint 2 uses independent behaviour and measured-edit facets for PETRI and WildChat eval, searches every integer k=10..200, and selects by original-space cosine silhouette. All four selected cuts land at k=200, so the search does not establish an optimal number of clusters. Representation comparison step 1 fixes Ward at k=200.

Ward assigns every input; there is no noise category. IDs are local to the run and cut. See saved assignments, linkage arrays, candidate scores and nulls before interpreting a fine partition. Early claimed-cause text and later measured edits are different inputs.

## Reviews

- [results/final_explanations/review.html](../../../results/final_explanations/review.html)
- [results/behaviour_facets/review.html](../../../results/behaviour_facets/review.html)
- [results/checkpoint2/review.html](../../../results/checkpoint2/review.html)
- [results/petri_representation_variants/step1/review.html](../../../results/petri_representation_variants/step1/review.html)

## Implementation and reproduction

- [cluster_explanations.py](../../../scripts/cluster_explanations.py)
- [cluster_behaviours.py](../../../scripts/cluster_behaviours.py)
- [cluster_petri_causes.py](../../../scripts/cluster_petri_causes.py)
- [run_checkpoint2.py](../../../scripts/run_checkpoint2.py)
- [compare_petri_representations.py](../../../scripts/compare_petri_representations.py)

Run scripts from the repository root with `.venv/bin/python scripts/<script>.py`. See [reproduction and dependencies](../../REPRODUCING.md) before rerunning: most producers can overwrite their output folders. Saved output metadata describes the original execution.
