# Gaussian mixture model

The checkpoint-3 GMM fits the saved 10-dimensional E5 UMAP coordinates for the same 801 PETRI behaviour texts. Settings: 20 components, full covariance, one initialization, k-means initialization, seed=20261004, max_iter=500, tol=0.001 and reg_covar=1e-6.

Hard assignments select the highest-posterior component. All 20 per-item posterior probabilities, mixture weights, means and covariances are saved under `results/petri_representation_variants/checkpoint3/e5_gmm/`. Unlike HDBSCAN, this fit assigns every item. A high posterior expresses confidence under this fitted mixture; it is not confidence in a semantic interpretation.

There is one observed setting, with no component-count search or stability claim. Shared diagnostics include original-space cosine silhouette, cluster sizes, concentration, 200 scenario permutations and 10 refitted geometry nulls. UMAP geometry influences the mixture and should not be treated as the original text space.

## Reviews

- [results/petri_representation_variants/checkpoint3/review.html](../../../results/petri_representation_variants/checkpoint3/review.html)

## Implementation and reproduction

- [run_petri_checkpoint3.py](../../../scripts/run_petri_checkpoint3.py)

Run scripts from the repository root with `.venv/bin/python scripts/<script>.py`. See [reproduction and dependencies](../../REPRODUCING.md) before rerunning: most producers can overwrite their output folders. Saved output metadata describes the original execution.
