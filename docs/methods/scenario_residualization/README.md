# Scenario-mean residualization + HDBSCAN

This is a representation transformation followed by the unchanged UMAP/HDBSCAN protocol, not a new clustering algorithm. For MPNet and E5 separately, subtract each scenario-family mean from its normalized vectors, renormalize, then refit UMAP/HDBSCAN. The scenario key is a provisional prefix derived from PETRI IDs.

The 12 singleton-family rows remain unchanged; 23 two-member families become antipodal pairs by construction. Residualization can remove real behaviour shared within a scenario as well as scenario wording. Residual-space silhouette measures relative deviations; an original-space score is also reported.

Checkpoint 3 retains both variants and their baselines. The saved status notes that the E5 residual result separates the 789 transformed rows from the 12 unchanged singleton rows, so that result is not recommended. Scenario labels helped construct the representation: their permutation comparison is descriptive, not an independent significance test. Geometry nulls shuffle the final normalized residual vectors; they do not repeat residualization or preserve its induced within-family constraints.

## Reviews

- [results/petri_representation_variants/checkpoint3/review.html](../../../results/petri_representation_variants/checkpoint3/review.html)

## Implementation and reproduction

- [run_petri_checkpoint3.py](../../../scripts/run_petri_checkpoint3.py)

Run scripts from the repository root with `.venv/bin/python scripts/<script>.py`. See [reproduction and dependencies](../../REPRODUCING.md) before rerunning: most producers can overwrite their output folders. Saved output metadata describes the original execution.
