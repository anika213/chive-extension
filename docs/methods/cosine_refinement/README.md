# Cosine-cohesive hierarchical refinement

This is a family of exploratory selections from Ward, average-linkage and complete-linkage trees. Ward uses Euclidean normalized vectors; average/complete linkage use cosine distance. A group must pass a mean pairwise cosine threshold; that does not guarantee every pair passes it.

For the 269 latest PETRI records, the within-parent analysis recursively splits the six existing HDBSCAN groups using average linkage. It retains proper subgroups with mean cosine >0.4 and minimum sizes of either 2 or 10. The 97 original noise records are outside this within-parent analysis.

The separate global search includes all 269 records. It examines cuts, rejects groups smaller than 10, trims peripheral members, and requires at least two surviving groups. The recommended candidate maximizes coverage within 0.02 of the best worst-cluster mean cosine. Strict-best and coverage-at-threshold alternatives are retained. Silhouette is diagnostic, not the selection criterion.

Small scenario analyses use a minimum of two members and their own saved candidate selections: 8 medical-authority investigations; 19 investigations across three related scenarios; 36 eligible measured edits; and 19/34 verifier assessments across three/five scenarios. These use all qualifying released checkpoints, not the latest-only 269-record selection. Check each summary for preset thresholds and actual method. A preset called `cosine_first` is a selection rule, not a distinct clustering algorithm.

## Reviews

- [results/petri_latest_qualified/cosine_refinement/review.html](../../../results/petri_latest_qualified/cosine_refinement/review.html)
- [results/scenario_clustering/medical_authority_sycophancy/review.html](../../../results/scenario_clustering/medical_authority_sycophancy/review.html)
- [results/scenario_clustering/related_health_counterfactuals/review.html](../../../results/scenario_clustering/related_health_counterfactuals/review.html)
- [results/scenario_clustering/five_scenario_explanation_assessments/review.html](../../../results/scenario_clustering/five_scenario_explanation_assessments/review.html)

## Implementation and reproduction

- [refine_behaviour_clusters.py](../../../scripts/refine_behaviour_clusters.py)
- [cluster_one_scenario.py](../../../scripts/cluster_one_scenario.py)
- [cluster_related_counterfactuals.py](../../../scripts/cluster_related_counterfactuals.py)
- [cluster_explanation_assessments.py](../../../scripts/cluster_explanation_assessments.py)

Run scripts from the repository root with `.venv/bin/python scripts/<script>.py`. See [reproduction and dependencies](../../REPRODUCING.md) before rerunning: most producers can overwrite their output folders. Saved output metadata describes the original execution.
