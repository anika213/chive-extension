# Average and complete linkage on cosine distance

These agglomerative methods start with individual records and repeatedly merge groups. Average linkage uses mean inter-group distance; complete linkage uses the largest inter-group distance. The refinement/scenario scripts build both trees from `1 - cosine_similarity` of normalized MPNet embeddings, inspect cuts, and apply minimum-size and mean-cohesion rules afterward.

They differ from Ward, which minimizes variance in Euclidean space, and from HDBSCAN, which estimates density and supports noise. The raw hierarchies partition all rows, but the later selective refinement can leave rows unassigned. A mean-cohesion filter does not imply a bound on every within-cluster pair.

The historical final-explanation average-linkage pilot produced a dominant group containing more than 99% of standard WildChat eval records. Its `average_linkage_pilot/` outputs are preserved inside the per-run final-explanation folders; Ward became the primary result. This pilot used a different text facet/model from the later MPNet scenario refinements.

- [Cosine-refinement selection rules](../cosine_refinement/README.md)
- [Refinement review](../../../results/petri_latest_qualified/cosine_refinement/review.html)
- [Scenario analyses](../../../results/scenario_clustering/README.md)
- [Historical pilot notes](../../../results/final_explanations/README.md)
- [Implementation](../../../scripts/refine_behaviour_clusters.py)
- [Reproduction](../../REPRODUCING.md)

Use saved per-preset summaries to identify which tree supplied a reported group. Preset names describe selection priorities, not a guarantee that every preset uses the same linkage. All selections are exploratory and in-sample.
