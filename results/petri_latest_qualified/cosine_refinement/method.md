# Cosine refinement of current PETRI behaviours

Input: 269 current PETRI behaviour records and existing normalized MPNet embeddings.
Previous records, vectors, assignments, and cause analyses remain unchanged.

## Within existing clusters
Average linkage on original cosine distance. Force a split of each parent at its
root so outputs are proper subgroups. Recursively accept the first descendant
with mean off-diagonal pairwise cosine strictly >0.4 and enough members; otherwise
split again. Groups smaller than the minimum become residual. Two minimum sizes,
2 and 10, are provided. Disjoint groups partition a subset of the original 172
assigned records. Original 97 noise records are excluded only from this part.
A mean threshold does not constrain every pair or every individual member.

## Global candidate search
Fit Ward in normalized Euclidean space and average/complete linkage in cosine
distance. Inspect every cut k=2..268. Discard groups smaller than 10. For each
remaining partition, try mean-cohesion targets just above .4 and .425 through .7
in .025 steps, plus .75, .8, .85, .9, .95: remove the least-central member until
the target is reached, or reject the group if fewer than 10 remain. Also try
centrality-trimmed cores capped at 10, 12, 15 and 20 members. At each variant,
evaluate the path of removing the least-cohesive cluster until two remain. Apply the
same refinements to the original HDBSCAN groups. Deduplicate identical
partitions; retain deterministic first-generation provenance.

At least two clusters are required: silhouette is undefined for one cluster.
All silhouettes use the original cosine distances on assigned rows only, with
noise excluded; individual scores are independently checked against sklearn.
Recommend the maximum-coverage candidate within 0.02 absolute of the best
worst-cluster mean pairwise cosine. Ties prefer worst-cluster mean cosine, then
average cluster cosine. Silhouette never influences selection or rejection.
Also save the strict best and maximum-coverage candidates with every cluster
mean cosine >.4, >.5 and >.6, if feasible. These floors are
explicit operational choices, not universally calibrated quality thresholds.

These are in-sample exploratory selections. They are not a global optimum,
held-out validation, a null test, or evidence for a taxonomy of causes. No
semantic cluster names or generative-model calls are involved. Minimum sizes
apply to investigations, not distinct scenario families. Similarities may reflect
topic, wording or repeated scenario structure. Group IDs are local to each result.
