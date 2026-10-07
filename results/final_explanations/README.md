# Investigator final-explanation clustering

This analysis clusters the investigator's **claimed explanations**. It does not
establish verified causes, cluster behaviour summaries, produce a cross-tab, or
assign semantic names to clusters.

## Inputs and separation

Each of the four downloaded Qwen3-8B runs is fitted independently. PETRI is the
alignment-relevant source; WildChat runs are kept separate from PETRI and from
one another. Cluster IDs are local to a run and cannot be compared across runs.

The input is the `Final answer` section of
`structured_findings.structured_answer`. Five training records use the explicit
`structured_findings.answer` fallback. Missing final explanations are excluded
and listed in each run's `excluded.json`. No verification-score filter is applied;
scores are carried into the output for later analysis. These are all available
final explanations, not a subset asserted to represent harmful behaviour.

## Method

1. Embed text locally with `sentence-transformers/all-MiniLM-L6-v2`, revision
   `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
2. Split long texts into non-overlapping 220-token chunks, below the model's
   context limit. Average normalized chunk vectors with token-count weights,
   then normalize the document vector. No final explanation is intentionally
   truncated at the model context limit.
3. Fit `sklearn.cluster.AgglomerativeClustering` with Ward linkage, Euclidean
   distance on normalized embeddings, and a full hierarchy (`n_clusters=None`, `distance_threshold=0`).
4. Cut the hierarchy at 5, 10, 15, 20, 30, 40, and 50 clusters. Select the highest
   cosine silhouette score among these cuts (ties prefer fewer clusters).
   Evaluate on all records for runs up to 2,000 records; otherwise use a fixed
   random sample of 2,000 (seed 42). Save every candidate cut.
5. Save numeric IDs and up to three examples closest to each cluster's embedding
   centroid. Examples are for inspection, not semantic cluster labels.

Selection by silhouette is an exploratory heuristic, not proof of a natural
number of causes. Similarity may reflect shared topics, phrasing, or prompt
context as well as causal explanations. Cluster sizes and singleton counts are exposed in diagnostics. The initial
average-linkage/cosine pilot produced a dominant cluster containing over 99% of
standard WildChat eval records. Its results are retained per run under
`average_linkage_pilot/`; Ward is the primary result. Ward minimizes increases
in within-cluster squared Euclidean dispersion. Cosine is used to evaluate
the candidate cuts, not as Ward's linkage metric. For the training run, silhouette
scores are sampled and can omit small clusters. No stability claim is made.

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

## Outputs per run

- `explanations.jsonl`: exact input texts, IDs, provenance and verification scores.
- `excluded.json`: investigations without a usable final explanation.
- `embeddings.npy`, `embedding_metadata.json`: reusable vectors and provenance.
- `linkage.npy`: full hierarchy in SciPy linkage format.
- `assignments.csv`: selected cluster ID and IDs for every candidate cluster count.
- `diagnostics.json`: silhouette scores, cluster sizes and singleton counts.
- `cluster_examples.json`: numeric clusters, member counts and example explanations.
- `summary.json`: selected settings and scope.

From the workspace root, rerun with:

```bash
HF_HUB_OFFLINE=1 clustering/.venv/bin/python clustering/cluster_explanations.py
```

Dependencies are pinned in `clustering/requirements.lock.txt`; the embedding model
is stored in `clustering/models/all-MiniLM-L6-v2/`. Original data is unchanged.

## Results

Open `review.html` for numeric cluster sizes and representative explanations.

| Run | Explanations | Selected clusters | Cosine silhouette |
|---|---:|---:|---:|
| PETRI eval | 868 | 50 | 0.072 |
| WildChat eval | 1,735 | 5 | 0.009 |
| WildChat thinking eval | 1,441 | 50 | 0.010 |
| WildChat train | 21,780 | 10 | 0.094 |

Separation is weak under these embeddings, especially for WildChat eval and
thinking eval. Two selected counts are at the upper end of the tested range;
50 is the best tested cut for those runs, not an estimated true number of causes.
The primary Ward cuts have no singleton clusters. Every assignment, source/run
membership, cluster count, and representative-example membership was checked;
see `validation.json`. Two records without final explanations were excluded.

The retained average-linkage pilot is a method comparison, not another dataset.
