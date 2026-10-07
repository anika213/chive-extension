# Qwen3-8B behaviour facet clustering

**[Browse all behaviour texts across agglomerative cuts](agglomerative_sweep.html)**. PETRI includes every k=2–800; WildChat includes the seven originally saved cuts. All cluster members are available, not just centroid examples. This page reuses saved assignments without changing the analyses. Rebuild it from the repository root with `.venv/bin/python scripts/build_behaviour_sweep_review.py`.

This run clusters the screener's `behavior_summary`, one item per investigation, from PETRI eval and standard WildChat eval separately. It excludes WildChat training and thinking runs. PETRI is the alignment-relevant set. Only records with a matched numeric verification score >= 7 are retained. A verification score is an investigation-level filter; it does not turn a behaviour cluster into a verified cause.

Reproduce from repository root:

```bash
clustering/.venv/bin/python clustering/cluster_behaviours.py
```

The model is `sentence-transformers/all-mpnet-base-v2`, pinned to revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` in `clustering/models/all-mpnet-base-v2/download_provenance.json`. Embeddings are computed on CPU and L2-normalized. All texts are preserved verbatim; token lengths are checked against the model's 384-token input limit before encoding, with an explicit failure for overlength texts. No chunking or truncation is used.

The hierarchy is fit using `sklearn.cluster.AgglomerativeClustering(linkage="ward", metric="euclidean", n_clusters=None, distance_threshold=0, compute_distances=True)`. It is cut at 5, 10, 15, 20, 30, 40 and 50 clusters. Selection maximizes cosine silhouette, evaluated on every retained record separately within each source; ties prefer fewer clusters. All cuts are retained in `assignments.csv`; `cluster_id` is the selected cut. Numeric IDs are local to each run and cut, not matched across sources. No semantic cluster labels are assigned.

On unit vectors, squared Euclidean distance equals twice cosine distance. This makes the pairwise distances compatible, but Ward's variance-minimization objective remains different from maximizing cosine silhouette. A selection at 5 or 50 is a boundary of the tested range, not evidence of a global optimum. Cluster-count search scores are descriptive, not held-out validation. Short texts can group by topic and wording as well as behaviour.

The previous final-explanation implementation already L2-normalized its embeddings. This run changes text facet, verification filtering and embedding model together; score differences from the previous run cannot isolate the benefit of any one change.

## Outputs

- `summary.json`: counts, selected cluster counts and silhouette scores.
- `cluster_selection.png`, `.pdf`, `.svg`: silhouette versus candidate cluster count.
- `review.html`: numeric clusters with up to three examples nearest each normalized centroid.
- Per source: `behaviours.jsonl` (input text, IDs, score, original investigation/verification line numbers), `excluded.json`, `input_provenance.json` (SHA-256 hashes, coverage, exclusions and screener provenance), `embeddings.npy`, `embedding_metadata.json`, `token_lengths.json`, `linkage.npy`, `assignments.csv`, `diagnostics.json`, `cluster_examples.json`, `summary.json`.
- `runtime.json`: package versions and producing script hash.

Embeddings and linkage leaves follow `behaviours.jsonl` order, sorted by `prompt_id`. No repeated text is deduplicated: the requested analysis unit is one investigation. Input IDs and verification IDs must be unique; orphan verification rows cause failure. The pipeline verifies finite vectors, shape, unit norms, and cluster membership counts at every cut.
