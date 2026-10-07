# PETRI cause facet: investigator-supported statements

The same 801 Qwen3-8B PETRI investigations with investigation-level verification score >=7 are used as in the behaviour analysis. One combined cause-facet item is formed per investigation from the exact statement/headings inside `structured_findings.structured_answer` → `Core causes` with exactly `Status: supported`. Statement text before the status is preserved verbatim, excluding numbering; multiple statements are joined with newlines. No statement is rewritten or semantically labelled.

The extraction includes 1,790 supported statements (55 investigations with one, 503 with two, 243 with three). It omits 195 partially supported statements and one refuted statement found inside Core causes. All 801 investigations have at least one supported statement. Observed behaviour, evidence narrative, negative evidence/refuted hypotheses outside Core causes, and repeated final answers are not embedded. The other 67 PETRI investigations are excluded for scores below seven.

**Supported is the investigator's claimed status, not independently established causal ground truth.** The >=7 filter applies to whole investigations, not individual cause claims. Measured classifier outcomes are saved separately in `measured_outcomes_audit.jsonl` but are not embedding features or inputs to cluster construction. No automatic per-claim experiment mapping or causal validation is asserted. Original experiments and complete prompts remain in the source investigation file; source line numbers are saved.

## Method and results

The same pinned `sentence-transformers/all-mpnet-base-v2` revision `e8c3b32edf5434bc2275fc9bab85f82640a19130` is run on CPU with L2 normalization. Cause facets contain 18–131 tokens (median 65), including special tokens, below the 384-token limit. No truncation or chunking occurs.

`sklearn.cluster.AgglomerativeClustering` fits a complete Ward hierarchy using Euclidean distance. Every valid cut k=2..800 is scored with cosine silhouette on all 801 items. On unit vectors squared Euclidean distance is twice cosine distance; Ward's variance objective remains different from cosine silhouette optimization.

Peak: **k=336, silhouette 0.123968**. The contiguous range within 0.005 absolute score of the peak is **k=278..389**, a broad near-peak region. This tolerance is descriptive, not a confidence interval or formal change-point estimate. At the peak the median cluster has 2 investigations, 91 clusters are singletons, and the largest has 8. sklearn assigns singletons zero silhouette. Searching all cuts and reporting the maximum is descriptive, not held-out validation or an established number of causes.

For the identical 801 IDs, the behaviour-facet peak is k=249, silhouette 0.204086, with a near-peak range 217..276. The cause statement representation is less separated by this metric. This is not proof that causal categories do or do not exist. Combining multiple claims into one vector averages distinct mechanisms, while exact claims can retain specific topic or prompt language. Similarity in investigator claims does not itself establish shared verified causes.

## Artifacts and reproduction

Run from repository root:

```bash
clustering/.venv/bin/python clustering/cluster_petri_causes.py
```

- `causes.jsonl`: exact included facet and all Core causes entries with statuses, inclusion flags, source character spans, investigation and verification line numbers, and structured-answer hashes.
- `input_provenance.json`, `excluded.json`: hashes, extraction scope, coverage and exclusions.
- `measured_outcomes_audit.jsonl`: classifier question, baseline outcomes, experiment grader counts/verdicts and edits, kept separately from embedded text.
- `embeddings.npy`, `embedding_metadata.json`, `token_lengths.json`: vectors, model revision, normalization and length checks.
- `linkage.npy`: Ward hierarchy.
- `all_cuts.npz`: arrays `k` and `labels`; label row order matches `causes.jsonl`, sorted by prompt ID.
- `diagnostics.csv/json`: all 799 cuts with cosine silhouettes, cluster sizes and singleton diagnostics.
- `best_assignments.csv`, `cluster_examples.json`, `review.html`: selected numeric clusters and up to three examples nearest each normalized centroid.
- `cause_vs_behaviour_sweep.png/pdf/svg`: comparable curves on identical investigations.
- `summary.json`, `runtime.json`, `validation.json`: results and validation.

Cluster IDs are local to the selected source/facet/cut, not aligned with behaviour clusters. No cross-tab or semantic cluster labels were produced.
