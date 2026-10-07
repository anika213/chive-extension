# Checkpoint 2 reproduction

Review [results](../results/checkpoint2/report.md), [methods and metrics](../results/checkpoint2/report.md),
[approved preprocessing and 30 original summaries per source](../data/checkpoint1_approved/preprocessing_log.md),
and [validation](../results/checkpoint2/validation.json).

Run from the repository workspace root, using the existing local model and revision-2 context diffs:

```bash
clustering/.venv/bin/python clustering/scripts/prepare_checkpoint2.py
clustering/.venv/bin/python clustering/scripts/run_checkpoint2.py
clustering/.venv/bin/python clustering/scripts/finish_checkpoint2_report.py
clustering/.venv/bin/python clustering/scripts/validate_checkpoint2.py
```

Only PETRI and WildChat eval are processed. Embeddings use local all-mpnet-base-v2
on CPU, with L2 normalization. No generative LLM calls are made. Each source and
facet has independent Ward (all integer k=10..200) and UMAP/HDBSCAN runs, plus ten
coordinate-shuffled embedding null replicates. The saved design records parameters.

Flagged cause rows remain in the approved `causes.parquet`; the eligible subset is
`causes_for_clustering.parquet`. Original summaries are retained for review; only
stripped edit descriptions are embedded. All diversity statistics report positive
denominators and excluded zero-cause counts. Near duplicates are reported and retained.

All four Ward maxima occur at the search boundary k=200; these results do not
establish an optimal cluster count or a plateau beyond that range. Above-null
silhouette indicates embedding structure, not validated causal explanations.

Stopped at Checkpoint 2 before semantic cluster labelling. Train and thinking are deferred.
