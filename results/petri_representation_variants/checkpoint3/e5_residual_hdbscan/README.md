# e5_residual_hdbscan

Scenario-mean residualized E5 followed by the fixed UMAP/HDBSCAN protocol. Saved analysis does not recommend this variant because it separates transformed rows from unchanged singletons.

All variants use the same 801 behaviour texts and 147 scenario families.

- [Comparison review](../review.html)
- [Method guide](../../../../docs/methods/scenario_residualization/README.md)
- [Metrics](metrics.json)
- [Random member samples](samples.json)
- `assignments.parquet`: per-record numeric labels and saved membership information.
- `embeddings.npy`: vectors used by this variant, in the same input order.
- `scenario_null_replicates.json` and `embedding_null_replicates.json`: reference diagnostics.

The shared [protocol](../protocol.json) records exact settings. These files are preserved output, not a claim that this variant is superior.
