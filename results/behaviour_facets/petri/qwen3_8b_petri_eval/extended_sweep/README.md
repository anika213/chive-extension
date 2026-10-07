# Extended PETRI behaviour clustering sweep

Same 801 PETRI investigations with verification >=7, saved L2-normalized MPNet
embeddings and Ward hierarchy. No changes to source data or earlier outputs.
Every valid k from 2 to 800 is evaluated using cosine silhouette on all records.
The original candidate scores were reproduced to within 1e-6.

The curve levels off approximately around 200–275 clusters. The maximum is
0.204086 at k=249. A contiguous near-peak range within 0.005 absolute score of
that maximum is k=217–276. That threshold is descriptive, not a statistically
estimated change point or confidence interval. Scores decline after the peak.
At k=249, median cluster size is 3, 44 clusters are singletons, and maximum size
is 16. Higher scores do not demonstrate a broad behavioural taxonomy or
establish verified causes. Singleton silhouettes are zero under sklearn.

Files: diagnostics.json/csv (all cuts), summary.json, all_cuts.npz (k and labels;
row order matches ../behaviours.jsonl), best_assignments.csv (k=249), and plot
PNG/PDF/SVG. Cluster IDs remain numeric and local to this cut.

Reproduce from workspace root:
`clustering/.venv/bin/python clustering/sweep_petri_clusters.py`
