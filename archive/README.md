# Retained history and intermediate artifacts

Nothing was deleted during organization. `logs/` contains the original execution logs.
`artifact_inventory.json` records the pre-organization research files, sizes, and SHA-256
hashes (large raw downloads have sizes only). It is a historical snapshot: subsequent authorized removals are listed in [removals.json](removals.json).

Historical result folders and preprocessing checkpoints remain at their original paths
because later analyses load their saved vectors, records and context diffs:

- [Early final explanations](../results/final_explanations/README.md)
- [Early behaviour facets](../results/behaviour_facets/README.md)
- [Early claimed-cause facets](../results/cause_facets/README.md)
- [Cause null/stability evaluation](../results/cause_evaluation/README.md)
- [Checkpoint 2](../results/checkpoint2/README.md)
- [Preprocessing checkpoints and input policy](../data/README.md)
- [Original running research notes](../docs/history.md)

Candidate future cuts: old logs, parameter sweeps, average-linkage pilots and duplicate
review exports. The representation-comparison branch has been removed as requested. Keep reviews, assignments, provenance and
necessary input artifacts until a specific retention decision is made.
