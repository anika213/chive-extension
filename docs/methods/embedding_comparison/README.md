# MPNet versus instruction-conditioned E5

Representation comparison step 1 holds the dataset fixed at 801 PETRI behaviour summaries across 147 scenario families and compares saved normalized 768-dimensional MPNet vectors with 1024-dimensional instruction-conditioned E5 vectors. E5 uses mean pooling, CPU float32 and the recorded instruction; complete texts fit without truncation.

Ward stays at k=200 and the UMAP/HDBSCAN settings remain fixed. Metrics include cluster sizes, original-space cosine silhouette, assignment coverage, ARI, scenario concentration, 200 label permutations and 10 refitted coordinate-shuffle nulls. This separates representation choice from changes to row selection. Embedding metadata and protocol files record model revisions, instruction, input hashes and package versions.

Checkpoint 3 adds Leiden, GMM and residualization on the saved representations, with no new embedding inference. Qwen weights exist locally but Qwen embedding inference and clustering remain deferred. The historical step-1 protocol's statement that later steps had not started describes that earlier snapshot.

## Reviews

- [results/petri_representation_variants/step1/review.html](../../../results/petri_representation_variants/step1/review.html)
- [results/petri_representation_variants/checkpoint3/review.html](../../../results/petri_representation_variants/checkpoint3/review.html)

## Implementation and reproduction

- [embed_petri_variants.py](../../../scripts/embed_petri_variants.py)
- [compare_petri_representations.py](../../../scripts/compare_petri_representations.py)
- [run_petri_checkpoint3.py](../../../scripts/run_petri_checkpoint3.py)

Run scripts from the repository root with `.venv/bin/python scripts/<script>.py`. See [reproduction and dependencies](../../REPRODUCING.md) before rerunning: most producers can overwrite their output folders. Saved output metadata describes the original execution.
