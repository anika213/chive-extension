# PETRI cause-facet evaluation

The same 801 Qwen3-8B PETRI eval investigations with verification score >=7 are
used throughout. Each input combines the investigator's explicitly supported
Core causes statements. These are claimed reasons, not verified causal ground
truth. All outputs retain numeric IDs; no semantic cluster names are assigned.

## Revised interpretation

The cause embeddings show structure well above the two tested artificial nulls.
A low absolute silhouette alone was insufficient grounds for dismissing that
structure. However, neither the null tests nor the stability checks establish
a discrete taxonomy of causes. Reading reveals both mechanism-related language
and subject-matter grouping. Exact memberships depend on sampling and embedding
model, and several items are turns from the same PETRI scenario.

## Null baselines

Ward observed maximum silhouette is 0.123968. Each random-embedding replicate
refits the hierarchy and repeats the entire k=2..800 search, accounting for
selection of the best k:

| Null | Replicates | Mean maximum | Largest maximum |
|---|---:|---:|---:|
| Isotropic unit vectors | 20 | 0.033470 | 0.034294 |
| Independently permuted coordinates, renormalized | 20 | 0.026284 | 0.027929 |

No null replicate exceeds observed. With only 20 replicates per family, the
plus-one empirical upper-tail proportion is 1/21=0.0476, not zero. These are
synthetic nulls, not matched controls for repeated topics, templates or scenarios.
A separate 200-permutation, size-preserving label baseline is also saved; it is
an easier comparison and does not redo cluster-count selection. Row shuffling
and reclustering would preserve geometry and is not used as a null.

## Stability

Ward is deterministic for fixed inputs, so changing a seed alone cannot test
its robustness. Twenty bootstrap samples and twenty 80% subsamples were used;
ARI compares matching original IDs only. Different-model agreement uses MiniLM
versus MPNet on the identical text, normalized and without truncation.

| Comparison with full reference | Ward k50 | Ward k336 |
|---|---:|---:|
| Bootstrap, mean ARI | 0.307 | 0.376 |
| 80% subsamples, fixed k, mean ARI | 0.443 | 0.677 |
| MiniLM vs MPNet ARI | 0.220 | 0.318 |

The bootstrap contains about 510 unique investigations; keeping k336 fixed makes
it relatively finer. Proportionally scaled subsample results and all pairwise
ARIs are included in `null_stability/`. MiniLM's k336 silhouette is 0.122 despite
membership agreement with MPNet of only 0.318: similar scores need not mean
similar partitions.

## UMAP + HDBSCAN

The declared primary preset is 10-dimensional UMAP, cosine metric, n_neighbors=30,
min_dist=0, seed=42; HDBSCAN uses Euclidean distance, min_cluster_size=10,
min_samples=5, and EOM selection. It was not picked by maximizing silhouette.
A 90-run grid varies neighbors 15/30/50, cluster sizes 10/20/30, min_samples 5/10,
and five UMAP seeds. Noise remains label -1 and is never forced into a cluster.

Primary result: **18 clusters, 563/801 assigned (70.3%), 238 unclustered**.
Cosine silhouette on assigned records in original MPNet space is **0.04795**.
Reduced-space Euclidean silhouette is **0.44014**; it is not interchangeable
with the original-space score. UMAP changes geometry/density and can create
apparently separate groups, including on random inputs.

The fixed pipeline was run on 10 new samples of each embedding null family.
Random embeddings also form HDBSCAN clusters; their original-space silhouette
is near zero. Cluster existence alone is therefore not the evidence of structure.
The observed original-space score exceeds all 20 null scores, but this small
replicate count and changing assigned subsets limit inference.

| HDBSCAN comparison | ARI on shared assigned items | Shared assigned fraction | ARI including noise as one label |
|---|---:|---:|---:|
| Five UMAP seeds, pairwise mean | 0.857 | 63.1% | 0.567 |
| Twenty 80% subsamples vs full, mean | 0.742 | 56.4% of sampled IDs | 0.384 |
| MiniLM vs MPNet | 0.737 | 44.7% | 0.238 |

The shared-assigned score describes a repeatable core, not all records. Assignment
coverage changes. Noise-inclusive ARI is also imperfect because it treats all
rejected records as one cluster; both scores and assignment overlap are saved.
The cross-model MiniLM result has 14 clusters and assigns 422 investigations.

## Reading and c-TF-IDF

[Open random-member reading](reading/review.html). It contains five random members
from each of ten sampled clusters for Ward k50, Ward k336 and primary HDBSCAN.
The five-member sample conditions on clusters with >=5 members. All 50 Ward k50
and all 18 HDBSCAN clusters qualify, but only 22 of 336 fine Ward clusters qualify.
An additional unconditional cluster sample is saved in the JSON for each method.

c-TF-IDF descriptors are supplied for every non-noise cluster. Counts use English
stopword removal, 1–2 grams, min_df=2; cluster term frequencies are L1-normalized
and weighted by log(1 + integer mean class token count / corpus term count).
Words are descriptors and were not used to tune the clustering. They are not
human-assigned labels or independent causal evidence.

The 50 HDBSCAN examples were read. Several clusters are strongly topical (e.g.
language/translation and Chinese politics) while mixing distinct claimed reasons.
Other clusters share recurring references to conversation framing, lexical cues
and register. See [reading observations](reading/reading_notes.md).

## Artifacts and reproduction

- `evaluation_overview.png` / `.pdf` / `.svg`: combined evaluation figure.
- `null_stability/`: all null curves, resample memberships, ARIs, MiniLM vectors,
  figures, scripts' design metadata and independent validation.
- `hdbscan/`: all 90 parameter runs and labels, seed stability, full preset
  assignments/probabilities, noise, null runs, subsample/model checks.
- `reading/`: deterministic random-member review, all descriptors and notes.
- `validation.json`: HDBSCAN/reading checks; Ward checks in null_stability.

Run from workspace root with `clustering/.venv/bin/python`, in order:
`clustering/evaluate_cause_null_stability.py`,
`clustering/evaluate_cause_hdbscan.py`,
`clustering/evaluate_hdbscan_robustness.py`,
`clustering/build_cause_reading_review.py`,
`clustering/summarize_cause_evaluation.py`.
Dependency versions are in `clustering/requirements.lock.txt`. Prior datasets,
clusterings and sweep outputs are preserved; no cross-tab is produced here.

Method references:
- https://umap-learn.readthedocs.io/en/latest/clustering.html
- https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
- https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html
