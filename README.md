# Qwen3-8B clustering research

Start with **[the HTML review catalogue](reviews/index.html)** or [results by analysis family](results/README.md). The retained analyses include reviews, plots, assignments, embeddings, alternatives and intermediate research artifacts. Nothing has been pushed.

## Read first

- [Scenario-focused behaviour, counterfactual and verifier-prose analyses](results/scenario_clustering/README.md)
- [Latest qualifying PETRI investigations and cosine refinement](results/petri_latest_qualified/README.md)
- [One investigation, field by field](example.html)
- [Clustering method guides](docs/methods/README.md)

## Folder layout

| Folder | Contents |
|---|---|
| `reviews/` | Browsable catalogue of every saved research HTML page |
| `results/` | Analysis families, reviews and supporting machine-readable outputs |
| `docs/methods/` | Separate guides for Ward, average/complete linkage, UMAP/HDBSCAN and cosine refinement |
| `docs/` | Preprocessing, reproduction, location/ID notes and research history |
| `scripts/` | Analysis, review-building and validation scripts |
| `examples/` | Detailed investigation examples, data, JavaScript and builders |
| `data/` | Preprocessing checkpoints, provenance and locally retained raw downloads |
| `archive/` | Original run logs and pre-organization artifact inventory |
| `tests/` | Existing preprocessing and pipeline checks |

Result and data paths are preserved because later analyses reuse earlier records and vectors. Historical/intermediate status is documented in the result and data READMEs. Scripts now live in `scripts/`; see [reproduction](docs/REPRODUCING.md).

## Evidence and retention

Behaviour summaries, investigator claims, verifier assessments and measured counterfactual edits are separate inputs. Similarity does not establish a causal mechanism; cluster IDs are local to each run, facet and method. Scenario-prefix identity is a provisional proxy. Read each method's coverage, nulls and limitations alongside its review.

Git tracks research results and supporting intermediate artifacts. Raw release downloads under `data/petri/` and `data/wildchat/`, model weights, caches and environments remain on disk but are ignored. Download manifests and a full artifact inventory document the retained source files. See [data policy](data/README.md) and [archive policy](archive/README.md). Requested cuts are recorded in [the removal log](archive/removals.json); further cleanup is deferred.
