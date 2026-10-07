# Data and preprocessing artifacts

The release is `adamkarvonen/chive-data` at revision `0283d021bd373d361aadd46a9cf1f9520399d50a`.
[download_manifest.json](download_manifest.json), [hf_file_inventory.json](hf_file_inventory.json)
and [validation_summary.json](validation_summary.json) preserve download provenance and validation.

| Folder | Role | Git policy |
|---|---|---|
| `petri/`, `wildchat/` | Published raw JSONL releases; about 12 GB | Retained locally, ignored |
| `seed_provenance/` | Pinned prompt metadata and its manifest | Tracked |
| `preprocessed/` | Legacy preprocessing and HTML preview | Tracked, historical |
| `checkpoint1/` | Initial observation-only preprocessing | Tracked, historical |
| `checkpoint1_v2/` | Revised exact context diffs and preprocessing review | Tracked, intermediate dependency |
| `checkpoint1_approved/` | Approved filtering and review samples | Tracked |
| `checkpoint1_locations/` | Location-stripped causes and scenario metadata | Tracked, active input |
| `petri_latest_qualified/` | Latest qualifying checkpoint selection | Tracked |

All intermediate files are retained pending a cleanup decision. Large files are intentional in this local snapshot, including the initial input-observation JSONL. Git has no remote and nothing has been pushed. A future publication can choose a smaller retention policy.

See [preprocessing](../docs/PREPROCESSING.md), [location handling](../docs/LOCATION_SEED.md), and [reproduction](../docs/REPRODUCING.md). Raw source files must be restored before recomputing pipelines that read them; the tracked HTML reviews are available without them.
