# Results by analysis family

Start with the [browsable review catalogue](../reviews/index.html). Retained result files remain at their existing paths, so reports and downstream analyses keep their dependencies. New orientation READMEs explain each family; requested removals are recorded in [the removal log](../archive/removals.json).

| Folder | Scope |
|---|---|
| [final_explanations](final_explanations/README.md) | Historical MiniLM/Ward analysis of investigator claims across four source runs. Includes the retained average-linkage pilot. |
| [behaviour_facets](behaviour_facets/README.md) | Historical MPNet/Ward fits to verification-qualified PETRI and WildChat eval summaries, with the extended PETRI sweep retained. |
| [cause_facets](cause_facets/README.md) | Historical PETRI supported Core causes prose. These claimed reasons differ from the measured-edit inputs used by later checkpoints. Open the nested PETRI review. |
| [cause_evaluation](cause_evaluation/README.md) | Historical nulls, resampling, cross-model agreement, HDBSCAN parameter grid and random-member reading. |
| [behaviour_cosine_calibration](behaviour_cosine_calibration/README.md) | Six groups of ten summaries plus a random-group reference. This illustrates cosine scale; it is not a clustering method or universal threshold. |
| [checkpoint2](checkpoint2/README.md) | Separate PETRI/WildChat eval behaviour and measured-edit facets; Ward k=10..200 and preset UMAP/HDBSCAN, with shuffled-embedding nulls. |
| [checkpoint2_locations](checkpoint2_locations/README.md) | Re-embedded causes after stripping explicit unquoted edit-location phrases. Prior behaviour embeddings and assignments are reused. |
| [location_seed](location_seed/README.md) | Metadata cross-tabs and provisional PETRI scenario-prefix concentration, not another clustering algorithm. WildChat seed identities remain unavailable. |
| [petri_latest_qualified](petri_latest_qualified/README.md) | 269 investigations across 147 scenario families, with 1,052 eligible measured edits. Includes original clusters, within-cluster similarity reviews and cosine refinement. |
| [scenario_clustering](scenario_clustering/README.md) | All qualifying checkpoints within selected scenarios. Behaviour summaries, measured counterfactual edits and verifier assessment prose are separate analyses. |
