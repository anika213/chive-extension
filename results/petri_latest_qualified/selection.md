# PETRI: latest qualifying checkpoint per run

Apply existing verification >=7 and preprocessing first; choose largest t per (scenario,e). No fallback to earlier checkpoints based on cause availability.

- Behaviour rows: 801 → 269 (147 scenario families).
- Eligible cause experiments: 3030 → 1052.
- Investigations with eligible causes: 213; zero eligible causes: 56 (excluded from cause-diversity denominators).
- e1 and e2 remain separate. Selected t can be earlier than the final published checkpoint if later checkpoints did not qualify.
- Cause rows retain all existing artifact/size/location-text filters. Flagged rows remain in causes.parquet. Earlier discarded duplicate representatives are not reintroduced.
- Previous results preserved; WildChat unchanged. No semantic cluster labels.

[Selected investigations](../../data/petri_latest_qualified/petri/selected_investigations.json)
[Behaviour inputs](../../data/petri_latest_qualified/petri/behaviours.parquet)
[Cause inputs](../../data/petri_latest_qualified/petri/causes_for_clustering.parquet)
