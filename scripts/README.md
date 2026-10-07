# Analysis scripts

Run from the repository root: `.venv/bin/python scripts/<name>.py`.
Scripts resolve inputs and outputs relative to the repository, so moving them here does not move research artifacts.

| Purpose | Entry points |
|---|---|
| Download and preprocessing | `download_qwen.py`, `preprocessing.py`, `prepare_checkpoint2.py` |
| Early explanation/behaviour/claimed-cause fits | `cluster_explanations.py`, `cluster_behaviours.py`, `cluster_petri_causes.py` |
| Nulls and robustness | `evaluate_cause_null_stability.py`, `evaluate_cause_hdbscan.py`, `evaluate_hdbscan_robustness.py` |
| Observation-only facets | `run_checkpoint2.py`, `finish_checkpoint2_report.py` |
| Edit location | `location_seed_analysis.py`, `run_location_clusters.py`, `report_location_seed.py` |
| Latest PETRI selection | `recluster_latest_petri.py`, `build_similarity_reviews.py`, `refine_behaviour_clusters.py` |
| Scenario subsets and text facets | `cluster_one_scenario.py`, `cluster_related_counterfactuals.py`, `cluster_explanation_assessments.py` |
| Evidence reviews | `analyze_assessment_cluster0.py`, `review_three_scenario_experiments.py`, `plot_counterfactual_patterns.py`, `plot_high_improvement_counts.py` |
| Validation | `validate_*.py`, plus `../tests/` |

See [reproduction](../docs/REPRODUCING.md) for prerequisites and saved-output caveats. Plot/build/report scripts generate presentation artifacts; they are not extra clustering methods.
