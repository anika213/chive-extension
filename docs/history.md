# Qwen3-8B investigation data

Source: https://huggingface.co/datasets/adamkarvonen/chive-data
Pinned revision: `0283d021bd373d361aadd46a9cf1f9520399d50a`.

[Cluster 0 counterfactual evidence review](../results/scenario_clustering/five_scenario_explanation_assessments/cluster_0_counterfactual_review/review.html)
audits the 16-member cluster from the five-scenario assessment run's `cosine_above_60`
preset. Its 223 measured tests include 63 changes of at least 50 percentage points;
26 edits from 11 investigations pass the existing filters. Includes every test,
actual message diffs, exclusion reasons, and a qualified comparison of possible
combined effects. No new embeddings or cluster assignments. Reproduce with
`clustering/.venv/bin/python clustering/scripts/analyze_assessment_cluster0.py`.

[Five-scenario explanation-assessment clustering](../results/scenario_clustering/five_scenario_explanation_assessments/review.html)
extends the three-scenario verifier-prose analysis with
`gangstalking_paranoia_sycophancy` and `manic_episode_sycophancy`: 34 assessments,
all qualifying checkpoints, verification >=7. Complete texts fit MPNet (64–353
tokens). The >0.70 option retains 23 assessments in nine groups, six spanning
multiple scenarios; alternatives and every member are in the review. Earlier
results are preserved. Reproduce with
`clustering/.venv/bin/python clustering/scripts/cluster_explanation_assessments.py --scenarios medical_authority_sycophancy sycophancy_false_endorsement doctor_poisoning_paranoia gangstalking_paranoia_sycophancy manic_episode_sycophancy --name five_scenario_explanation_assessments`.

[Verifier explanation-assessment clustering](../results/scenario_clustering/related_health_explanation_assessments/review.html)
uses the top-level `verification.jsonl → explanation_assessment` for the same 19
investigations across three related scenarios. This is a separate, explicitly
requested analysis of interpretive review prose, not measured causes. Complete
texts fit MPNet's limit (80–353 tokens); no truncation or chunking. Full-coverage
and cosine-first alternatives display every assessment. Reproduce with
`clustering/.venv/bin/python clustering/scripts/cluster_explanation_assessments.py`.

[Example: different parts of one investigation](../example.html) is an interactive
field guide to the medical-authority e2_t4 record: prompt/context, original and
cleaned summaries, classifier labels, all 13 counterfactual edits and measured
rates, all 30 baseline and 390 experimental responses, and the investigator
report. It compares possible clustering inputs without running any embeddings
or clustering. Rebuild with `clustering/.venv/bin/python clustering/scripts/build_parts_example.py`.

Counterfactual comparison: [cluster the measured edits from the same three related scenarios](../results/scenario_clustering/related_health_counterfactuals/review.html).
Contains 36 eligible experiments from 14/19 qualifying investigations, with all
turns and both runs retained. Every experiment page includes original/cleaned
text, response-label-checked rates, classifier question, and actual context diff.
One residual interpretation sentence was removed and one MPNet vector refreshed
locally; the other 35 vectors are reused. No generative LLM calls or semantic
labels. Reproduce with `clustering/.venv/bin/python clustering/scripts/cluster_related_counterfactuals.py`.

Related-scenario analysis: [three related health scenarios clustered together](../results/scenario_clustering/related_health_scenarios/review.html).
Combines all 19 qualifying investigations from `medical_authority_sycophancy`,
`sycophancy_false_endorsement`, and `doctor_poisoning_paranoia`, including every
qualifying turn from both runs. Includes separate group pages, scenario
composition, all pairwise similarities, and coverage/cosine-first alternatives.
Reproduce with `clustering/.venv/bin/python clustering/scripts/cluster_one_scenario.py --scenarios medical_authority_sycophancy sycophancy_false_endorsement doctor_poisoning_paranoia --name related_health_scenarios`.

Single-scenario analysis: [medical_authority_sycophancy across both runs and every available checkpoint](../results/scenario_clustering/medical_authority_sycophancy/review.html).
All eight released investigations qualify (e1/e2, t2–t5); t1 prompts have no released
investigations. Includes full-entry cluster pages, alternative cuts, a cosine
heatmap, and a Ward hierarchy. Groups may have two members because the scenario
has fewer than ten investigations. Reproduce with
`clustering/.venv/bin/python clustering/scripts/cluster_one_scenario.py`.

Latest behaviour review: [cosine-cohesive subgroups and selective reclustering](../results/petri_latest_qualified/cosine_refinement/review.html).
Uses the same 269 PETRI records and saved MPNet embeddings. Within the original
six HDBSCAN clusters, nine proper subgroups with mean cosine >0.4 retain 163/172
members; a separate minimum-10 view is available. A new search prioritizes
within-cluster cosine similarity, allowing rejection and requiring >=10 members
per cluster. It provides cosine-first and broader-coverage alternatives, each
with every member displayed. Silhouette is diagnostic only; existing assignments
remain unchanged. Reproduce with `clustering/.venv/bin/python clustering/scripts/refine_behaviour_clusters.py`
and validate with `clustering/.venv/bin/python clustering/scripts/validate_cosine_refinement.py`.

Latest PETRI clustering: [one latest qualifying checkpoint per run](../results/petri_latest_qualified/review.html),
with 269 investigations across 147 scenario families and 1,052 eligible cause experiments.
The [within-cluster similarity review](../results/petri_latest_qualified/similarity/review.html)
orders all six HDBSCAN behaviour clusters by cosine neighbourhood or average-similarity
rank, with closest neighbours, heatmaps, and exploratory subgroup cuts. All members
link to their cause rows. This diagnostic preserves the original assignments.
Rebuild it with `clustering/.venv/bin/python clustering/scripts/build_similarity_reviews.py`.
See [selection rules](../results/petri_latest_qualified/selection.md). Reproduce with
`clustering/.venv/bin/python clustering/scripts/recluster_latest_petri.py` from the workspace root.
Both facets are refit separately; WildChat retains its previous results. No semantic labels.

Previous revision: [edit locations and behaviour seed concentration](../results/location_seed/review.html).
Edit location is metadata; explicit unquoted location spans are removed from cause
embedding text. [Revised cause clustering](../results/checkpoint2_locations/review.html)
preserves the prior behaviour assignments. PETRI seed concentration uses a provisional
scenario-prefix proxy; WildChat seed identities are unavailable. No semantic labels.

Previous workflow: [observation-only preprocessing](PREPROCESSING.md), completed through
[Checkpoint 2 for review](../results/checkpoint2/review.html) for PETRI and WildChat eval.
Behaviour and cause facets are embedded and clustered separately within each source.
The [approved filtering amendments and 30 original cause summaries per source](../data/checkpoint1_approved/review.html)
are preserved. Train and thinking are deferred. Stopped before semantic labelling.
See [reproduction instructions](CHECKPOINT2.md); older clustering results below are historical.

`data/hf_files.txt` lists every file in the HF repository (149 files).
`data/hf_file_inventory.json` preserves the repository API inventory.
`data/download_manifest.json` records selected downloads, byte sizes, parsed JSONL
record counts, and SHA-256 checksums.

## Source separation

- `data/petri/qwen3_8b_petri_eval/`: PETRI alignment-audit investigations; the alignment-relevant set.
- `data/wildchat/qwen3_8b_wildchat_eval/`: standard non-thinking evaluation run.
- `data/wildchat/qwen3_8b_wildchat_thinking_eval/`: thinking-mode evaluation run.
- `data/wildchat/qwen3_8b_wildchat_train/`: training-scale investigation run.

WildChat release folders contain WildChat plus an agentic slice. Sources and runs
are preserved separately; cluster IDs are local to each source, facet, and method.

## Full and filtered artifacts

`investigations.jsonl` is the full published investigation file for each run;
`verification.jsonl` contains independent verification scores. There are no
separate filtered/unfiltered investigation files in the inventoried release.
The published files can already reflect upstream exclusions: “full” here means
all records released in that file, not an unreleased pre-filter original.

`binary_counterfactual_questions.jsonl` is the derived claim bank.
`eval_set_conc3_n2000.jsonl`, where available, is the publisher's filtered,
balanced **claim-level** evaluation set (mechanism concreteness >= 3, cap 2000).
It is not a second investigation dataset. The training run has no such eval file.
These overlapping artifacts must not be treated as independent investigations.

Only Qwen3-8B investigation-related data and its provenance are downloaded.
Screening files, completion files, and other models are excluded. Matching PETRI
and WildChat eval prompt metadata was subsequently fetched at the same pinned revision
under `data/seed_provenance/` to audit seed availability (separate checksum manifest).
The full experiment logs and reports are inside investigations.

For later cause analysis, use measured counterfactual outcomes as evidence;
free-text explanations and reports are not ground truth.

Run `../chive-env/bin/python scripts/download_qwen.py` from this directory to resume the
selected download and JSONL validation. The script uses system curl for HTTPS.

## Download validation

28 files downloaded (12,772,225,890 bytes), containing 25,826 investigations.
All JSONL records parse, all downloaded file sizes match publisher manifests,
and all available publisher row counts match. SHA-256 checksums are local
provenance checksums. See `data/validation_summary.json` for results.

| Source / run | Investigations | Verification records |
|---|---:|---:|
| PETRI eval (alignment-relevant) | 868 | 868 |
| WildChat eval | 1,736 | 1,735 |
| WildChat thinking eval | 1,442 | 1,441 |
| WildChat train | 21,780 | 21,780 |

All investigation and verification prompt IDs are unique within each run.
Two investigations have no published verification record: `prompt_003073` in
WildChat eval and `prompt_003369` in WildChat thinking eval. Both are retained;
do not assume they have verification scores. There are no orphan verification
records. Prompt IDs are scoped to their run, not globally unique identifiers.

## Final-explanation clustering

Completed separately for each run using local MiniLM sentence embeddings and
`sklearn.cluster.AgglomerativeClustering` with Ward linkage. See
[the review page](../results/final_explanations/review.html) and
[method/results notes](../results/final_explanations/README.md).
25,824 final explanations are clustered; two missing explanations are excluded.
Numeric cluster IDs only; no cluster names. This step clusters claimed
explanations, not verified causes. No behaviour clustering or cross-tab was
performed in this step, per the clarified scope.
