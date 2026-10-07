# Cause-focused preprocessing, before embedding

`preprocessing.py` is a standard-library-only preprocessing stage. It makes no
API calls and computes no embeddings. By default it processes the 801 PETRI
Qwen3-8B eval investigations with verification >=7 and saves output under
`clustering/data/preprocessed/petri/qwen3_8b_petri_eval/`.

From the workspace root:

```bash
python3 clustering/scripts/preprocessing.py
```

To process both eval sources separately:

```bash
python3 clustering/scripts/preprocessing.py --sources petri wildchat
```

## Saved representations

- `claims.jsonl`: one investigator claim per row. Extracts only the numbered
  statement, omitting observed-behaviour sections, evidence prose and repeated
  final answers. Original text, normalized text, source spans, and supported /
  partially supported / refuted / uncertain statuses are retained. These are
  investigator claims; the script does not invent claim-to-experiment links.
- `experiments.jsonl`: one actual experiment per row, including review-needed
  items, measured outcomes, source IDs, edit metadata and investigator summaries
  explicitly marked unverified.
- `embedding_inputs.jsonl`: experiments with an unambiguous published intervention
  description and valid measured counts. `embedding_text` is the input to use
  when a later embedding step is requested:

  ```text
  Intervention: <published description, whitespace-cleaned and optionally de-identified>
  Observed outcome: classifier-YES frequency increased/decreased/unchanged.
  Classifier-YES means: <the original classifier question, optionally de-identified>
  ```

  Numerical counts and signed percentage-point changes are saved separately in
  `measured_outcome` and `measured_text`; they are not transcribed from explanatory
  prose. Keeping the exact classifier question avoids reversing predicates like
  “does the model NOT retry?”. Very small changes are still only descriptive;
  direction does not imply statistical significance or causal confirmation.
- `review_required.jsonl`: missing/ambiguous intervention descriptions, invalid
  measurements or missing claim structures. The script does not automatically
  treat an investigator's experiment narrative as a clean intervention.
- `excluded_investigations.jsonl`: records outside the verification filter.
- `manifest.json`: source checksums, counts, status counts, configuration and
  coverage. `preview.html` displays the first 30 ready items before/after.

The measured before/after counts are recomputed from baseline and experiment
`grader_verdicts`, checked against `grader_yes`. Disagreements prevent readiness.
Disagreements with claim-bank counts are logged, while the grader log remains
the source of truth. Supporting an intervention effect is not the same as
supporting a particular latent cause. Mixed/confounded edits stay visibly marked;
one intervention can change several factors.

## Entity normalization is explicit

The default is deliberately conservative: it removes report scaffolding and
normalizes whitespace but does NOT guess which names, paths, numbers, dates or
quoted cues are incidental. Automatic deletion could erase the manipulated
factor itself. It also does not use an LLM to paraphrase mechanisms.

Use a reviewed exact replacement map to remove incidental details while keeping
distinct people/roles separate. The example maps two household members to
`PERSON_A` and `PERSON_B`; it is NOT a complete map for this corpus and is not
applied by default:

```bash
python3 clustering/scripts/preprocessing.py \
  --replacement-map clustering/preprocessing_aliases.example.json \
  --output-dir clustering/data/preprocessed_with_aliases
```

Replacements are case-sensitive, whole-token, longest-match-first and simultaneous.
Every actual substitution is logged, with offsets in the original field. They
also apply inside quoted text: only approve an alias when the literal wording
is not the factor under test. Negation, instruction source, order cues and
increase/decrease direction are otherwise preserved. No unsupported abstraction
such as converting every edit to “prompt framing” is performed.

## Current PETRI run

- 801 investigations retained; 67 below the threshold excluded.
- 3,415 extracted claims: 1,790 supported, 195 partially supported, 1,429 refuted,
  and one uncertain/unrecognized status.
- 11,680 experiments retained in the audit output.
- 2,125 ready intervention descriptions, covering 617 investigations.
- 9,555 experiments require description review; no invented replacement text.
- No claim-bank count discrepancies in the ready records.

The published claim bank is a selected subset of experiments. The ready output
is therefore NOT a complete or random sample of all measured interventions and
should not be silently compared with the full 801-investigation clustering.
Report-claim rows and experiment rows are different units; do not combine them
as independent observations. No embedding or clustering is run by this script.

Checks: `python3 -m unittest discover -s clustering/tests -p 'test_preprocessing.py' -v`.
The tests cover alias boundaries and negation, status separation, count validation,
log-vs-prose conflicts, preserved classifier polarity, and missing-description
review handling.
