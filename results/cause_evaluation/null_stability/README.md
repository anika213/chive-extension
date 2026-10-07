# PETRI claimed-cause facet: null baselines and stability

These diagnostics use the same 801 investigations, cause texts and saved normalized MPNet embeddings as `results/cause_facets/petri/qwen3_8b_petri_eval`. Cause text is the investigator's explicitly supported Core causes statements combined per investigation. Investigation verification >=7 does not independently verify each cause. No labels or semantic taxonomy are assigned.

## Null baselines

Observed Ward cosine silhouettes are **0.056636 at k=50** and **0.123968 at k=336**; k=336 is the observed maximum across every valid cut k=2..800.

1. **Random label assignment, 200 replicates.** Permute observed cluster labels among points, preserving the exact observed cluster-size distribution at each fixed k. Means are -0.099472 (k50) and -0.327180 (k336). No permutation exceeds observed; plus-one empirical upper-tail proportion is 1/201=0.004975. This is an easy random-assignment baseline and is not selection-aware. It is not a test that meaningful discrete categories exist.
2. **Isotropic random unit vectors, 20 replicates.** Generate 801x768 independent standard Gaussian coordinates and L2 normalize rows. Refit Ward and scan all k2..800 separately per replicate. Null optimized maxima have mean **0.033470**, range 0.032441..0.034294.
3. **Independent coordinate permutations, 20 replicates.** Independently permute each MPNet embedding column across investigations, then L2 normalize each row. This preserves each coordinate's empirical marginal before renormalization, but destroys cross-coordinate relationships. Refit Ward and repeat the full k2..800 sweep. Null optimized maxima have mean **0.026284**, range 0.024974..0.027929.

Both refitted null families are **selection-aware**: observed maximum is compared with the maximum across the same complete k search in each null, not only the null score at observed k336. Neither null maximum exceeds observed in 20 trials. The plus-one empirical upper-tail proportion is **1/21=0.047619**, the smallest possible with 20 replicates; it must not be reported as zero or as a more precise tiny p-value. Fixed-k comparisons are also saved separately. No multiple-test correction across the various exploratory comparisons is asserted.

The nulls establish a contrast with those specific artificial geometries. They do not preserve all structure of natural sentence embeddings, establish causality, prove a discrete taxonomy, or exclude a continuous semantic manifold. The isotropic null does not preserve embedding anisotropy/covariance; independent coordinate permutation does not preserve covariance either. Null-family choice matters. Row shuffling and reclustering was deliberately not used as a null, because it leaves pairwise geometry unchanged.

## Resampling and alternate representation

ARI (adjusted Rand index) is computed on shared **unique original investigation IDs**, not duplicated bootstrap copies or held-out ID placeholders. Duplicate copies are checked to share one cluster before mapping to the original ID. There are 20 resamples per condition and 190 pairwise comparisons, which are dependent and not 190 independent trials. Reported quantiles describe distributions, not confidence intervals.

| Perturbation | k50 mean ARI vs full / between resamples | k336 mean ARI vs full / between resamples |
|---|---|---|
| Bootstrap with replacement | 0.307 / 0.263 | 0.376 / 0.276 |
| 80% subsample, fixed k | 0.443 / 0.397 | 0.677 / 0.662 |
| 80% subsample, proportionally scaled k | 0.427 / 0.376 | 0.694 / 0.609 |

Bootstrap draws contain 801 records with replacement and an average 510.15 unique IDs (~63.7% coverage); pairs share an average 324.15 unique IDs. Holding k336 fixed therefore makes clustering relatively much finer on the reduced unique set. Duplicate sampling also changes observation weights. This partly changes the resolution being tested; it is not a pure seed test.

The 80% subsamples contain 641 unique records. Identical subsets are reused between fixed and scaled variants. Proportional scaling uses k40 for the k50 comparison and k269 for k336. ARI vs full compares these coarser subsample assignments with the original fixed-k partition restricted to those IDs; this deliberately changes the number of clusters. Neither fixed nor scaled variant perfectly solves changing coverage. Ward itself is deterministic for fixed inputs; repeating only a random seed would not be a useful stability test.

The alternate local model is pinned `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It embeds the exact same cause texts on CPU with L2 normalization. Token lengths are 18..131, within its 256-token limit, so no truncation/chunking. At matched k:

| k | MiniLM–MPNet ARI | MiniLM cosine silhouette |
|---|---:|---:|
| 50 | 0.219784 | 0.040756 |
| 336 | 0.317537 | 0.122011 |

Similar silhouette scores can coexist with substantially different membership. The two embedding models are alternative representations, not independent ground truth. Model sensitivity and bootstrap sensitivity argue against treating the selected assignment as an established cause taxonomy, even though observed scores exceed these null baselines. High-k clusters are small, and local-pair agreement and singleton effects matter.

## Files and reproduction

Run from repository root:

```bash
clustering/.venv/bin/python clustering/evaluate_cause_null_stability.py
clustering/.venv/bin/python clustering/plot_cause_null_stability.py
```

- `design.json`: seeds, exact IDs, source hashes, counts and observed scores.
- `label_permutation_replicates.json`, `label_permutation_summary.json`: all 200 label permutations.
- `embedding_null_replicates.json`, `embedding_null_curves.npz`, `embedding_null_summary.json`: every full null curve plus fixed-k and selection-aware summaries.
- `resample_replicates.json`, `resample_pairwise.json`, `resample_memberships.npz`, `resample_summary.json`: all membership draws/labels, per-replicate and pairwise scores. Membership matrices have one row per original ID, -1 for absent IDs, columns for reference k50 and k336.
- `minilm_embeddings.npy`, `minilm_labels.npy`, `minilm_token_lengths.json`, `alternate_model_summary.json`: alternate model outputs. IDs/order match the original cause JSONL; labels columns are k50/k336.
- `summary.json`, `runtime.json`, `validation.json`: combined results and checks.
- `null_sweep_comparison.png/pdf/svg`, `resampling_stability.png/pdf/svg`: diagnostic plots. Null shading is a pointwise empirical range, not a simultaneous confidence band.
