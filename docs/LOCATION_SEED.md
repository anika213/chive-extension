# Edit location and scenario concentration

Review [location cross-tabs, text changes, and per-cluster concentration](../results/location_seed/review.html)
and [revised cause clustering](../results/checkpoint2_locations/review.html).

Sources remain PETRI and WildChat eval only. Behaviour texts, vectors, and clusters
are reused unchanged. Causes are re-embedded with local CPU MPNet and the same
Ward k=10..200, UMAP10/HDBSCAN, and ten shuffled-embedding null protocol.
No generative LLM calls or semantic cluster labelling.

Changed stored messages are identified with exact-message sequence matching,
using unchanged messages as anchors. `changed_message_roles` preserves raw roles;
`edit_locations` identifies system prompt, user message, prior assistant turn,
and tool output. PETRI tool outputs are often user-role messages with explicit
`<tool_response>` wrappers. Mixed user/tool messages include both locations and
are flagged; this is message-level location, not exact within-message attribution.
Changed baseline/experiment indices are saved. Baseline and experiment serialized
hashes must match the prior verified context-diff table.

Only `cause_text` enters the embedder. The deterministic deletion rules strip
explicit unquoted role/turn location phrases and empty retention scaffolds.
Original summaries, pre-removal text, and deleted locator spans are saved. Quoted
payloads stay intact even if they mention roles, since they can be the intervention.
Thus this removes explicit location labels, not all meaning correlated with location.
Location-only remnants and new exact duplicates are excluded, but retained in parquet.
All existing measurement/size exclusions still apply; positive-only diversity
denominators and zero counts are preserved.

No explicit seed field exists in the downloaded investigation or prompt metadata.
PETRI concentration uses a **provisional scenario-prefix proxy** from the exact
`petri_<scenario>_e<episode>_t<turn>` ID shape, grouping episodes and turns together.
This is not independently verified seed provenance. WildChat statistics remain
null: its IDs and conversation hashes are not assumed to encode scenario seeds.
The prompt metadata lives in `data/seed_provenance/`, pinned to the original
dataset revision, with byte counts and SHA-256 in its manifest.

Per behaviour cluster, report dominant scenario, top-one and top-two fractions,
distinct scenario count, and missingness. HDBSCAN noise is separate. The reference
shuffles scenario prefixes 500 times among assigned rows, retaining cluster sizes
and scenario frequencies. Singletons are necessarily 100% one scenario. Cross-tabs
include exact edit-location combinations against both behaviour and cause clusters,
with experiment counts and distinct-investigation counts.

From the workspace root:

```bash
clustering/.venv/bin/python clustering/scripts/location_seed_analysis.py --scenario-prefix-proxy
clustering/.venv/bin/python clustering/scripts/run_location_clusters.py
clustering/.venv/bin/python clustering/scripts/report_location_seed.py
clustering/.venv/bin/python clustering/scripts/validate_location_seed.py
```

These preserve all prior checkpoint artifacts. Review outputs stop before semantic labels.
