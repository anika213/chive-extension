# Reading observations

Inspected all 50 randomly selected members from 10 preset UMAP/HDBSCAN clusters
in `umap_hdbscan_preset.json`. These observations concern text coherence, not
whether the investigator's claimed mechanisms are true. No cluster names assigned.

The sample supports a mixture of causal phrasing and topical organization:

- Cluster 0's five examples concern language/translation, but their claimed
  reasons differ: localized-edit freezing, language-dependent safety, default
  phrase translation, terminology density, and continuation of prior style.
  Its c-TF-IDF words such as "Arabic" and "translation" capture subject matter.
- Cluster 1 groups Chinese politics/identity content. The five examples cite
  censorship thresholds, narrative date errors, language-dependent framing,
  balanced question framing, and model identity. Similar subject matter does
  not imply the same claimed cause.
- Cluster 16 has more recurring causal language about metaphors, lexical cues,
  and register, although several examples also share a contemplative-writing
  scenario. The repeated scenario makes this a limited test of generality.
- Cluster 9 contains diverse topics with recurring references to final-question
  framing and earlier conversational context. That is suggestive of a mechanism
  relationship, but the actual claims range from memory lookup to erroneous
  factual anchoring to roleplay suppression; they are not one uniform mechanism.
- Cluster 5 has prominent reward/XML terms, yet two of the five sampled members
  concern refusal of monitoring manipulation and restoration of prompt rules.
  This exposes mixed membership rather than a perfectly coherent cause group.
- Cluster 6 combines database-query edits/errors with benchmark-weighting
  examples. The vocabulary "table", "column" and "weight" can link different
  tasks, while some members also share minimal-edit/anchoring explanations.

The review contains verbatim cause facets and collapsible behaviour context, so
these judgments can be checked. c-TF-IDF terms were never used for fitting or
selecting clusters. They are descriptors, not evidence of a correct taxonomy.

Sampling details: the five-member audit chooses 10 clusters uniformly among
clusters with >=5 members. All 18 preset HDBSCAN clusters qualify. For Ward k336,
only 22 of 336 clusters qualify, so that audit cannot represent its many small
clusters. A second unconditional sample of 10 clusters, with up to five members
each, is provided in every partition JSON. The separate Ward k50 audit includes
five random members from 10 of its 50 clusters; all 50 qualify.

Several prompt IDs represent turns from the same PETRI audit scenario. Neither
this random-member inspection nor investigation-level bootstrap stability
establishes generalization to entirely new audit scenarios. The nulls test the
chosen mathematical baselines, not absence of all topical or template effects.
