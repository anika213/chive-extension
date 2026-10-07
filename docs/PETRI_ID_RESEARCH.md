# PETRI investigation ID provenance — checked 2026-10-04

The repeated-audit-run interpretation is supported by primary sources and the
released prompt records. “Episode” was an informal interpretation, not verified
CHIVE terminology. PETRI calls repetitions of a scenario “epochs.”

## Primary sources

- CHIVE paper, Appendix B.2: https://arxiv.org/html/2608.16747v1#A2.SS2
  Reports 173 seed scenarios run twice per target, with Claude Sonnet 4.6 auditing.
  Transcripts are cut at assistant-response boundaries; each resulting prefix is
  an investigation prompt. Prefixes above 4,096 tokens are dropped.
- PETRI seed documentation: https://meridianlabs-ai.github.io/inspect_petri/components/seeds.html
  Seeds are auditor instructions, and the documentation explicitly lists
  `medical_authority_sycophancy` as a built-in seed ID.
- PETRI epochs documentation: https://meridianlabs-ai.github.io/inspect_petri/using/running.html#epochs
  Documents repeated scenario audits with `--epochs`.
- CHIVE dataset build provenance: https://github.com/adamkarvonen/chive/blob/main/experiments/dataset_build/README.md
  Describes 343 Qwen3-8B transcripts producing 2,061 prompt prefixes.

## Direct checks against the pinned Qwen3-8B prompt file

Input: `data/seed_provenance/petri_prompts.jsonl`, pinned dataset revision
`0283d021bd373d361aadd46a9cf1f9520399d50a` (checksum in that folder's manifest).

- 2,061 prompt records; 172 distinct scenario-name prefixes represented.
- 343 distinct `(scenario, e-number)` groups. Only `e1` and `e2` occur.
- All 2,061 suffix turn numbers equal the stored `n_turns`.
- All 2,061 suffix turn numbers equal the number of stored assistant messages plus one.
- Every earlier retained prefix exactly matches the beginning of the next retained
  prefix within its `(scenario, e-number)` group: 1,718 checks, zero failures.
- Turn number is not universally the count of stored user-role messages: 103
  records differ, because tool/user messages can occur consecutively.

For `medical_authority_sycophancy`, both e1 and e2 have t1 through t5 in the prompt
pool. The first e1 user introduces themselves as Dr. Rahul Bhatnagar; e2 begins
with Dr. Fischler. These are different generated conversations under the same
named scenario, with nested prefixes within each run.

The paper's configured 173 seeds should not be confused with the 172 scenario
names present in the released Qwen3-8B prefix pool. No reason for the missing
scenario/run is assigned without further provenance.

## Remaining uncertainty and correction

The published config names the original helper as
`investigations/petri_self_explanation/build_shards.py`, but that helper was not
found in the public CHIVE repository. Consequently the literal expansion of
the character `e` cannot be verified directly from its ID-construction code.
Interpreting it as the repeat/run index is strongly supported; “epoch” is PETRI's
documented repetition term. It is not the counterfactual experiment number.

Use: scenario seed → repeated audit run → target-response checkpoint → CHIVE
investigation → counterfactual experiments. The existing prefix-based concentration
calculations do not depend on whether the repeat is called an episode or epoch.
No clustering or metadata tables were changed by this research.
