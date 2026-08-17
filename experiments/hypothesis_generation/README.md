# Hypothesis generation vs. discrimination — replication

**Purpose.** Tests whether the published hypothesis-generation case studies in the NLA paper
([Fraser-Taliente et al. 2026](https://transformer-circuits.pub/2026/nla/index.html)) could have
been guessed from the behavior alone: give Opus a description of each behavior (blinded to the
cause) and ask for its top five hypotheses, then judge at what rank the NLA-discovered hypothesis
first appears. Hypothesizer models are chosen with training cutoffs preceding the NLA paper, to
rule out memorization of the paper itself.

**Paper artifact.** Table `tab:hypothesis_generation` in Appendix `app:hypothesis_discrimination`
of `paper/main.tex`. All numbers live in the paper and the auto-generated `results/RESULTS.md`
(regenerated on every run — don't hand-edit).

## Setup

Everything (behavior prompts, ground truths, generation, judge) lives in the single script
`run_hypothesis_generation.py`.

- **Behaviors** (4): the NLA paper's published case studies — language switching, misreported
  tool call (rigged arithmetic tool), planning in poetry, answer thrashing. Behavior descriptions
  are written from the observable behavior only, blinded to the cause and to the target model's
  identity ("a frontier LLM").
- **Hypothesizers**: `claude-opus-4-1` and `claude-opus-4-5` (both training cutoffs precede the
  NLA paper). *The paper's original informal test used Opus 4, which was retired from the API on
  2026-06-15; Opus 4.1 is the closest live stand-in.*
- **Prompt**: "what are your top five hypotheses for the cause?", N samples per (model, behavior),
  thinking enabled (budget 4096), temperature forced to 1.0.
- **Judge**: `claude-opus-4-8` (adaptive thinking), blinded to which model produced the
  hypotheses. It extracts the numbered hypotheses and reports the rank of the first one matching
  the ground-truth cause under explicit per-behavior match criteria (including explicit
  exclusions).
- **Metrics**: match@1 / match@3 / match@5 per (model, behavior).

## Reproduce

```bash
# from repo root, .env sourced; API-only (Anthropic), no GPU
.venv/bin/python experiments/hypothesis_generation/run_hypothesis_generation.py --num-samples 10
```

Outputs `results/results.jsonl` (full transcripts: hypothesizer prompt/response/thinking, judge
prompt/response, parsed verdict) and `results/RESULTS.md` (summary table). After editing any
match criteria, re-grade the stored generations without resampling via `--rejudge`
(`results/results_pre_rejudge_backup.jsonl` is the snapshot from before the last re-judge).

## Caveats

- **Language switching is graded on the root cause only** (malformed SFT data pairing
  English-translated prompts with untranslated foreign-language responses); the proximate
  mechanism the NLA surfaced (model fixated on the user being a non-native speaker) explicitly
  does NOT count.
- **Match strictness**: the judge applies the exclusions strictly (e.g. "phonetic bias toward the
  rhyme sound" does not count as forward-planning for poetry; generic sampling /
  multilingual-training-data stories do not count for language switching). A more lenient judge
  would raise all numbers.
- Poetry planning is framed as a mechanism question ("how does it accomplish this?") rather than
  a surprising-behavior question, matching how that case study is used in the NLA paper.
