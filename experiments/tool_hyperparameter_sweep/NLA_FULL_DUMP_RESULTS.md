# NLA full-transcript-dump ceiling eval (2026-06-13)

> **Archived result — not reproducible from this release.** Run 2026-06-13 on the OLD pre-cf7
> n1000 claim bank; the raw result JSONLs and the standalone script (`run_nla_full_eval.py`)
> were not retained, so this write-up is the only record. Cited by `SUMMARY.md`
> ("Additionally tried") and the paper's `app:toolcalls`.

**Question.** The agent NLA arm only gets 5 reads of ≤32 token positions each, so it must
*guess* which tokens hold the signal. Was that **targeting** the bottleneck? If we removed it
entirely — dump **every** token's NLA verbalization into one Opus call and let it reason over
all of it — would NLA finally beat the transcript-only `plain` baseline?

This is a **ceiling** test, not a new tool. If full exposure doesn't help, no smarter
targeting (e.g. a search tool over the reads) would have helped either.

**Method.** A standalone script, `run_nla_full_eval.py` (not included in this release; it reused
`run_binary_eval.py`'s `BinaryClaim` / `print_report` / metrics and left the existing arms
untouched). For every claim whose transcript is **≤800 tokens** it builds a single Opus call:
transcript + question + claim + the verbalization of *every* token, in order — no agent loop,
no tool budget. Same answerer as `plain` (`claude-opus-4-8`, effort `high`, the `answer` tool).

- Why ≤800 tok: a 800-token transcript dumps ~92K tokens of reads (largest in slice ~119K) —
  comfortably in Opus context; longer transcripts blow up. The ≤800 cut keeps **398 / 1000**
  claims (194 unique transcripts). 2 claims skipped (same transcript, no tool call).
- **Prompt was deliberately simplified** (per Adam): kept the base `plain` instructions + the
  per-token NLA framing + **only the memorization-vs-context caveat**. The agent arm's longer
  caveat block (esp. *"representation strength ≠ counterfactual robustness"*) was **dropped**.
  This is a confound — see caveat 2.
- All three arms scored on the **identical 398-claim subset** (`plain` and the agent-NLA arm
  read back from their existing result files and restricted to these claim_ids — no rerun).

**Results** (≤800-tok slice, n=398; certified panel excludes 67 band claims → n=331):

| arm | acc@0.5 | AUROC | Brier | certified acc | certified AUROC |
|---|---|---|---|---|---|
| **plain** (transcript only) | 71.9% | **0.806** | 0.182 | 73.4% | **0.819** |
| **nla_full** (every token's read dumped) | 70.6% | 0.784 | 0.191 | 71.3% | 0.789 |
| **nla agent** (5 targeted reads) | 74.6% | 0.801 | 0.181 | 75.8% | 0.812 |

## Verdict

**Targeting was not the bottleneck.** Dumping *every* NLA read does **not** beat `plain` — it's
slightly *worse* (0.789 vs 0.819 certified AUROC, worse Brier). The targeted agent arm ≈ `plain`.
On this slice, the safe reading is **NLA reads add no measurable signal over Opus's own
transcript-level reasoning**, whether targeted or fully exposed; the full dump if anything
dilutes with noise. All gaps are within ~1 SE (n≈331–398, AUROC SE ≈0.02), so the honest call
is **"no benefit," not "NLA hurts."**

Implication: a **search tool** over the reads (the original idea that prompted this) would not
have helped here — there is no hidden signal being missed for lack of targeting.

## Caveats (read before trusting this)

1. **This is the slice where `plain` is strongest.** Short transcripts are easy for `plain` and
   give NLA the least headroom. It does **not** rule out NLA helping on **long** transcripts
   where `plain` degrades — but those are exactly the ones we can't fully dump (context). Real
   limitation of the ceiling design.
2. **Prompt-simplification confound.** With only the memorization caveat, `nla_full` is the
   weakest of the three on `supported_stays` (false-claims-dressed-as-"stays" — the class the
   *dropped* counterfactual-robustness caveat was written to protect): 60% vs `plain` 66% vs
   agent 64%. So part of full-dump's deficit may be the stripped prompt over-trusting "present"
   reads, not the dump itself. Headline (no benefit over plain) holds regardless. **To settle
   it:** re-run `nla_full` with the full caveat set and see if the gap to `plain` closes.
3. **No qualitative spot-check.** Full-dump transcripts were never read end-to-end to confirm
   Opus actually engages with the reads vs ignores them.
