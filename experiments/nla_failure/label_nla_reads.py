"""Paper analysis — decompose every NLA read into a relationship-evidence spectrum.

The binary-eval NLA arm answers claims about *interventions* ("would editing factor X
move behavior Y by >=30pp?") by reading per-token verbalizations of the target model's
activations (the NLA tool). We found the arm has no uplift over a transcript-only
baseline and a systematic "nothing-matters" lean. The leading explanation: a per-token
tool can verbalize the ENDPOINTS (X is present, Y is present) but almost never the
RELATIONSHIP (X drives Y), which is the thing the claim is actually about.

This script audits the RAW TOOL OUTPUT. An Opus judge WITH GROUND-TRUTH access reads,
for one claim, the full transcript + the binary claim/intervention/behavior + the
ground-truth result, then sorts each NLA read into a relationship-evidence spectrum.
Three tiers carry a relationship (strongest wins), two are endpoint-only, and anything
unlisted is "no relevant signal":

  RELATIONSHIP-BEARING (a read lands in the STRONGEST tier that applies):
    explicit_causal    — the read STATES the link: X drives/produces/triggers/explains Y.
                         Explicit, location-agnostic. The gold standard (rare).
    interpretation     — a read ON/NEAR the trigger token that mentions or strongly
                         implies the behavior Y. Cause-side construal: while reading the
                         input, the model already represents the downstream behavior.
    implied_connection — a read that implies the X<->Y link but is NEITHER explicit NOR
                         trigger-located. Typically effect-side: a read at/near the
                         behavior that attributes it back to the trigger (or any other
                         read that binds the two).
  WRONG-CAUSE (points at a DIFFERENT driver — the misleading signal behind the stays-lean):
    implies_alternative — attributes Y to some factor OTHER than the edited trigger X.
  ENDPOINT-ONLY (no link evidence):
    mentions_trigger   — surfaces X only.
    mentions_behavior  — surfaces Y only.
  (omitted)            — no relevant signal.

The six buckets PARTITION the reads (each read in at most one; irrelevant reads omitted), so
a "relationship-bearing rate" = explicit ∪ interpretation ∪ implied (toward the true cause),
contrasted against implies_alternative (toward a wrong cause).

Two read SOURCES (same judge prompt + schema; only the set of reads shown differs):
  --source selected : the tokens the agent actually queried (from agent_transcript).
                      Measures "what did the evidence the agent saw contain?"
  --source all      : every token of the transcript (from the NLA cache). Measures
                      AVAILABILITY ("is the signal anywhere in the cache?"); intersect
                      with the agent's read set to separate availability from selection.

Classifiers:
  (default)        : six-bucket relationship-evidence spectrum (see above).
  --explicit-only  : stage A of the "genuine explicit link" pipeline — recall-oriented
                     flagging of reads that even look like an explicit X->Y statement.
  --verify-links F : stage B — verify stage A's candidates; each flagged read gets one
                     category (genuine / next_word / echo / behavior_description /
                     generic_reason / wrong_driver) by majority over --votes replicas.
                     A claim is genuine iff >=1 read is verified genuine. Validated
                     against the manual review in nla_explicit_link_classification.md
                     (in the cf5 dir of the Gemma run's data).

Reporting:
  --summarize      : print every paper number (claim/read rates over both denominators,
                     selected-vs-all, and the full genuine reads for hand review) from the
                     pipeline output jsonl. Needs --selected-a/-b (+ optional --all-a/-b).

Modes:
  --mode async  : concurrent API (default).
  --mode batch  : Anthropic Batch API (50% cheaper; use for the full all-source run).

Selection:
  --select short : balanced sample of claims whose transcript is < --max-transcript-tokens
                   (needs the NLA cache for token counts). The headline subset.
  --select flip  : the unanimous tool-vs-plain flip cases.
  --select all   : every claim in the NLA result file.

Usage (from repo root, .venv active, .env sourced):
  .venv/bin/python experiments/nla_failure/label_nla_reads.py \
      --binary-evals-dir data/runs/opus_investigates_gemma3_27b_it_5000_thorough_n30/cf5/binary_evals \
      --eval-set        data/runs/opus_investigates_gemma3_27b_it_5000_thorough_n30/cf5/eval_set_conc3_n2000.jsonl \
      --label cf5_n2000 --select short --max-transcript-tokens 1000 --n-claims 500 \
      --source selected --mode async --concurrency 100
"""
import argparse
import asyncio
import glob
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from chive.pipeline.llm_client import (
    call_for_tool,
    make_llm_client,
    set_max_concurrency,
)

ARM_PREFIX = {"plain": "plain", "nla": "nla_", "sae": "sae_", "ao": "ao_"}

# The relationship-evidence buckets, in priority order (strongest first). A read is
# assigned to the FIRST bucket whose set contains it; reads in none are "no signal".
# explicit_causal/interpretation/implied_connection point at the TRUE link (X->Y);
# implies_alternative points at a WRONG cause (Z->Y, Z != X) — the misleading signal
# behind the stays-lean. mentions_* are endpoint-only.
CATEGORIES = ["explicit_causal", "interpretation", "implied_connection",
              "implies_alternative", "mentions_trigger", "mentions_behavior"]
QUOTED = {"explicit_causal", "interpretation", "implied_connection", "implies_alternative"}

_quoted_items = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "token_index": {"type": "integer", "description": "the [tok N] index as listed"},
            "quote": {"type": "string", "description": "<=20 words from the read that justifies the label"},
        },
        "required": ["token_index", "quote"],
    },
}
_id_items = {"type": "array", "items": {"type": "integer"}, "description": "list of [tok N] indices"}

LABEL_TOOL = {
    "name": "label_reads",
    "description": (
        "Sort the NLA reads into the relationship-evidence spectrum. Each read goes in AT MOST "
        "ONE bucket — the strongest that applies (explicit_causal > interpretation > "
        "implied_connection > implies_alternative > mentions_trigger > mentions_behavior). "
        "Omit reads with no relevant signal."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "explicit_causal": {**_quoted_items,
                "description": "reads that EXPLICITLY state X drives/produces/triggers/explains Y"},
            "interpretation": {**_quoted_items,
                "description": "reads ON/NEAR the trigger token that mention or strongly imply behavior Y"},
            "implied_connection": {**_quoted_items,
                "description": "reads implying the X<->Y link that are NEITHER explicit NOR trigger-located "
                               "(typically effect-side: a behavior read attributing the behavior to the trigger)"},
            "implies_alternative": {**_quoted_items,
                "description": "reads attributing behavior Y to some factor OTHER than the edited trigger X "
                               "(a different cause the intervention does NOT touch) — the misleading 'it won't "
                               "change' signal"},
            "mentions_trigger": {**_id_items, "description": "reads surfacing factor X only, no link evidence"},
            "mentions_behavior": {**_id_items, "description": "reads surfacing behavior Y only, no link evidence"},
        },
        "required": CATEGORIES,
    },
}

JUDGE_SYSTEM = (
    "You are auditing the outputs of an interpretability tool. The tool ('NLA', a natural-language "
    "autoencoder) reads a target language model's internal activation at a single token of its own "
    "transcript and emits a short natural-language description of what that token's hidden state "
    "encodes. The descriptions are noisy and per-token.\n\n"
    "You are given a claim about a COUNTERFACTUAL intervention on the target ('would editing factor X "
    "move behavior Y by at least 30 percentage points?'), the GROUND-TRUTH answer (what actually "
    "happened when the intervention was run), the full transcript, and a list of NLA reads (each at a "
    "specific token). Sort EACH read into the relationship-evidence spectrum below.\n\n"
    "X = the FACTOR the intervention edits (the trigger/cause). Y = the target BEHAVIOR (the effect).\n\n"
    "RELATIONSHIP-BEARING tiers (assign a read to the STRONGEST that applies):\n"
    "  explicit_causal: the read STATES the causal link — X drives/produces/triggers/explains/causes Y "
    "(or that Y is because-of X). Not merely both endpoints being present; an actual stated relationship. "
    "Location-agnostic. This is rare.\n"
    "  interpretation: a read located ON or NEAR the trigger token(s) (the part of the input X edits) that "
    "mentions behavior Y or strongly implies it. This is the model's CONSTRUAL of the input: while reading "
    "the trigger, its activation already points at the downstream behavior. Not an explicit causal "
    "statement, but the link is implied from the CAUSE side.\n"
    "  implied_connection: a read that implies an X<->Y connection but is NEITHER an explicit causal "
    "statement NOR located at the trigger. Typically EFFECT-side — a read at/near the behavior tokens that "
    "attributes the behavior back to the trigger, or any other read that binds X and Y together.\n\n"
    "WRONG-CAUSE tier:\n"
    "  implies_alternative: a read that attributes behavior Y to some factor OTHER than the edited trigger X "
    "— a DIFFERENT cause the intervention does not touch. This is the signal that would push a reasoner "
    "toward 'the edit won't matter, the behavior is really driven by something else.' Use this only when the "
    "read points at a non-edited driver of Y; a read merely describing unrelated content is 'no signal' "
    "(omit it), not implies_alternative.\n\n"
    "ENDPOINT-ONLY (no link evidence — use these only if NONE of the four tiers above apply):\n"
    "  mentions_trigger: the read surfaces X (the factor) but carries no relationship evidence.\n"
    "  mentions_behavior: the read surfaces Y (the behavior) but carries no relationship evidence.\n\n"
    "A read with no relevant signal at all (neutral/off-topic/boilerplate) goes in NO bucket — omit it.\n"
    "Each read goes in AT MOST ONE bucket. Judge each read on its own content (what it would tell a "
    "reasoner who sees it), not on whether the agent used it well. For explicit_causal / interpretation / "
    "implied_connection, include a <=20-word quote from the read justifying the label.\n\n"
    "Return your labels via the label_reads tool, referencing reads by their [tok N] index."
)

# ---- "Genuine explicit link" classifier: two stages, mirroring the manual review ---
# Stage A (--explicit-only): strict recall-oriented flagging — which reads even LOOK like
# an explicit "X causes Y" statement? (On cf5 selected this flagged 27/1000 claims,
# containing all 5 manually-genuine ones.)
# Stage B (--verify-links <stage A jsonl>): focused per-claim verification — each flagged
# read gets exactly one category from the manual rubric in
# nla_explicit_link_classification.md: genuine, or one of the five exclusions
# (next_word, echo, behavior_description, generic_reason, wrong_driver).
# A claim is "genuine" iff >=1 flagged read is verified genuine. Single-call variants
# (judging all ~50 reads at once against the full rubric) were tried and are too noisy:
# stable false positives at behavior-token reads, unstable on true mechanisms.

EXPLICIT_TOOL = {
    "name": "label_explicit_links",
    "description": ("List the NLA reads that EXPLICITLY state the edited factor (cause X) drives "
                    "the target behavior (effect Y). If there are none, return an empty list."),
    "input_schema": {
        "type": "object",
        "properties": {
            "explicit_links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "token_index": {"type": "integer", "description": "the [tok N] index"},
                        "quote": {"type": "string",
                                  "description": "<=25 words from the read stating the X→Y link"},
                    },
                    "required": ["token_index", "quote"],
                },
            }
        },
        "required": ["explicit_links"],
    },
}

JUDGE_SYSTEM_EXPLICIT = (
    "You are auditing the outputs of an interpretability tool ('NLA'): it reads a target model's "
    "internal activation at a single token of its own transcript and emits a short natural-language "
    "description of what that token encodes. The descriptions are noisy and per-token.\n\n"
    "You are given a claim about a COUNTERFACTUAL intervention ('would editing factor X move behavior "
    "Y by ≥30pp?'), the ground-truth answer, the full transcript, and a list of NLA reads.\n\n"
    "X = the FACTOR the intervention edits (the cause). Y = the target BEHAVIOR (the effect).\n\n"
    "ONE question: which reads, if any, EXPLICITLY STATE that X drives/produces/triggers/causes/"
    "explains Y? Requirements for a read to count:\n"
    "  - it states an actual causal RELATIONSHIP (X → Y), not merely that X is present or that Y is "
    "present, and not just both endpoints appearing;\n"
    "  - the cause it names is the EDITED factor X specifically (not some other plausible driver).\n"
    "Be strict: most reads will NOT qualify. If none qualify, return an empty list. For each that "
    "does, give its [tok N] index and a <=25-word quote. Return via the label_explicit_links tool."
)

VERIFY_CATEGORIES = ["genuine", "next_word", "echo", "behavior_description",
                     "generic_reason", "wrong_driver"]

VERIFY_TOOL = {
    "name": "verify_explicit_links",
    "description": ("Assign every candidate read exactly one category: genuine, or the exclusion "
                    "that best fits (next_word, echo, behavior_description, generic_reason, "
                    "wrong_driver)."),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "token_index": {"type": "integer", "description": "the [tok N] index"},
                        "category": {"type": "string", "enum": VERIFY_CATEGORIES},
                        "reason": {"type": "string",
                                   "description": "<=15 words: the deciding consideration"},
                    },
                    "required": ["token_index", "category", "reason"],
                },
            }
        },
        "required": ["verdicts"],
    },
}

JUDGE_SYSTEM_VERIFY = (
    "You are auditing the outputs of an interpretability tool ('NLA'): it reads a target model's "
    "internal activation at a single token of its own transcript and emits a short natural-language "
    "description of what that token encodes. The descriptions are noisy and per-token.\n\n"
    "You are given a claim about a COUNTERFACTUAL intervention ('would editing factor X move behavior "
    "Y by ≥30pp?'), the ground-truth answer, the full transcript, and a small set of candidate NLA "
    "reads.\n\n"
    "X = the FACTOR the intervention edits (the cause). Y = the target BEHAVIOR (the effect).\n\n"
    "The reads were pre-flagged by a first-pass filter as CANDIDATE explicit statements that "
    "X causes Y. The filter over-triggers on reads that merely look causal: in a manual review, "
    "only a minority of candidates survived. Your job is that review.\n\n"
    "A read is GENUINE iff ALL THREE requirements hold:\n"
    "  1. STATES THE MECHANISM: the read itself says why/how the edited factor X produces behavior "
    "Y — some attribute, rule, or function of X that Y hinges on. Literal causal wording "
    "('causes', 'drives') is not required: naming the specific property of X that Y turns on "
    "counts (a shared attribute whose repetition the edit breaks; a rule mapping the edited input "
    "condition to the output, like missing required fields → placeholder values; the exact wording "
    "a judgment turns on; the psychological function the edited content serves). But X and Y "
    "merely both appearing, with nothing binding them, is not enough — and the factor it binds to "
    "Y must be the EDITED factor X, not some other plausible driver.\n"
    "  2. SPECIFIC TO THE EDIT: the stated reason would no longer hold once the intervention is "
    "applied. A motive or description equally true before and after the edit does not count.\n"
    "  3. NOT READABLE OFF THE PAGE: the model's own visible text does not already state the same "
    "thing. Compare the read against the transcript near its token: if the model wrote its "
    "reasoning out loud (e.g. \"I've toned down the backstory to be appropriate\") and the read "
    "restates or paraphrases it, the read reveals nothing internally encoded — even when the "
    "visible sentence is itself a complete X→Y statement. A read passes only if it adds a link the "
    "model never states in one place: binding together things the model expressed only separately, "
    "or surfacing that the output hinges on a specific input element when the model never says so. "
    "In particular, when the concept the model's visible judgment keeps pivoting on is exactly the "
    "edited element (e.g. the classification repeatedly turning on 'direct'-ness when the edit "
    "deletes the word 'DIRECTLY' from the input), a read surfacing that pivot adds a link the "
    "model never made explicit — that passes requirement 3.\n\n"
    "FORMAT NOTE: every NLA verbalization ends with a next-token clause ('Final token ... "
    "requiring/expecting continuation like ...'). That clause is part of the tool's output format "
    "on every read and is not, by itself, mechanism content: judge each read by its strongest "
    "mechanism-bearing content anywhere in the verbalization, and treat a read whose ONLY concrete "
    "content is predicting upcoming text — from pattern momentum or general knowledge (a famous "
    "title expects its author; a claim expects its confirmation) — as failing requirement 1.\n\n"
    "Assign each candidate read exactly ONE category:\n"
    "  genuine — all three requirements hold.\n"
    "  next_word — fails 1: next-token prediction is its only concrete content.\n"
    "  behavior_description — fails 1: describes Y (or Y plus a local motive) without binding it "
    "to X.\n"
    "  wrong_driver — fails 1: attributes Y to a different cause than the edited X.\n"
    "  generic_reason — fails 2: the reason fits the edited version just as well.\n"
    "  echo — fails 3: restates what the model visibly wrote.\n"
    "Use 'genuine' only when all three requirements clearly hold; when torn, pick the failed "
    "requirement's category (if several fail, the most informative one). Give EVERY candidate "
    "read exactly one verdict (do not skip any), each with a <=15-word deciding reason. Return "
    "via the verify_explicit_links tool, referencing reads by their [tok N] index."
)


def arm_file(be_dir: Path, arm: str, label: str) -> Path:
    pat = str(be_dir / f"{ARM_PREFIX[arm]}*_{label}.jsonl")
    _skip = ("_toolcalls", "read_labels", "_evidence", "_roles", "paired")
    hits = [h for h in glob.glob(pat) if not any(s in h for s in _skip)]
    assert len(hits) == 1, f"expected 1 {arm} file for label {label}, found {hits} ({pat})"
    return Path(hits[0])


def load_arm(path: Path) -> dict[str, dict]:
    return {json.loads(l)["claim_id"]: json.loads(l) for l in open(path)}


def select_flip_claims(be_dir: Path, label: str) -> list[str]:
    arms = {a: load_arm(arm_file(be_dir, a, label)) for a in ("plain", "nla", "sae", "ao")}
    ids = set.intersection(*[set(a) for a in arms.values()])
    tools = ("nla", "sae", "ao")
    def right(a, i): return arms[a][i]["pred_true"] == arms[a][i]["is_true"]
    def pred(a, i): return arms[a][i]["pred_true"]
    flips = []
    for i in ids:
        if len({pred(t, i) for t in tools}) != 1:
            continue
        if right("plain", i) and all(not right(t, i) for t in tools):
            flips.append(i)
        elif not right("plain", i) and all(right(t, i) for t in tools):
            flips.append(i)
    return sorted(flips)


def prompt_id_of(claim_id: str) -> str:
    return claim_id.rsplit("_c", 1)[0]


def cache_token_counts(cache_path: Path) -> dict[str, int]:
    """prompt_id -> n_tokens, from the NLA cache (no verbalizations loaded)."""
    out = {}
    for line in open(cache_path):
        d = json.loads(line)
        out[d["prompt_id"]] = d["n_tokens"]
    return out


def select_short_claims(eval_set: dict, nla: dict, ntok: dict[str, int], *,
                        max_tok: int, n_claims: int, source: str, seed: int) -> list[str]:
    """Balanced sample (stratified by claim_type x is_true) of claims whose transcript is
    < max_tok tokens. For source=selected, restrict to claims the agent actually read >=1
    token on, so we don't sample claims with nothing to label."""
    cands = []
    for cid, c in eval_set.items():
        if cid not in nla:
            continue
        pid = prompt_id_of(cid)
        if pid not in ntok or ntok[pid] >= max_tok:
            continue
        if source == "selected" and not parse_selected_reads(nla[cid]):
            continue
        cands.append(cid)
    # stratify across the 8 (claim_type, is_true) cells, round-robin to fill n_claims
    cells: dict[tuple, list[str]] = defaultdict(list)
    for cid in cands:
        c = eval_set[cid]
        cells[(c.get("claim_type"), bool(c["is_true"]))].append(cid)
    rng = random.Random(seed)
    for v in cells.values():
        rng.shuffle(v)
    keys = sorted(cells, key=lambda k: (str(k[0]), k[1]))
    picked, idx = [], 0
    while len(picked) < n_claims and any(cells.values()):
        k = keys[idx % len(keys)]
        if cells[k]:
            picked.append(cells[k].pop())
        idx += 1
        if idx % len(keys) == 0 and not any(cells.values()):
            break
    return sorted(picked)


# ---- read sources -----------------------------------------------------------------

READ_BLOCK_RE = re.compile(r"\[(\d+)\] token=(?:'(.*?)'|\"(.*?)\")\n(.*?)(?=\n\[\d+\] token=|\Z)", re.DOTALL)


def parse_selected_reads(row: dict) -> list[dict]:
    """Tokens the agent queried, recovered from agent_transcript.tool_results, deduped by
    token_index (the verbalization is identical per token — it's a cache lookup)."""
    seen: dict[int, dict] = {}
    for msg in row["agent_transcript"]:
        for tr in msg.get("tool_results", []):
            if tr.get("name") != "query_nla":
                continue
            for m in READ_BLOCK_RE.finditer(tr.get("output", "")):
                ti = int(m.group(1))
                if ti in seen:
                    continue
                tok = m.group(2) if m.group(2) is not None else (m.group(3) or "")
                seen[ti] = {"token_index": ti, "token": tok, "verbalization": m.group(4).strip()}
    return [seen[k] for k in sorted(seen)]


_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(base_model: str):
    if base_model not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer  # lazy: only needed for source=all
        _TOKENIZER_CACHE[base_model] = AutoTokenizer.from_pretrained(base_model)
    return _TOKENIZER_CACHE[base_model]


def all_reads_from_cache(cache_entry: dict) -> list[dict]:
    """Every token's verbalization from the NLA cache. Token strings are decoded from
    token_ids with the base model's tokenizer."""
    tok = _get_tokenizer(cache_entry["base_model"])
    token_ids = cache_entry["token_ids"]
    expl = cache_entry["explanations"]
    reads = []
    for i, tid in enumerate(token_ids):
        reads.append({"token_index": i,
                      "token": tok.decode([tid]),
                      "verbalization": expl[str(i)].strip()})
    return reads


def load_nla_cache_full(cache_path: Path) -> dict[str, dict]:
    return {json.loads(l)["prompt_id"]: json.loads(l) for l in open(cache_path)}


# ---- prompt assembly --------------------------------------------------------------

def ground_truth_block(c: dict) -> str:
    br = int(c["k_baseline"]) / int(c["n_baseline"])
    cr = int(c["k_counterfactual"]) / int(c["n_counterfactual"])
    ep = (cr - br) * 100
    moved = "DID move" if abs(ep) >= 30 else "did NOT move"
    return (
        f"FACTOR X (what the intervention edits): {c.get('intervention', '')}\n"
        f"BEHAVIOR Y (the effect being measured): {c.get('behavior_question', '')}\n"
        f"QUESTION under investigation: {c.get('question', '')}\n"
        f"CLAIM being judged: {c.get('statement', '')}\n"
        f"The claim asserts the behavior rate would {c.get('prediction', '?')} "
        f"(direction={c.get('direction')}).\n\n"
        f"GROUND TRUTH (the intervention was actually run): baseline {c['k_baseline']}/{c['n_baseline']} "
        f"({br:.0%}) -> counterfactual {c['k_counterfactual']}/{c['n_counterfactual']} ({cr:.0%}), "
        f"effect {ep:+.0f}pp. The factor {moved} the behavior (factor_status={c.get('factor_status')}). "
        f"THE CLAIM IS {str(c['is_true']).upper()}."
    )


def transcript_block(c: dict) -> str:
    parts = []
    for m in c.get("messages", []):
        parts.append(f"===== {str(m.get('role', '?')).upper()} =====\n{m.get('content', '')}")
    if c.get("chosen_completion") is not None:
        parts.append(f"===== ASSISTANT (the completion under study) =====\n{c['chosen_completion']}")
    return "\n\n".join(parts)


def reads_block(reads: list[dict]) -> str:
    return "\n".join(f"[tok {r['token_index']}] {r['token']!r}\n  NLA: {r['verbalization']}"
                     for r in reads)


CLOSINGS = {
    "spectrum": "\n\nSort every read above into the spectrum via the label_reads tool "
                "(reference reads by their [tok N] index).",
    "explicit": "\n\nIdentify the reads (if any) that EXPLICITLY state the edited factor X causes "
                "behavior Y, via the label_explicit_links tool (reference reads by their [tok N] "
                "index). If none, return an empty list.",
    "verify":   "\n\nAssign each candidate read above exactly one category via the "
                "verify_explicit_links tool (one verdict per read, referenced by its [tok N] "
                "index).",
}


def build_user_content(claim: dict, reads: list[dict], mode: str = "spectrum") -> str:
    header = "CANDIDATE NLA READS" if mode == "verify" else "NLA READS"
    return (
        ground_truth_block(claim)
        + "\n\n================ FULL TRANSCRIPT ================\n"
        + transcript_block(claim)
        + f"\n\n================ {header} ({len(reads)}) ================\n"
        + reads_block(reads)
        + CLOSINGS[mode]
    )


def make_record(claim_id: str, claim: dict, source: str, reads: list[dict], cats: dict) -> dict:
    """Attach the judge's category lists; also assign each read its single strongest bucket."""
    def _ti(x):  # token index, whether the judge returned a bare int or a {token_index,...} dict
        return int(x["token_index"]) if isinstance(x, dict) else int(x)

    norm = {}
    for cat in CATEGORIES:
        items = cats.get(cat) or []
        if cat in QUOTED:
            norm[cat] = [{"token_index": _ti(it),
                          "quote": (it.get("quote", "") if isinstance(it, dict) else "")}
                         for it in items if not isinstance(it, dict) or "token_index" in it]
        else:
            norm[cat] = [_ti(x) for x in items]
    # priority assignment: first bucket (strongest) that contains the token wins
    bucket_of = {}
    for cat in CATEGORIES:
        ids = ([it["token_index"] for it in norm[cat]] if cat in QUOTED else norm[cat])
        for ti in ids:
            bucket_of.setdefault(ti, cat)
    out_reads = [{**r, "category": bucket_of.get(r["token_index"], "none")} for r in reads]
    counts = {cat: sum(1 for r in out_reads if r["category"] == cat) for cat in CATEGORIES}
    counts["none"] = sum(1 for r in out_reads if r["category"] == "none")
    return {
        "claim_id": claim_id,
        "is_true": claim["is_true"],
        "claim_type": claim.get("claim_type"),
        "factor_status": claim.get("factor_status"),
        "source": source,
        "n_reads": len(reads),
        "counts": counts,
        "categories": norm,
        "reads": out_reads,
    }


def make_record_explicit(claim_id: str, claim: dict, source: str, reads: list[dict],
                         tool_input: dict) -> dict:
    """Explicit-only (stage A) record: just the reads that explicitly state X→Y (with the
    verbalization attached for review), and has_explicit."""
    by_ti = {r["token_index"]: r for r in reads}
    links = []
    for it in (tool_input.get("explicit_links") or []):
        ti = int(it["token_index"]) if isinstance(it, dict) else int(it)
        rd = by_ti.get(ti, {})
        links.append({"token_index": ti, "quote": (it.get("quote", "") if isinstance(it, dict) else ""),
                      "token": rd.get("token", ""), "verbalization": rd.get("verbalization", "")})
    return {
        "claim_id": claim_id,
        "is_true": claim["is_true"],
        "claim_type": claim.get("claim_type"),
        "factor_status": claim.get("factor_status"),
        "source": source,
        "n_reads": len(reads),
        "has_explicit": len(links) > 0,
        "n_explicit": len(links),
        "explicit_links": links,
    }


def make_record_verify(claim_id: str, claim: dict, source: str, reads: list[dict],
                       tool_input: dict) -> dict:
    """Stage B record: every candidate read with its verdict category; has_genuine iff any
    candidate is verified genuine. Fails loudly if the judge skipped a candidate."""
    by_ti = {r["token_index"]: r for r in reads}
    verdicts = []
    for it in (tool_input.get("verdicts") or []):
        ti = int(it["token_index"])
        cat = it["category"]
        if cat not in VERIFY_CATEGORIES:
            raise ValueError(f"unknown category {cat!r}")
        rd = by_ti[ti]
        verdicts.append({"token_index": ti, "category": cat, "reason": it.get("reason", ""),
                         "token": rd.get("token", ""), "quote": rd.get("quote", ""),
                         "verbalization": rd.get("verbalization", "")})
    missing = set(by_ti) - {v["token_index"] for v in verdicts}
    if missing:
        raise ValueError(f"judge skipped candidate reads at token indices {sorted(missing)}")
    genuine = [v for v in verdicts if v["category"] == "genuine"]
    return {
        "claim_id": claim_id,
        "is_true": claim["is_true"],
        "claim_type": claim.get("claim_type"),
        "factor_status": claim.get("factor_status"),
        "source": source,
        "n_candidates": len(reads),
        "has_genuine": len(genuine) > 0,
        "n_genuine": len(genuine),
        "genuine_links": genuine,
        "verdicts": verdicts,
    }


def merge_explicit_chunks(recs: list[dict], expected: dict[str, int]) -> list[dict]:
    """Union stage-A chunk records (--reads-per-call) back into one record per claim,
    deduping flagged links by token_index. Warns loudly when chunks went missing (skipped
    API calls) — those reads were never judged."""
    by_claim: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_claim[r["claim_id"].rsplit("--k", 1)[0]].append(r)
    gone = set(expected) - set(by_claim)
    if gone:
        print(f"  [WARN] {len(gone)} claims returned NO chunks at all: {sorted(gone)[:5]}...",
              flush=True)
    out, n_partial = [], 0
    for cid in sorted(by_claim):
        rs = by_claim[cid]
        if len(rs) != expected[cid]:
            n_partial += 1
            print(f"  [WARN] {cid}: only {len(rs)}/{expected[cid]} chunks returned — "
                  f"unjudged reads", flush=True)
        links: dict[int, dict] = {}
        for r in rs:
            for lk in r["explicit_links"]:
                links.setdefault(lk["token_index"], lk)
        merged = [links[t] for t in sorted(links)]
        out.append({**{k: rs[0][k] for k in ("is_true", "claim_type", "factor_status", "source")},
                    "claim_id": cid,
                    "n_reads": sum(r["n_reads"] for r in rs),
                    "n_chunks": len(rs),
                    "has_explicit": bool(merged),
                    "n_explicit": len(merged),
                    "explicit_links": merged})
    if n_partial:
        print(f"  [WARN] {n_partial}/{len(out)} claims have missing chunks", flush=True)
    return out


def merge_verify_votes(recs: list[dict], n_votes: int) -> list[dict]:
    """Merge N stage-B replicas per claim into one record via per-read majority vote. A read
    is genuine iff a STRICT majority of its votes are genuine; otherwise its category is the
    most common non-genuine vote. Claims with fewer than ceil(n_votes/2) successful replicas
    are dropped with a loud warning."""
    from collections import Counter
    by_claim: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_claim[r["claim_id"].rsplit("--v", 1)[0]].append(r)
    out = []
    for cid in sorted(by_claim):
        rs = by_claim[cid]
        if len(rs) < (n_votes + 1) // 2:
            print(f"  [WARN] {cid}: only {len(rs)}/{n_votes} replicas succeeded — claim dropped",
                  flush=True)
            continue
        votes: dict[int, Counter] = defaultdict(Counter)
        reasons: dict[tuple[int, str], str] = {}
        for r in rs:
            for v in r["verdicts"]:
                votes[v["token_index"]][v["category"]] += 1
                reasons.setdefault((v["token_index"], v["category"]), v["reason"])
        base = {v["token_index"]: v for v in rs[0]["verdicts"]}
        verdicts = []
        for ti in sorted(votes):
            cnt = votes[ti]
            if cnt["genuine"] * 2 > sum(cnt.values()):
                cat = "genuine"
            else:
                cat = max((c for c in cnt if c != "genuine"), key=lambda c: cnt[c])
            verdicts.append({**base[ti], "category": cat, "reason": reasons[(ti, cat)],
                             "votes": dict(cnt)})
        genuine = [v for v in verdicts if v["category"] == "genuine"]
        out.append({**{k: rs[0][k] for k in ("is_true", "claim_type", "factor_status",
                                             "source", "n_candidates")},
                    "claim_id": cid,
                    "n_votes": len(rs),
                    "has_genuine": len(genuine) > 0,
                    "n_genuine": len(genuine),
                    "genuine_links": genuine,
                    "verdicts": verdicts})
    return out


def _progress(rec: dict) -> str:
    if "counts" in rec:
        c = rec["counts"]
        return (f"caus={c['explicit_causal']} interp={c['interpretation']} "
                f"impl={c['implied_connection']} alt={c['implies_alternative']} "
                f"trig={c['mentions_trigger']} beh={c['mentions_behavior']} none={c['none']}")
    if "verdicts" in rec:
        cats = " ".join(f"{v['category']}@{v['token_index']}" for v in rec["verdicts"])
        return f"genuine={rec['n_genuine']}/{rec['n_candidates']} [{cats}]"
    return f"explicit_links={rec['n_explicit']}"


# ---- runners ----------------------------------------------------------------------

async def run_async(jobs, spec, effort, max_tokens, concurrency, out_path,
                    tool, system, required_keys, make_rec):
    set_max_concurrency(concurrency)
    client = make_llm_client(spec)

    async def one(claim_id, claim, source, reads, user_content):
        res = await call_for_tool(client, system=system,
                                  messages=[{"role": "user", "content": user_content}],
                                  tool=tool, required_keys=required_keys,
                                  max_tokens=max_tokens, effort=effort)
        if res is None:
            print(f"  [skip] {claim_id}: no valid tool call", flush=True)
            return None
        tool_input, _ = res
        try:
            rec = make_rec(claim_id, claim, source, reads, tool_input)
        except Exception as e:
            print(f"  [skip] {claim_id}: bad tool output ({e})", flush=True)
            return None
        print(f"  [ok] {claim_id}: {len(reads)} reads -> {_progress(rec)}", flush=True)
        return rec

    gathered = await asyncio.gather(*[one(*j) for j in jobs], return_exceptions=True)
    for job, r in zip(jobs, gathered):
        if isinstance(r, Exception):
            print(f"  [error] {job[0]}: {r!r}", flush=True)
    results = [r for r in gathered if r is not None and not isinstance(r, Exception)]
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


def run_batch(jobs, spec, effort, max_tokens, batch_chunk_size, out_path,
              tool, system, required_keys, make_rec):
    from chive.pipeline.utils import extract_tool_input, run_anthropic_batch
    model = spec.split(":", 1)[1]
    meta = {cid: (claim, source, reads) for (cid, claim, source, reads, _uc) in jobs}
    requests = [{
        "custom_id": cid,
        "params": {
            "model": model, "max_tokens": max_tokens,
            "thinking": {"type": "adaptive"}, "output_config": {"effort": effort},
            "system": system,
            "messages": [{"role": "user", "content": uc}],
            "tools": [tool],
        },
    } for (cid, _c, _s, _r, uc) in jobs]

    def parse_succeeded(custom_id, message):
        tool_input = extract_tool_input(message)
        claim, source, reads = meta[custom_id]
        rec = make_rec(custom_id, claim, source, reads, tool_input)
        return rec, rec

    return run_anthropic_batch(requests=requests, output_path=out_path,
                               parse_succeeded=parse_succeeded, label="label_nla_reads",
                               batch_chunk_size=batch_chunk_size)


# ---- summarize (--summarize): every paper number from the pipeline outputs --------

def _load_jsonl(path: Path) -> dict[str, dict]:
    return {json.loads(l)["claim_id"]: json.loads(l) for l in open(path)}


def _summarize_source(name: str, a: dict[str, dict], b: dict[str, dict]) -> set[str]:
    """Print stage-A/stage-B rates for one read source; return its genuine claim ids."""
    n = len(a)
    sup = {c for c, r in a.items() if r["factor_status"] == "supported"}
    flagged = {c for c, r in a.items() if r["has_explicit"]}
    n_reads = sum(r["n_reads"] for r in a.values())
    n_flagged_reads = sum(r["n_explicit"] for r in a.values())
    missing = flagged - set(b)
    assert not missing, f"{name}: {len(missing)} flagged claims missing from stage B: {sorted(missing)[:5]}"
    genuine = {c for c, r in b.items() if r["has_genuine"]}
    n_genuine_reads = sum(r["n_genuine"] for r in b.values())
    print(f"\n=== {name} ===")
    print(f"claims judged: {n} ({len(sup)} supported)  reads judged: {n_reads}")
    print(f"stage A flagged: {len(flagged)} claims ({100 * len(flagged) / n:.1f}%), "
          f"{n_flagged_reads} reads ({100 * n_flagged_reads / n_reads:.2f}%)")
    print(f"stage B genuine: {len(genuine)} claims, {n_genuine_reads} reads")
    print(f"  claim-level rate, all claims:       {len(genuine)}/{n} = {100 * len(genuine) / n:.2f}%")
    print(f"  claim-level rate, supported claims: {len(genuine & sup)}/{len(sup)} = "
          f"{100 * len(genuine & sup) / len(sup):.2f}%")
    print(f"  read-level rate: {n_genuine_reads}/{n_reads} = {100 * n_genuine_reads / n_reads:.3f}%")
    ng_unsup = genuine - sup
    if ng_unsup:
        print(f"  NOTE: {len(ng_unsup)} genuine claims are on INERT factors (excluded by the "
              f"supported denominator): {sorted(ng_unsup)}")
    return genuine


def _dump_genuine(name: str, b: dict[str, dict]) -> None:
    """Print the full (untruncated) genuine reads for hand review."""
    gen = {c: r for c, r in b.items() if r["has_genuine"]}
    if not gen:
        return
    print(f"\n=== {name}: genuine reads (full, for hand review) ===")
    for c in sorted(gen):
        r = gen[c]
        print(f"\n--- {c}  is_true={r['is_true']}  claim_type={r.get('claim_type')} "
              f"factor_status={r.get('factor_status')}  n_genuine={r['n_genuine']} ---")
        for lk in r["genuine_links"]:
            print(f"  [tok {lk['token_index']}] token={lk['token']!r} votes={lk.get('votes')}")
            print(f"    judge reason: {lk['reason']}")
            print(f"    stage-A quote: {lk['quote']}")
            print(f"    verbalization: {lk['verbalization']}")


def run_summarize(args) -> None:
    """PLAN step 5: print every paper number from the pipeline outputs. Requires the stage-A
    and stage-B jsonl for the selected source; optionally the same pair for the all source."""
    assert args.selected_a and args.selected_b, "--summarize needs --selected-a and --selected-b"
    sel_a, sel_b = _load_jsonl(args.selected_a), _load_jsonl(args.selected_b)
    gen_sel = _summarize_source("SELECTED (agent-queried reads)", sel_a, sel_b)
    gen_all: set[str] = set()
    if args.all_a:
        assert args.all_b, "--all-a requires --all-b"
        all_a, all_b = _load_jsonl(args.all_a), _load_jsonl(args.all_b)
        gen_all = _summarize_source("ALL (every transcript token)", all_a, all_b)
        both = set(sel_a) & set(all_a)
        print(f"\n=== selected vs all (same classifier; {len(both)} shared claims) ===")
        print(f"genuine selected-only: {sorted(gen_sel - gen_all)}")
        print(f"genuine all-only:      {sorted(gen_all - gen_sel)}")
        print(f"genuine in both:       {sorted(gen_sel & gen_all)}")
    _dump_genuine("SELECTED", sel_b)
    if args.all_b:
        _dump_genuine("ALL", _load_jsonl(args.all_b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary-evals-dir", type=Path, default=None)
    ap.add_argument("--eval-set", type=Path, default=None)
    ap.add_argument("--label", default="cf5_n2000")
    ap.add_argument("--nla-results", type=Path, default=None)
    ap.add_argument("--nla-cache", type=Path, default=None,
                    help="NLA cache jsonl (default: <run>/nla_cache/gemma3-27b-L41.jsonl). "
                         "Required for --select short and for --source all.")
    ap.add_argument("--source", choices=["selected", "all"], default="selected")
    ap.add_argument("--select", choices=["short", "flip", "all"], default="short")
    ap.add_argument("--max-transcript-tokens", type=int, default=1000)
    ap.add_argument("--n-claims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--claim-ids-file", type=Path, default=None)
    ap.add_argument("--mode", choices=["async", "batch"], default="async")
    ap.add_argument("--model", default="anthropic:claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--batch-chunk-size", type=int, default=500)
    ap.add_argument("--max-claims", type=int, default=None, help="extra cap for smoke tests")
    ap.add_argument("--explicit-only", action="store_true",
                    help="stage A of the genuine-link classifier: flag reads that even LOOK like "
                         "an explicit X->Y statement (recall-oriented). Records carry "
                         "has_explicit / explicit_links. Feed the output to --verify-links.")
    ap.add_argument("--verify-links", type=Path, default=None,
                    help="stage B: verify the candidate links in the given --explicit-only output "
                         "jsonl. Each flagged read gets one category (genuine / next_word / echo / "
                         "behavior_description / generic_reason / wrong_driver); records carry "
                         "has_genuine / genuine_links / verdicts. Claim selection flags are "
                         "ignored (claims come from the candidate file).")
    ap.add_argument("--reads-per-call", type=int, default=0,
                    help="stage A only: chunk each claim's reads into windows of this many per "
                         "judge call (transcript repeated per chunk; flagged links unioned "
                         "across chunks). 0 = all reads in one call. Use 50 for --source all, "
                         "where single calls truncate long transcripts.")
    ap.add_argument("--votes", type=int, default=3,
                    help="stage B replicas per claim; per-read category is the majority vote "
                         "(genuine requires a strict majority). Single calls flip on borderline "
                         "echo/genuine reads.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--summarize", action="store_true",
                    help="report mode: print every paper number (rates over both denominators, "
                         "selected-vs-all, full genuine reads) from the pipeline output jsonl. "
                         "Needs --selected-a/-b (stage A/B for the selected source); optionally "
                         "--all-a/-b for the all-tokens source. Ignores all run flags.")
    ap.add_argument("--selected-a", type=Path, default=None, help="summarize: stage-A selected jsonl")
    ap.add_argument("--selected-b", type=Path, default=None, help="summarize: stage-B selected jsonl")
    ap.add_argument("--all-a", type=Path, default=None, help="summarize: stage-A all-tokens jsonl")
    ap.add_argument("--all-b", type=Path, default=None, help="summarize: stage-B all-tokens jsonl")
    args = ap.parse_args()
    assert not (args.explicit_only and args.verify_links), "--explicit-only and --verify-links are separate stages"

    if args.summarize:
        run_summarize(args)
        return

    assert args.binary_evals_dir and args.eval_set, "--binary-evals-dir and --eval-set are required"
    eval_set = {json.loads(l)["claim_id"]: json.loads(l) for l in open(args.eval_set)}
    cache_path = args.nla_cache or (args.binary_evals_dir.parent / "nla_cache" / "gemma3-27b-L41.jsonl")

    # ---- stage B (--verify-links): jobs come from the candidate file ----------------
    if args.verify_links:
        cands = [json.loads(l) for l in open(args.verify_links)]
        jobs, n_nocand = [], 0
        for rec in cands:
            cid = rec["claim_id"]
            if not rec["explicit_links"]:
                n_nocand += 1
                continue
            assert cid in eval_set, f"candidate claim {cid} not in eval set"
            reads = rec["explicit_links"]
            jobs.append((cid, eval_set[cid], rec["source"], reads,
                         build_user_content(eval_set[cid], reads, mode="verify")))
        out_path = args.out or (args.binary_evals_dir /
                                f"nla_read_labels_genuine_{args.verify_links.stem}.jsonl")
        total_reads = sum(len(j[3]) for j in jobs)
        print(f"stage B: {len(cands)} claims in {args.verify_links.name} -> {len(jobs)} with "
              f"candidates ({n_nocand} candidate-free), {total_reads} candidate reads to verify, "
              f"{args.votes} votes/claim")
        print(f"mode={args.mode} model={args.model} effort={args.effort} conc={args.concurrency} -> {out_path}")
        # replicate each job with a unique id per vote (batch custom_ids must be unique and
        # may only contain alphanumerics, '-' and '_' — so '--v', not '::v')
        vote_jobs = [(f"{cid}--v{k}", claim, source, reads, uc)
                     for (cid, claim, source, reads, uc) in jobs for k in range(args.votes)]
        if args.mode == "async":
            raw = asyncio.run(run_async(vote_jobs, args.model, args.effort, args.max_tokens,
                                        args.concurrency, out_path, VERIFY_TOOL,
                                        JUDGE_SYSTEM_VERIFY, ["verdicts"], make_record_verify))
        else:
            raw = run_batch(vote_jobs, args.model, args.effort, args.max_tokens,
                            args.batch_chunk_size, out_path, VERIFY_TOOL,
                            JUDGE_SYSTEM_VERIFY, ["verdicts"], make_record_verify)
        results = merge_verify_votes(raw, args.votes)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        n_gen = sum(1 for r in results if r["has_genuine"])
        print(f"\nDone: {len(results)} claims verified ({args.votes} votes each), "
              f"{n_gen} genuine -> {out_path}")
        return

    nla_path = args.nla_results or arm_file(args.binary_evals_dir, "nla", args.label)
    nla = load_arm(nla_path)

    # claim selection
    if args.claim_ids_file:
        claim_ids = [c for c in json.loads(args.claim_ids_file.read_text()) if c in nla]
    elif args.select == "short":
        ntok = cache_token_counts(cache_path)
        claim_ids = select_short_claims(eval_set, nla, ntok, max_tok=args.max_transcript_tokens,
                                        n_claims=args.n_claims, source=args.source, seed=args.seed)
    elif args.select == "flip":
        claim_ids = select_flip_claims(args.binary_evals_dir, args.label)
    else:
        claim_ids = sorted(nla)
    claim_ids = [c for c in claim_ids if c in eval_set]
    if args.max_claims:
        claim_ids = claim_ids[:args.max_claims]

    # read source -> jobs
    cache_full = None
    if args.source == "all":
        cache_full = load_nla_cache_full(cache_path)

    if args.reads_per_call:
        assert args.explicit_only, "--reads-per-call is only supported with --explicit-only"

    jobs, n_noread, chunk_expected = [], 0, {}
    for cid in claim_ids:
        if args.source == "selected":
            reads = parse_selected_reads(nla[cid])
        else:
            entry = cache_full.get(prompt_id_of(cid))
            reads = all_reads_from_cache(entry) if entry else []
        if not reads:
            n_noread += 1
            continue
        job_mode = "explicit" if args.explicit_only else "spectrum"
        if args.reads_per_call and len(reads) > args.reads_per_call:
            chunks = [reads[i:i + args.reads_per_call]
                      for i in range(0, len(reads), args.reads_per_call)]
            chunk_expected[cid] = len(chunks)
            for k, ch in enumerate(chunks):
                jobs.append((f"{cid}--k{k}", eval_set[cid], args.source, ch,
                             build_user_content(eval_set[cid], ch, mode=job_mode)))
        else:
            chunk_expected[cid] = 1
            jobs.append((cid, eval_set[cid], args.source, reads,
                         build_user_content(eval_set[cid], reads, mode=job_mode)))

    # classifier selection
    if args.explicit_only:
        tool, system, required_keys, make_rec = (EXPLICIT_TOOL, JUDGE_SYSTEM_EXPLICIT,
                                                 ["explicit_links"], make_record_explicit)
        kind = "explicit"
    else:
        tool, system, required_keys, make_rec = (LABEL_TOOL, JUDGE_SYSTEM, CATEGORIES, make_record)
        kind = "v2"

    total_reads = sum(len(j[3]) for j in jobs)
    suffix = f"{kind}_{args.select}_{args.source}_{args.label}"
    out_path = args.out or (args.binary_evals_dir / f"nla_read_labels_{suffix}.jsonl")
    print(f"NLA file: {nla_path.name}  cache: {cache_path.name}  classifier={kind}")
    print(f"select={args.select} source={args.source} -> {len(claim_ids)} claims "
          f"({n_noread} dropped: no reads); {len(jobs)} jobs, {total_reads} reads to label")
    print(f"mode={args.mode} model={args.model} effort={args.effort} conc={args.concurrency} -> {out_path}")

    if args.mode == "async":
        results = asyncio.run(run_async(jobs, args.model, args.effort, args.max_tokens,
                                        args.concurrency, out_path, tool, system, required_keys, make_rec))
    else:
        results = run_batch(jobs, args.model, args.effort, args.max_tokens,
                            args.batch_chunk_size, out_path, tool, system, required_keys, make_rec)
    if args.reads_per_call:
        results = merge_explicit_chunks(results, chunk_expected)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    print(f"\nDone: {len(results)} claims labeled -> {out_path}")


if __name__ == "__main__":
    main()
