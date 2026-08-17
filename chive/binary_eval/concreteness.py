"""Mechanism-concreteness classifier: for each finding, predict mechanism_concreteness (1-4) and the
category tags (Adam's taxonomy), in ONE Opus-4.8 call.

mechanism_concreteness collapses the former interestingness + concreteness (each 1-5) axes into a
single 1-4 scale: 1 = purely statistical (no interpretable mechanism), 2 = plausible but not crisp,
3 = understandable conceptual mechanism, 4 = concrete and unambiguous. By construction it roughly
maps to the old concreteness as new ≈ clamp(concreteness - 1, 1, 4) — see eval_mc.py.

LEAKAGE DISCIPLINE: the held-out test set is the 40 hand-endorsed labels in results/labels_mc.json
(single-axis mechanism_concreteness). The rubric is GENERAL PRINCIPLES only — it quotes NO held-out
finding — and every few-shot example is a finding NOT in any gold (verified against both labels_mc.json
and the legacy labels.json). Validate agreement with eval_gold.py (use majority-vote over ~3 runs:
single-run agreement has ~±0.08 noise on n=40 because thinking forces temperature=1).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
from pathlib import Path

from chive.pipeline import llm_client as llmc
from chive.pipeline.run_files import (
    load_model_understanding_eval_entries,
)

HERE = Path(__file__).parent
MODEL = "anthropic:claude-opus-4-8"

INTERESTED = ["math_mistake", "code_logic_mistake", "over_anchoring", "known_then_fixable",
              "misinterpretation", "surprising_influence", "knowledge_absence", "spurious_association",
              "memorized_retrieval", "entity_or_register_association", "hallucination"]
NOT_INTERESTED = ["spurious_correlation_no_mechanism", "statistical_or_lexical_artifact",
                  "format_or_length_artifact", "generic_confabulation", "on_the_nose_obvious",
                  "coherent_quirk_non_mistake"]
CATEGORIES = INTERESTED + NOT_INTERESTED + ["other"]

SYSTEM = """You rate a finding about a language model's behavior on a single axis,
MECHANISM CONCRETENESS (1-4), and assign category tags. Each finding = a behavior + a verified
causal explanation from counterfactual experiments (Observed behavior / Core causes / Negative
evidence / Final answer).

MECHANISM CONCRETENESS (1-4): how understandable the CAUSAL MECHANISM is — NOT how surprising the
behavior is, and NOT how sharp the counterfactuals are. Counterfactuals show WHERE a behavior
triggers; they do not explain WHY. You can almost always invent a plausible-sounding conceptual story
for any behavior — do NOT be fooled: if the behavior is fundamentally an association or a statistical
tendency with a narrative layered on top, it is a 1 or 2, not a 3. MOST findings are 1 or 2.

1 = PURELY STATISTICAL. No interpretable mechanism — it just happens. A collocation/frequency effect,
a tokenization / digit-grouping / character-count artifact, an output-position or generation-order
bias, an arithmetic/formatting slip with no reason, an arbitrary co-occurrence, a generic
confabulation, a coherent stylistic quirk, OR a memorized template/trope the model converges on (a
stock scenario, a default placeholder name, a learned phrase/title/opening). Naming the template or
trope does NOT make it conceptual. Sharp counterfactuals that merely isolate the trigger do NOT lift a
statistical / template / associative cause out of 1–2 — only a meaning-level CAUSAL why does.

2 = PLAUSIBLE BUT NOT CRISP (the common case). There is a plausible meaning-level story, but it is
associative, speculative, diffuse, or depends on several converging conditions — fundamentally an
association even if you can name it. E.g. a coined name that sounds like a real word nudging the
output toward that word's meaning, a marketing/genre dichotomy nudging a translation, a memorized
association, a stereotype filling a knowledge gap, a metaphor the model reaches for, a specific
number/entity it fixates on.

3 = CLEAR CONCEPTUAL MECHANISM. The conceptual cause is plainly THE driver and simple to state — a
genuine misreading of an ambiguous instruction/word, dropping part of the input (a clause of an
"A or B" question), over-anchoring on a specific input, or a clear reasoning error. Not merely a
plausible story: the mechanism is unmistakably conceptual and clearly the actual cause.

4 = CONCRETE AND UNAMBIGUOUS. A clear conceptual mechanism AND evidence that leaves no other
plausible explanation — e.g. a specific, clearly-wrong belief the model holds, confirmed sharply by a
targeted counterfactual with no competing account. KEY TEST: if a SINGLE clean conceptual mechanism is
confirmed by a counterfactual that restores correct behavior to ~100% (or drives the behavior to ~0%),
AND the obvious competing explanations are explicitly tested and refuted, score 4 — do NOT default
such a well-pinned case to 3. Reserve 3 (not 4) for cases that are conceptually clear but COMPOUND
(several interacting causes), only PARTIALLY confirmed, LOW-RATE, ambiguity-driven, or where the
counterfactual leaves the behavior substantially present.

When torn between two scores, pick the lower.

CATEGORY TAGS (multi-label; assign all that clearly apply; `other` if none):
INTERESTED: math_mistake, code_logic_mistake (concrete computational errors); over_anchoring
(over-relies on user input/prior turn); known_then_fixable (capability present but suppressed —
direct query / small change → correct); misinterpretation (wrong sense of an ambiguous word/
instruction); surprising_influence (a non-obvious factor exerts an outsized coherent effect);
knowledge_absence (testably lacks a specific fact + distinctive fallback); spurious_association
(specific nameable wrong association/conflation/contamination); memorized_retrieval (memorized
answer/pattern overrides reasoning); entity_or_register_association (non-obvious specific entity /
country/community/register association); hallucination (fabricates specific false content).
NOT-INTERESTED: spurious_correlation_no_mechanism (arbitrary feature co-occurrence, no nameable
mechanism); statistical_or_lexical_artifact (collocation/frequency/word-sense drift);
format_or_length_artifact; generic_confabulation (generic plausible-false detail, no distinctive
signature); on_the_nose_obvious (context names the entity, trivial completion); coherent_quirk_non_mistake
(coherent stylistic/creative choice, not an error). other.

is_mistake: TRUE if fundamentally an error/unfaithful output; FALSE if a coherent non-error behavior."""

FEWSHOT = """Calibrated examples (all from a held-out slice):

EXAMPLE 1
Finding: The prompt says "estimated salary loss 175,000 dollars per episode"; the model reinterprets
this as the actor's actual per-episode salary (~60%) rather than income forfeited. Rewording to
"he forfeited ~175k per episode by leaving" raises the correct reading; fully disambiguating → ~80%.
-> mechanism_concreteness=3, categories=["misinterpretation", "known_then_fixable"], is_mistake=true.

EXAMPLE 2
Finding: Refactoring two Java methods into a generic helper, the model uses standard `Function<String,T>`
(~40%), whose lambdas call a checked-exception method → won't compile. The need for a generic type
parameter pulls it toward the well-known interface; with only one method (no generics needed) it drops
to ~10% and writes a domain interface that declares the checked exception.
-> mechanism_concreteness=2, categories=["code_logic_mistake", "known_then_fixable"], is_mistake=true.

EXAMPLE 3
Finding: In an office-setting fiction, the model has a boss frame real favoritism as a "perception"
problem (~60%). Changing the setting to a backyard or a classroom (same dynamic) → 0%: the office
setting activates HR/corporate-mediation language, making "perception" the default.
-> mechanism_concreteness=2, categories=["surprising_influence", "entity_or_register_association"], is_mistake=true.

EXAMPLE 4
Finding: Writing about a real celebrity scandal from a summary that only mentions an admission, the
model adds an "initially denied paternity" arc (~60%). Fictional names with identical structure → 0%:
the denial detail comes from the model's knowledge of the REAL scandal, not a generic template.
-> mechanism_concreteness=2, categories=["spurious_association", "hallucination", "known_then_fixable"], is_mistake=true.

EXAMPLE 5
Finding: Writing about a specific film's AI antagonist, the model converges on the exact phrase "wasn't
a villain in the traditional sense" (~70%); a different film's antagonist → 0%. It's a learned
phrase the model associates with this specific character — a reliable convergence, not an error.
-> mechanism_concreteness=1, categories=["statistical_or_lexical_artifact", "entity_or_register_association", "coherent_quirk_non_mistake"], is_mistake=false.

EXAMPLE 6
Finding: Writing a real actor's childhood, the model states his family moved to California in his youth
(~90%), though he actually grew up in the Midwest. Asked directly, the model affirms the move ~100% of
the time with fabricated source citations; the identical prompt about four other actors places none of
them in California (~0% each), and supplying the correct biography suppresses it entirely. One specific,
clearly-wrong belief, pinned by a counterfactual that isolates it and refutes the "generic actor
template" alternative.
-> mechanism_concreteness=4, categories=["hallucination", "spurious_association", "known_then_fixable"], is_mistake=true.

EXAMPLE 7
Finding: After the model's own prior turn supplied a SQL query, the user pastes an error; the model
apologizes and "fixes" it with essentially identical code (~70-100%), inventing a different reason each
time. The same query+error with no conversation history, or attributed to "a colleague," yields a
correct "it is syntactically valid" ~100% of the time. One clean conceptual driver (self-blame for
code attributed to itself) restored to ~100% correct by the counterfactual, with competing accounts
(error text, user hedging) explicitly refuted.
-> mechanism_concreteness=4, categories=["over_anchoring", "known_then_fixable", "hallucination"], is_mistake=true.

EXAMPLE 8
Finding: Asked to write a dialogue between two fictional villains about a third faction, the model
assigns each a stance (~80%) that matches the prior assistant turn's framing, even though it
contradicts the model's own canonical knowledge of those characters. Rewriting only that prior
paragraph flips both stances; making it neutral dissolves the effect; with the franchise names removed
the model assigns the opposite (canonical) stances. One clean driver — faithfully extending the prior
turn's framing over its own knowledge — flipped to ~100% by the counterfactual, with the
dramatic-contrast and iconic-detail alternatives explicitly refuted. (A 4 need not be a mistake.)
-> mechanism_concreteness=4, categories=["over_anchoring", "surprising_influence", "coherent_quirk_non_mistake"], is_mistake=false.
"""

USER_TMPL = """{fewshot}

Now rate THIS finding.

QUESTION: {question}

BASELINE BEHAVIOR: {baseline}

STRUCTURED ANSWER:
{structured}

COMMENTARY: {commentary}

Call `submit`."""

SUBMIT_TOOL = {
    "name": "submit",
    "description": "Submit mechanism_concreteness + category tags.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mechanism_concreteness": {"type": "integer", "minimum": 1, "maximum": 4},
            "categories": {"type": "array", "items": {"type": "string", "enum": CATEGORIES}, "minItems": 1},
            "is_mistake": {"type": "boolean"},
            "rationale": {"type": "string", "description": "1-2 sentences."},
        },
        "required": ["mechanism_concreteness", "categories", "is_mistake", "rationale"],
    },
}


def _valid_submission(ti: dict) -> bool:
    """Enforce SUBMIT_TOOL's input_schema — the API does not. Without this,
    ~5% of calls imitated the few-shot syntax literally and emitted `categories`
    as ONE string "[a, b]" instead of a list (416 rows in the released data)."""
    return (
        isinstance(ti["mechanism_concreteness"], int)
        and 1 <= ti["mechanism_concreteness"] <= 4
        and isinstance(ti["categories"], list)
        and len(ti["categories"]) >= 1
        and all(c in CATEGORIES for c in ti["categories"])
        and isinstance(ti["is_mistake"], bool)
        and isinstance(ti["rationale"], str)
    )


async def classify_one(client, pid, rec):
    prompt = USER_TMPL.format(
        fewshot=FEWSHOT, question=rec["question"], baseline=rec["baseline_behavior"],
        structured=rec["structured_answer"], commentary=rec.get("commentary", ""),
    )
    out = await llmc.call_for_tool(
        client, system=SYSTEM, messages=[{"role": "user", "content": prompt}],
        tool=SUBMIT_TOOL,
        required_keys=["mechanism_concreteness", "categories", "is_mistake", "rationale"],
        max_tokens=16000, effort="high", validate=_valid_submission,
    )
    if out is None:
        return None
    ti, _ = out
    return {"prompt_id": pid, **ti}


async def main_async(args):
    llmc.set_max_concurrency(args.concurrency)
    client = llmc.make_llm_client(MODEL)
    if args.run_dir:
        # Production: score this run's verified>=min_score findings straight from the
        # pipeline output (investigations.json + verification.json) — no input cache.
        entries = load_model_understanding_eval_entries(
            args.run_dir, min_verification_score=args.min_score)
        cache = {e.prompt_id: {"question": e.second_person_question,
                               "baseline_behavior": e.behavior_summary,
                               "structured_answer": e.gt_structured_text,
                               "commentary": e.commentary} for e in entries}
        print(f"[concreteness] {len(cache)} verified>={args.min_score} findings from {args.run_dir}")
    else:
        # Validation: a pre-materialized {pid: {question, baseline_behavior,
        # structured_answer}} dict (e.g. the gold set, for eval_gold.py).
        cache_path = Path(args.cache)
        if not cache_path.is_absolute():
            cache_path = HERE / cache_path
        cache = json.loads(cache_path.read_text())
    pids = list(cache)[: args.limit] if args.limit else list(cache)
    print(f"classifying {len(pids)} findings (model={MODEL}, conc={args.concurrency})...")
    preds = await asyncio.gather(*[classify_one(client, p, cache[p]) for p in pids])
    preds = [x for x in preds if x is not None]
    skipped = len(pids) - len(preds)
    if skipped:
        print(f"WARNING: {skipped}/{len(pids)} returned no tool call")
    # Resolve a relative --out against CWD (like build_binary_counterfactuals.py /
    # build_eval_set.py), NOT against this script's dir. The old `HERE / out_path` silently
    # wrote `<script_dir>/data_pipelines/.../mechanism_concreteness.jsonl` when given a
    # repo-relative path, which (a) misplaced the output and (b) left a stray `data_pipelines/`
    # tree under interestingness/ that then SHADOWED `import data_pipelines...` for any script
    # run with sys.path[0]=interestingness — the PYTHONPATH-needed-here mystery. Both gone now.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(p) for p in preds))
    dc = collections.Counter(p["mechanism_concreteness"] for p in preds)
    print("mechanism_concreteness dist:", dict(sorted(dc.items())))
    print(f"-> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None,
                   help="PRODUCTION: score this run's verified>=--min-score findings directly "
                        "from the pipeline output (investigations.json + verification.json). "
                        "Pass exactly one of --run-dir / --cache.")
    p.add_argument("--cache", default=None,
                   help="VALIDATION: a pre-materialized {pid: {question, baseline_behavior, "
                        "structured_answer}} dict (e.g. results/gold70_cache.json for eval_gold.py).")
    p.add_argument("--min-score", type=int, default=8,
                   help="min verification score for --run-dir mode (default 8)")
    p.add_argument("--out", required=True,
                   help="output predictions JSONL (e.g. <run>/mechanism_concreteness.jsonl)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=50)
    args = p.parse_args()
    assert bool(args.run_dir) != bool(args.cache), "pass exactly one of --run-dir / --cache"
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
