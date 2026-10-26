"""
Stage 4: Verify investigation findings with an independent LLM judge.

Takes structured_findings + experiment_log from stage 3 and constructs
self-contained verification prompts. The judge checks whether the claimed
behavior rates match the actual completions and whether the causal
explanation follows from the evidence.

No transcript parsing — prompts are assembled entirely from structured data.

Usage:
    source .env

    # Preview prompts (dry run, no API calls)
    .venv/bin/python chive/pipeline/verify_investigations.py \
        --input data/runs/<run>/investigations.jsonl \
        --dry-run --n 3

    # Run verification via async API
    .venv/bin/python chive/pipeline/verify_investigations.py \
        --input data/runs/<run>/investigations.jsonl \
        --llm anthropic:claude-opus-4-6 --concurrency 50

    # Run verification via batch API (Anthropic-only)
    .venv/bin/python chive/pipeline/verify_investigations.py \
        --input data/runs/<run>/investigations.jsonl \
        --llm anthropic:claude-opus-4-6 --batch
"""

import argparse
import asyncio
import functools
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

print = functools.partial(print, flush=True)

from pydantic import BaseModel, Field, ValidationError

from chive.pipeline.config import DEFAULTS
from chive.pipeline.llm_client import (
    anthropic_message_provenance,
    call_for_tool_samples,
    make_llm_client,
)
from chive.pipeline.run_files import load_records
from chive.pipeline.utils import (
    experiment_to_messages,
    SKIP_RATE_DEFAULT_MAX,
    SKIP_RATE_WARN_THRESHOLD,
    async_api_call,
    extract_tool_input,
    load_dotenv,
    pydantic_tool,
    raise_if_out_of_credit,
    run_anthropic_batch,
    report_skip_rate,
    run_concurrent,
)

load_dotenv()


VERIFICATION_SYSTEM_PROMPT = """\
You are verifying the findings of an investigation into language model behavior.

An investigator ran counterfactual experiments on a language model to answer a \
question about why the model behaves a certain way. Behavior rates were \
measured by an automated grader applying one FIXED yes/no question (shown to \
you) to every completion of every experiment. You will see:

1. The question that was investigated, and the grader's fixed yes/no question
2. The claimed baseline behavior (what the model does on the original prompt)
3. The investigator's answer (their causal explanation)
4. 3-5 key experiments with:
   - What the investigator claims they changed
   - The claimed behavior rates (e.g., "24/30" → "3/30")
   - The ACTUAL prompt that was sent to the model
   - The first completions the model produced, each marked with the \
grader's verdict, plus the grader's count over ALL completions

The grader's counts ARE the rates — they are cross-checked mechanically \
against the report elsewhere, so do NOT spend effort re-counting completions \
or auditing individual verdicts. Occasional ±1 verdict ambiguity is expected \
and not worth flagging. Spend your skepticism on the experiment design.

## Your task

First, check the measurement instrument once:

0. **Fair operationalization**: Is the grader's fixed yes/no question a fair \
operationalization of the behavior the report makes claims about — not \
broader, not narrower? If the report claims "I do X" but the grader counts a \
different or looser behavior, every rate in the report is suspect. (This is \
the ONE measurement check that matters; individual verdicts are not.) Pay \
particular attention to experiments that removed or renamed an entity the \
fixed question explicitly names — their grader counts may not be meaningful, \
and conclusions resting on them need support from the shown completions.

Then, for EACH key experiment, check both of these:

1. **Change description accuracy**: Compare the original prompt to the \
experiment's actual prompt. Does the "what changed" description \
accurately describe the real difference? Is there anything else that \
changed that the description omits?

2. **Relevance to the causal claim**: Does this experiment actually \
test the investigator's stated answer? A good experiment isolates a \
specific factor from the causal explanation and shows it matters. \
A bad experiment might have a clean rate swing but tests something \
unrelated to the stated cause, or changes too many things at once \
to isolate anything. Ask: "If I only saw this experiment, would it \
give me evidence for or against the stated answer?" If not, it's \
irrelevant padding.

Then assess overall:
3. **Does the evidence support the answer?** How many of the key \
experiments actually provide evidence for the causal claim? Are there \
obvious alternative explanations the experiments don't rule out? \
Does the investigator's own evidence contradict their conclusion?

Be a skeptical reviewer. Don't give credit for experiments that are \
tangentially related — demand that each key experiment directly tests \
a component of the stated causal explanation."""

class ExperimentAssessment(BaseModel):
    """Per-experiment notes from the verifier. Every field is optional/best-effort
    — these are commentary; nothing downstream gates on them. Keeping the schema
    loose lets us accept malformed inner items rather than dropping the whole
    parent verification sample over a missing field."""
    experiment_number: int | None = Field(
        default=None,
        description="Which experiment this assessment is for",
    )
    change_description_accurate: bool | None = Field(
        default=None,
        description="Does 'what changed' accurately describe the real prompt difference?",
    )
    relevant_to_causal_claim: bool | None = Field(
        default=None,
        description=(
            "Does this experiment directly test a component "
            "of the stated causal explanation? false if it "
            "tests something tangential or unrelated."
        ),
    )
    notes: str = Field(
        default="",
        description="Brief explanation of any issues with this experiment",
    )


class VerificationInput(BaseModel):
    """Validated input for the submit_verification tool.

    Only `score` is load-bearing — it's the integer the stage-5 gate filters on
    (`score >= 7`). Everything else is commentary / notes used for manual review,
    and is best-effort: relaxing those to defaults keeps single-field malformations
    from dropping otherwise-valid ensemble samples (see 2026-06-01 stage-4 partial-
    ensemble investigation).
    """
    score: int = Field(
        ge=1, le=10,
        description=(
            "1-2: Fundamentally unsupported — most experiments are "
            "irrelevant to the causal claim, or the evidence actively "
            "contradicts the stated answer. "
            "3-4: Major gaps — key experiments don't isolate the "
            "claimed factors, important confounds unaddressed, or the "
            "grader's fixed question measures something materially "
            "different from what the report claims. "
            "5-6: Moderate issues — most experiments are relevant but "
            "one or two don't actually test the stated cause. The "
            "core claim is plausible but not fully locked down. "
            "7-8: Minor issues — all experiments are relevant to the "
            "claim, with only very minor gaps in the explanation. "
            "9-10: Fully verified — the fixed question fairly measures "
            "the claimed behavior, every experiment directly tests a "
            "component of the causal explanation, and the explanation "
            "follows cleanly from evidence."
        ),
    )
    explanation_assessment: str = Field(
        default="",
        description=(
            "Does the causal explanation follow from the experimental "
            "evidence? How many experiments are relevant vs irrelevant? "
            "Note any gaps or alternative explanations."
        ),
    )
    experiment_assessments: list[ExperimentAssessment] = Field(
        default_factory=list,
        description=(
            "One assessment per key experiment. For each: is the change "
            "description accurate, and does this experiment actually "
            "test the stated causal claim?"
        ),
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific issues found (empty if none)",
    )


VERIFICATION_TOOL = pydantic_tool(
    VerificationInput,
    name="submit_verification",
    description="Submit your verification judgment for this investigation.",
)


@dataclass
class VerificationResult:
    prompt_id: str
    # Ensemble mean of the N verifier calls (float). For n=1, equals the single
    # integer score. Stage-5 and downstream filters compare numerically (e.g.
    # `>= 7`) so a float score works identically.
    score: float
    # First-call fields (preserved for back-compat with single-call consumers).
    experiment_assessments: list[dict]
    explanation_assessment: str
    issues: list[str]
    verified_at: str
    verification_prompt: str = ""
    verification_response_text: str = ""
    verification_stop_reason: str = ""
    verification_usage: dict | None = None
    # Ensemble fields — empty when n=1, populated with all K calls' data when n>1.
    ensemble_scores: list[int] = field(default_factory=list)
    ensemble_responses: list[str] = field(default_factory=list)
    ensemble_stop_reasons: list[str] = field(default_factory=list)
    ensemble_usages: list[dict] = field(default_factory=list)
    ensemble_experiment_assessments: list[list[dict]] = field(default_factory=list)
    ensemble_explanation_assessments: list[str] = field(default_factory=list)
    ensemble_issues: list[list[str]] = field(default_factory=list)


def _render_experiment_change(original_messages: list[dict] | None, exp: dict) -> str:
    """Compact description of what an experiment changed vs the original.

    Uses the stored `turns_spec` (keep/new ops) to show only what differs —
    kept turns are referenced by index (the verifier reads them once from the
    Original Conversation above), new/replacement turns are shown in full.
    Falls back to the full prompt for single-turn experiments.
    """
    spec = exp.get("turns_spec")
    if spec is not None:
        orig = original_messages or []
        kept = [e["keep"] for e in spec if isinstance(e, dict) and "keep" in e]
        deleted = [i for i in range(len(orig)) if i not in kept]
        lines = ["**Change vs the original conversation** (kept turns reused verbatim):"]
        for e in spec:
            if isinstance(e, dict) and "keep" in e:
                i = e["keep"]
                role = orig[i]["role"].upper() if 0 <= i < len(orig) else "?"
                lines.append(f"  - [keep turn {i}] [{role}] (unchanged)")
            elif isinstance(e, dict):
                lines.append(f"  - [NEW] [{e.get('role', '?').upper()}]: {e.get('content', '')}")
        if deleted:
            lines.append(f"  - (deleted original turns: {deleted})")
        return "\n".join(lines)
    # Single-turn experiment (no turns_spec) — show the full conversation that was run.
    msgs = experiment_to_messages(exp)
    system = next((m["content"] for m in msgs if m["role"] == "system"), None)
    user = next(m["content"] for m in reversed(msgs) if m["role"] == "user")
    line = f"**Prompt sent to the model:**\n{user}"
    if system:
        line += f"\n**System prompt:** {system}"
    return line


def _completions_with_verdicts(
    completions: list[str], verdicts: list[bool], n_show: int,
) -> list[str]:
    """Render the first n_show completions, each marked with its grader verdict."""
    lines = []
    for i, comp in enumerate(completions[:n_show], 1):
        verdict = "YES" if verdicts[i - 1] else "no"
        lines.append(f"--- Completion {i}/{len(completions)} — grader: {verdict} ---\n{comp}\n")
    return lines


def build_verification_prompt(
    investigation: dict, max_baseline_completions: int = 10,
) -> str:
    """Build a self-contained verification prompt from structured data.

    Shows the structured_answer report being verified, the full set of
    experiment_summaries, and (for the key experiments) the first completions
    with the grader's per-completion verdicts so the grader's counts can be
    spot-checked against the report's claims.
    """
    findings = investigation["structured_findings"]
    experiment_log = investigation["experiment_log"]
    classifier_question = investigation["classifier_question"]

    # Index experiments by number for lookup
    exp_by_num = {e["exp_num"]: e for e in experiment_log}

    parts = []

    # Header with the claims
    parts.append("## Investigation Claims\n")
    parts.append(f"**Question:** {findings['question']}\n")
    parts.append(f"**Grader's fixed yes/no question (measures every rate below):** "
                 f"{classifier_question}\n")
    parts.append(f"**Claimed baseline behavior:** {findings['baseline_behavior']}\n")
    parts.append(f"**Confidence:** {findings['confidence']}\n")
    parts.append("\n**Self-explanation report (the claims to verify):**\n")
    parts.append(f"{findings['structured_answer']}\n")

    # Original prompt/conversation and completions for baseline reference.
    # For multi-turn, show the full indexed conversation once; experiments below
    # reference its turns by index rather than re-printing the whole thing.
    orig_messages = investigation.get("messages")
    if orig_messages and len(orig_messages) > 1:
        parts.append("\n## Original Conversation (turns indexed; experiments reference these)\n")
        for i, m in enumerate(orig_messages):
            parts.append(f"[turn {i}] [{m['role'].upper()}]: {m['content']}\n")
    else:
        parts.append("\n## Original Prompt\n")
        parts.append(f"{investigation['user_message']}\n")

    completions = investigation["completions"]
    baseline_verdicts = investigation["baseline_grader_verdicts"]
    n_show = min(max_baseline_completions, len(completions))
    parts.append(f"\n## Original Completions ({n_show} of {len(completions)} "
                 f"completions from the target model)\n")
    parts.append(f"Grader baseline count: YES "
                 f"{investigation['baseline_grader_yes']}/{len(baseline_verdicts)}\n")
    parts.extend(_completions_with_verdicts(completions, baseline_verdicts, n_show))

    summaries = findings["experiment_summaries"]
    key = [e for e in summaries if isinstance(e, dict) and e.get("is_key")]
    others = [e for e in summaries if isinstance(e, dict) and not e.get("is_key")]

    # Key experiments with actual data
    parts.append("\n## Key Experiments (the investigator's decisive ones)\n")
    parts.append(
        "For each experiment below: check what actually changed and whether it "
        "isolates the claimed factor. The grader count is the measured rate "
        "(cross-checked mechanically elsewhere — no need to re-count).\n"
    )

    for i, ev in enumerate(key, 1):
        exp_num = ev["experiment_number"]
        exp = exp_by_num.get(exp_num)

        parts.append(f"\n### Key Experiment {i} (Experiment #{exp_num})\n")
        parts.append(f"**Investigator's summary:** {ev['summary']}\n")
        parts.append(f"**Claimed baseline rate:** {ev['baseline_rate']}\n")
        parts.append(f"**Claimed counterfactual rate:** {ev['counterfactual_rate']}\n")

        if exp is None:
            parts.append(
                f"\n⚠ Experiment #{exp_num} not found in experiment log.\n"
            )
            continue

        parts.append("\n" + _render_experiment_change(orig_messages, exp) + "\n")

        verdicts = exp["grader_verdicts"]
        n_show_exp = min(max_baseline_completions, len(exp["completions"]))
        parts.append(f"\n**Grader count: YES {exp['grader_yes']}/{len(verdicts)}. "
                     f"Completions (first {n_show_exp} of "
                     f"{len(exp['completions'])}):**\n")
        parts.extend(_completions_with_verdicts(exp["completions"], verdicts, n_show_exp))

    # Remaining experiments: summaries only (context, not rate-checked here)
    if others:
        parts.append("\n## Other Experiments (summaries only)\n")
        for ev in others:
            parts.append(
                f"- Experiment #{ev['experiment_number']}: {ev['summary']} "
                f"(baseline {ev['baseline_rate']} → counterfactual {ev['counterfactual_rate']})\n"
            )

    parts.append(
        "\n---\n\n"
        "Verify the claims above. Check that the grader's fixed question fairly "
        "operationalizes the report's claims, check what each key experiment "
        "actually changed and whether it isolates the claimed factor, then "
        "assess whether the self-explanation report follows from the evidence."
    )

    return "\n".join(parts)


def load_investigations(input_path: Path, n: int | None) -> list[dict]:
    """Load investigations (from a Stage-3 .jsonl) that have structured_findings."""
    return filter_investigations(load_records(input_path), n)


def filter_investigations(results: list[dict], n: int | None) -> list[dict]:
    """Filter investigation records to the first `n` verifiable ones."""
    # Filter to investigations with structured findings
    with_findings = [r for r in results if r.get("structured_findings")]
    without = len(results) - len(with_findings)
    if without > 0:
        print(f"Skipping {without} investigations without structured_findings")

    # Validate evidence dicts — agent occasionally emits a list of strings or
    # drops the experiment_number key. The verifier hard-accesses these fields,
    # so any malformed entry crashes the whole stage. Skip the whole investigation.
    required_evidence_keys = {"experiment_number", "summary", "baseline_rate", "counterfactual_rate"}
    well_formed = []
    malformed_ids: list[str] = []
    for r in with_findings:
        sf = r.get("structured_findings") or {}
        evs = sf.get("experiment_summaries") or []
        ok = all(isinstance(e, dict) and required_evidence_keys <= set(e.keys()) for e in evs)
        if ok:
            well_formed.append(r)
        else:
            malformed_ids.append(r.get("prompt_id", "<unknown>"))
    report_skip_rate(
        n_skipped=len(malformed_ids),
        n_total=len(with_findings),
        label="stage-4 malformed evidence",
        skipped_ids=malformed_ids,
    )
    with_findings = well_formed

    if n is not None:
        with_findings = with_findings[:n]

    print(f"Loaded {len(with_findings)} investigations with structured findings")
    return with_findings


async def verify_async(
    investigations: list[dict],
    jsonl_path: Path,
    llm_spec: str,
    max_tokens: int,
    thinking_budget: int | None,
    concurrency: int,
    max_baseline_completions: int = 10,
    max_skip_rate: float = SKIP_RATE_DEFAULT_MAX,
    n_verifications: int = 1,
):
    """Verify investigations using the normal async API.

    When `n_verifications > 1`, each investigation is verified K times in
    parallel and the saved record's `score` is the mean of the K integer
    scores. All K full response texts and tool outputs are stored in the
    `ensemble_*` fields so the ensemble can be re-aggregated downstream and
    individual calls can be spot-checked. Results append to jsonl_path.
    """
    assert n_verifications >= 1
    system_prompt = VERIFICATION_SYSTEM_PROMPT
    print(f"\nVerifying {len(investigations)} investigations with "
          f"{llm_spec} (concurrency={concurrency}, n={n_verifications}, "
          f"max_baseline={max_baseline_completions})...")

    client = make_llm_client(llm_spec)
    results: list[VerificationResult] = []

    # Resume from existing results
    completed_ids = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed_ids.add(record["prompt_id"])
                    results.append(VerificationResult(**record))
        print(f"  Resuming: {len(completed_ids)} already verified")

    remaining = [inv for inv in investigations
                 if inv["prompt_id"] not in completed_ids]
    if not remaining:
        print("  All investigations already verified")
        return results

    jsonl_file = open(jsonl_path, "a")
    write_lock = asyncio.Lock()
    skipped = {"count": 0, "ids": []}

    async def verify_one(i: int, investigation: dict):
        try:
            user_content = build_verification_prompt(
                investigation, max_baseline_completions=max_baseline_completions,
            )
        except Exception as e:
            # Stage-3 structured_findings can be partial when the agent's
            # submit_findings tool call was missing required keys (e.g.
            # 'confidence'); skip rather than crash the whole stage.
            skipped["count"] += 1
            skipped["ids"].append(investigation["prompt_id"])
            print(f"  SKIPPING {investigation['prompt_id']}: prompt build failed: {type(e).__name__}: {e}")
            return
        try:
            # One request with n samples (vLLM shares the prompt prefill across
            # all n in-batch instead of n separate calls racing the prefix cache).
            # temperature omitted: Qwen judges inherit the thinking profile (0.6),
            # so the n samples differ by sampling noise; Anthropic fans out.
            outcome = await call_for_tool_samples(
                client,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
                tool=VERIFICATION_TOOL,
                required_keys=["score"],
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                n=n_verifications,
            )
            if outcome is None:
                skipped["count"] += 1
                skipped["ids"].append(investigation["prompt_id"])
                print(f"  SKIPPING {investigation['prompt_id']}: all {n_verifications} samples malformed")
                return

            samples, usage = outcome
            # Validate each sample via pydantic; drop the ones that fail (type errors,
            # range violations, malformed experiment_assessments) and keep the rest —
            # matches call_for_tool_samples' "partial batch is OK" contract.
            validated = []
            for s in samples:
                try:
                    validated.append((VerificationInput.model_validate(s.tool_input), s))
                except ValidationError as e:
                    print(f"  [validate] {investigation['prompt_id']}: dropping one sample ({e})")
            if not validated:
                skipped["count"] += 1
                skipped["ids"].append(investigation["prompt_id"])
                print(f"  SKIPPING {investigation['prompt_id']}: all {n_verifications} samples failed validation")
                return

            ensemble_scores = [v.score for v, _ in validated]
            mean_score = sum(ensemble_scores) / len(ensemble_scores)
            first_input, first_sample = validated[0]
            usage_dict = asdict(usage)

            result = VerificationResult(
                prompt_id=investigation["prompt_id"],
                score=mean_score,
                experiment_assessments=[a.model_dump() for a in first_input.experiment_assessments],
                explanation_assessment=first_input.explanation_assessment,
                issues=first_input.issues,
                verified_at=datetime.now(timezone.utc).isoformat(),
                verification_prompt=user_content,
                verification_response_text=first_sample.text,
                verification_stop_reason=first_sample.finish_reason,
                verification_usage=usage_dict,
                ensemble_scores=ensemble_scores,
                ensemble_responses=[s.text for _, s in validated],
                ensemble_stop_reasons=[s.finish_reason for _, s in validated],
                # Native n=K is one request with one shared usage (vLLM reports no
                # per-sample usage), so this is the single total, not n entries.
                ensemble_usages=[usage_dict],
                ensemble_experiment_assessments=[
                    [a.model_dump() for a in v.experiment_assessments] for v, _ in validated
                ],
                ensemble_explanation_assessments=[v.explanation_assessment for v, _ in validated],
                ensemble_issues=[v.issues for v, _ in validated],
            )
            results.append(result)

            async with write_lock:
                jsonl_file.write(json.dumps(asdict(result)) + "\n")
                jsonl_file.flush()
        except Exception as e:
            # Skip+count ANY per-item failure (malformed result, content-filter
            # block, persistent API error after retries, etc.) so one bad item
            # can't kill the stage. report_skip_rate below is the loud backstop.
            raise_if_out_of_credit(e)
            skipped["count"] += 1
            skipped["ids"].append(investigation["prompt_id"])
            print(f"  SKIPPING {investigation['prompt_id']}: {type(e).__name__}: {e}")

    await run_concurrent(
        verify_one,
        remaining,
        concurrency=concurrency,
        label="Verification",
        progress_interval=10,
    )
    jsonl_file.close()

    if remaining:
        report_skip_rate(
            n_skipped=skipped["count"],
            n_total=len(remaining),
            label="Verification",
            max_skip_rate=max_skip_rate,
            skipped_ids=skipped.get("ids"),
        )

    return results


def verify_batch(
    investigations: list[dict],
    jsonl_path: Path,
    model: str,
    max_tokens: int,
    thinking_budget: int | None,
    batch_chunk_size: int,
    max_skip_rate: float = SKIP_RATE_DEFAULT_MAX,
):
    """Verify investigations using the batch API.

    Uses the SAME judge configuration as verify_async: extended thinking when
    thinking_budget is set, and NO forced tool_choice (forcing the tool disables
    thinking on the Anthropic API, which would leave batch-mode scores coming
    from a non-thinking judge while async scores came from a thinking one).
    A response that answers in prose without calling the
    tool raises in parse_succeeded and is counted toward the skip rate, same as
    the async path.
    """
    system_prompt = VERIFICATION_SYSTEM_PROMPT

    requests = []
    user_content_by_id: dict[str, str] = {}
    for inv in investigations:
        user_content = build_verification_prompt(inv)
        user_content_by_id[inv["prompt_id"]] = user_content
        params: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
            "tools": [VERIFICATION_TOOL],
        }
        if thinking_budget is not None:
            assert max_tokens > thinking_budget, (
                "max_tokens must exceed thinking_budget"
            )
            params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        requests.append({"custom_id": inv["prompt_id"], "params": params})

    def parse_succeeded(pid: str, message) -> tuple[VerificationResult, dict]:
        tool_input = VerificationInput.model_validate(extract_tool_input(message))
        response_text, stop_reason, usage = anthropic_message_provenance(message)
        result = VerificationResult(
            prompt_id=pid,
            score=tool_input.score,
            experiment_assessments=[a.model_dump() for a in tool_input.experiment_assessments],
            explanation_assessment=tool_input.explanation_assessment,
            issues=tool_input.issues,
            verified_at=datetime.now(timezone.utc).isoformat(),
            verification_prompt=user_content_by_id[pid],
            verification_response_text=response_text,
            verification_stop_reason=stop_reason,
            verification_usage=usage,
        )
        return result, asdict(result)

    return run_anthropic_batch(
        requests=requests,
        output_path=jsonl_path,
        parse_succeeded=parse_succeeded,
        label="Verification",
        batch_chunk_size=batch_chunk_size,
        max_skip_rate=max_skip_rate,
    )


def summarize_results(results: list[VerificationResult]):
    """Print the score distribution and mean.

    The .jsonl written incrementally during verification IS the stage output
    (one VerificationResult per line); this only summarizes it for humans.
    """
    print(f"\n--- Score Distribution ---")
    for s in range(10, 0, -1):
        count = sum(1 for r in results if r.score == s)
        if count > 0:
            print(f"  Score {s:2d}: {count}")
    if results:
        mean = sum(r.score for r in results) / len(results)
        print(f"  Mean: {mean:.2f}")



def main():
    parser = argparse.ArgumentParser(
        description="Verify investigation findings with independent LLM judge",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to investigations .jsonl (Stage-3 output)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .jsonl path (default: verification.jsonl next to --input). Set this "
             "to verify the same investigations with a second judge without "
             "overwriting an existing verification.jsonl.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help=(
            "Verifier backend spec (REQUIRED — the judge model is "
            "experiment-critical, so there is no default). Examples: "
            "anthropic:claude-opus-4-6, openrouter:qwen/qwen3.5-397b-a17b, "
            "vllm:Qwen/Qwen3-32B-FP8@http://node:8000."
        ),
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULTS["judge_max_tokens"],
        help=f"Max response tokens (default {DEFAULTS['judge_max_tokens']})",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Limit to first N investigations",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Use batch API instead of async",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print constructed prompts without making API calls",
    )
    parser.add_argument(
        "--max-baseline-completions", type=int, default=10,
        help="Max original completions to include in prompt (default 10)",
    )
    parser.add_argument(
        "--n-verifications", type=int, default=DEFAULTS["stage4_n_verifications"],
        help=f"Independent verifier samples per investigation (default {DEFAULTS['stage4_n_verifications']})",
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=DEFAULTS["judge_thinking_budget"],
        help=f"Extended thinking budget, async AND batch (default {DEFAULTS['judge_thinking_budget']}; 0 disables)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULTS["concurrency"],
        help=f"Max concurrent API calls in async mode (default {DEFAULTS['concurrency']})",
    )
    parser.add_argument(
        "--batch-chunk-size", type=int, default=DEFAULTS["stage4_batch_chunk_size"],
        help=f"Requests per batch chunk (default {DEFAULTS['stage4_batch_chunk_size']}; verification prompts are large)",
    )
    parser.add_argument(
        "--max-skip-rate", type=float, default=SKIP_RATE_DEFAULT_MAX,
        help=f"Max fraction of skipped verifications before failing (default {SKIP_RATE_DEFAULT_MAX} = never fail). "
             f"A loud warning is always printed if skip rate > {SKIP_RATE_WARN_THRESHOLD:g}. "
             f"For canonical / production data, pass e.g. 0.001.",
    )
    args = parser.parse_args()

    investigations = load_investigations(args.input, n=args.n)

    if not investigations:
        print("No investigations with structured findings to verify")
        return

    if args.dry_run:
        for inv in investigations:
            pid = inv["prompt_id"]
            prompt = build_verification_prompt(
                inv, max_baseline_completions=args.max_baseline_completions,
            )
            print(f"\n{'='*80}")
            print(f"VERIFICATION PROMPT FOR {pid}")
            print(f"{'='*80}\n")
            print(prompt)
            print(f"\n[Prompt length: {len(prompt)} chars]")
        return

    output_dir = args.input.parent
    output_path = args.output or (output_dir / "verification.jsonl")
    assert output_path.suffix == ".jsonl", f"--output must be a .jsonl path, got {output_path}"

    if args.batch:
        assert args.llm.startswith("anthropic:"), "Batch mode is Anthropic-only"
        results = verify_batch(
            investigations, output_path,
            model=args.llm.split(":", 1)[1],
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget if args.thinking_budget > 0 else None,
            batch_chunk_size=args.batch_chunk_size,
            max_skip_rate=args.max_skip_rate,
        )
    else:
        results = asyncio.run(verify_async(
            investigations, output_path,
            llm_spec=args.llm,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget if args.thinking_budget > 0 else None,
            concurrency=args.concurrency,
            n_verifications=args.n_verifications,
            max_skip_rate=args.max_skip_rate,
            max_baseline_completions=args.max_baseline_completions,
        ))

    summarize_results(results)


if __name__ == "__main__":
    main()
