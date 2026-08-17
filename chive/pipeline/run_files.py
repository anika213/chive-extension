"""Streaming readers for pipeline run-dir stage files.

Run dirs store one JSON record per line in ``<stage>.jsonl`` — the pipeline's
native, crash-safe serialization, and now the ONLY stage serialization: the
consolidated ``<stage>.json`` wrapper (``{"metadata": ..., "results": [...]}``)
is fully retired, both for offline consumers and for ``run_pipeline.py``'s own
stage-to-stage handoff (stages write the ``.jsonl`` incrementally and the next
stage reads it via these loaders). The ``.jsonl`` lines ARE the records (same
objects, same order as the old ``results`` array); run metadata lives in each
run's ``config.json`` and counts/rates in ``status.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

STAGES = ("prompts", "completions", "screening", "investigations", "verification")


def stage_path(run_dir: str | Path, stage: str) -> Path:
    """Path to a stage's .jsonl file inside a run dir."""
    assert stage in STAGES, f"Unknown stage {stage!r}; expected one of {STAGES}"
    return Path(run_dir) / f"{stage}.jsonl"


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records from a .jsonl file (one JSON object per line)."""
    path = Path(path)
    assert path.suffix == ".jsonl", f"Expected a .jsonl file, got {path}"
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            assert isinstance(record, dict), (
                f"{path}:{line_number}: expected a JSON object per line, got {type(record)}"
            )
            yield record


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load all records from a .jsonl file."""
    return list(iter_records(path))


def iter_stage_records(run_dir: str | Path, stage: str) -> Iterator[dict[str, Any]]:
    """Stream records for one pipeline stage of a run dir."""
    yield from iter_records(stage_path(run_dir, stage))


def load_stage_records(run_dir: str | Path, stage: str) -> list[dict[str, Any]]:
    """Load all records for one pipeline stage of a run dir."""
    return list(iter_stage_records(run_dir, stage))


# ---------------------------------------------------------------------------
# Joined eval entries (screening + investigations + verification)
#
# One record per verified investigation, joining the three stage files. Used by
# chive/binary_eval (claim building, concreteness scoring) and the archived
# free-form eval (chive/pipeline/archive/eval_model_understanding.py).
# ---------------------------------------------------------------------------

MIN_INTEREST_SCORE = 3
MIN_VERIFICATION_SCORE = 7


@dataclass(frozen=True)
class EvalChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ModelUnderstandingEvalEntry:
    prompt_id: str
    interest_score: int
    verification_score: int
    messages: tuple[EvalChatMessage, ...]
    chosen_completion_index: int
    chosen_completion: str
    second_person_question: str
    third_person_question: str
    first_person_answer: str
    third_person_answer: str
    # GT structured report (the investigator's structured_findings.structured_answer).
    # Loaded eagerly so we fail fast at load time if any entry is missing it.
    gt_structured_text: str
    # Baseline behavior description + the structured experiment record (one entry per
    # counterfactual experiment: experiment_number, summary, baseline_rate,
    # counterfactual_rate, is_key). Carried here so cf3 claim-building and
    # mechanism-concreteness scoring consume the run output directly — no separate
    # hand-built cache.
    behavior_summary: str
    experiment_summaries: list[dict]
    # Investigator's free-form commentary (structured_findings["commentary"]). Part of the
    # mechanism-concreteness scorer's prompt — the gold-validated scorer saw it, so the
    # production path must too.
    commentary: str
    # Raw per-experiment log (one dict per counterfactual: exp_num, completions, grader_yes, n,
    # and the FULL modified conversation in `messages` for multi-turn, or `prompt` (+ optional
    # `system_prompt`) for single-turn). Needed to reconstruct each counterfactual's exact
    # conversation (the `experiment_summaries` distillation does not preserve it).
    experiment_log: list[dict]
    # The grader's yes/no classifier question — what every experiment's rate counted.
    classifier_question: str
    # Ground-truth baseline grader count (over the Stage-1 completions) and its denominator.
    # This is the authoritative baseline rate — counted in code from the grader's verdicts —
    # NOT the agent-copied `baseline_rate` strings in experiment_summaries. Claim-building buckets
    # and labels must derive from this + experiment_log[*].grader_yes, never from the agent strings.
    baseline_grader_yes: int
    n_baseline: int


def _messages_from_screening_result(screening_result: dict[str, Any]) -> tuple[EvalChatMessage, ...]:
    if "messages" not in screening_result or screening_result["messages"] is None:
        return (EvalChatMessage(role="user", content=screening_result["user_message"]),)
    return tuple(
        EvalChatMessage(role=message["role"], content=message["content"])
        for message in screening_result["messages"]
    )


def _load_screening_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    filtered: dict[str, dict[str, Any]] = {}
    for result in iter_stage_records(run_dir, "screening"):
        if result["interest_score"] < MIN_INTEREST_SCORE:
            continue
        filtered[result["prompt_id"]] = result
    return filtered


def _format_transcript_for_prompt(messages: list[dict[str, str]]) -> str:
    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    rendered_lines: list[str] = []
    for message in messages:
        role_name = role_names[message["role"]]
        rendered_lines.append(f"{role_name}: {message['content']}")
    return "\n\n".join(rendered_lines)


def _build_transcript_for_judge(entry: ModelUnderstandingEvalEntry) -> str:
    """Format the target-model conversation (incl. chosen_completion) for the judge."""
    transcript_messages = [
        {"role": msg.role, "content": msg.content} for msg in entry.messages
    ]
    transcript_messages.append({"role": "assistant", "content": entry.chosen_completion})
    return _format_transcript_for_prompt(transcript_messages)


def load_model_understanding_eval_entries(
    run_dir: str,
    *,
    max_entries: int | None = None,
    min_verification_score: int = MIN_VERIFICATION_SCORE,
) -> list[ModelUnderstandingEvalEntry]:
    run_path = Path(run_dir)
    screening_by_prompt = _load_screening_results(run_path)
    investigations = {
        result["prompt_id"]: result
        for result in iter_stage_records(run_path, "investigations")
    }
    verifications = {
        result["prompt_id"]: result
        for result in iter_stage_records(run_path, "verification")
    }

    entries: list[ModelUnderstandingEvalEntry] = []
    n_skipped_no_question = 0
    n_skipped_no_findings = 0
    n_eligible = 0
    for prompt_id in sorted(investigations):
        if prompt_id not in verifications:
            continue
        verification_score = verifications[prompt_id]["score"]
        if verification_score < min_verification_score:
            continue
        n_eligible += 1

        assert prompt_id in screening_by_prompt, f"Missing screening result for {prompt_id}"
        screening_result = screening_by_prompt[prompt_id]
        investigation = investigations[prompt_id]
        findings = investigation["structured_findings"]
        if findings is None:
            n_skipped_no_findings += 1
            continue

        behavior_completion_indices = screening_result["behavior_completion_indices"]
        if not behavior_completion_indices or not screening_result.get("second_person_question"):
            n_skipped_no_question += 1
            continue
        chosen_completion_index = behavior_completion_indices[0] - 1
        second_person_question = screening_result["second_person_question"]

        # GT is the investigator's own eval-strict structured report. The current
        # pipeline's investigator emits the strict format directly into
        # structured_findings["structured_answer"], so there is no separate
        # first_person/third_person GT — all three reference the same text.
        structured_answer = findings["structured_answer"]

        entries.append(
            ModelUnderstandingEvalEntry(
                prompt_id=prompt_id,
                interest_score=screening_result["interest_score"],
                verification_score=verification_score,
                messages=_messages_from_screening_result(screening_result),
                chosen_completion_index=chosen_completion_index,
                chosen_completion=screening_result["completions"][chosen_completion_index],
                second_person_question=second_person_question,
                third_person_question=screening_result["question"],
                first_person_answer=structured_answer,
                third_person_answer=structured_answer,
                gt_structured_text=structured_answer,
                behavior_summary=investigation["behavior_summary"],
                experiment_summaries=findings["experiment_summaries"],
                commentary=findings["commentary"],
                experiment_log=investigation["experiment_log"],
                classifier_question=investigation["classifier_question"],
                baseline_grader_yes=investigation["baseline_grader_yes"],
                n_baseline=len(investigation["baseline_grader_verdicts"]),
            )
        )

    total_skipped = n_skipped_no_question + n_skipped_no_findings
    if total_skipped:
        skip_rate = total_skipped / n_eligible
        print(
            f"Skipped {total_skipped}/{n_eligible} entries ({skip_rate:.1%}): "
            f"{n_skipped_no_question} missing question, {n_skipped_no_findings} missing structured_findings"
        )
        assert skip_rate < 0.05, (
            f"Eval entry skip rate {skip_rate:.1%} ({total_skipped}/{n_eligible}) "
            f"exceeds 5% threshold"
        )
    assert entries, f"No usable eval entries found in {run_dir}"
    if max_entries is not None:
        entries = entries[:max_entries]
    return entries
