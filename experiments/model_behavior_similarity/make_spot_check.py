"""Qualitative spot check for the behavior-similarity check (see README.md).

Renders the largest-disagreement units (top 5 per direction by |cross − self|)
and 3 near-agreement units per direction, each with the full conversation, the
classifier question, both rates, and full completions (no truncation) from both
models — with per-completion verdicts for the cross-graded side.
"""

import functools
import json
from pathlib import Path

print = functools.partial(print, flush=True)

from chive import paths

HERE = Path(__file__).parent
RUNS = paths.RUNS_DIR
QWEN_RUN = "opus_investigates_qwen3_8b_5000_thorough_n30"
LLAMA_RUN = "opus_investigates_llama3_1_8b_5000_thorough_n30"
VERDICTS_PATH = paths.experiment_data_dir("model_behavior_similarity") / "results" / "verdicts.jsonl"
OUT_PATH = paths.experiment_data_dir("model_behavior_similarity") / "results" / "spot_check.md"

CROSS_DIRECTIONS = ["qwen_bank_on_llama", "llama_bank_on_qwen"]
MODEL_NAME = {QWEN_RUN: "Qwen3-8B", LLAMA_RUN: "Llama-3.1-8B"}
N_TOP_DISAGREE = 5
N_AGREE = 3
N_COMPLETIONS_SHOWN = 3


def load_completions(pids: set[str]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for run in (QWEN_RUN, LLAMA_RUN):
        out[run] = {}
        for line in open(RUNS / run / "completions.jsonl"):
            d = json.loads(line)
            if d["id"] in pids:
                out[run][d["id"]] = d
    return out


def pick_indices(verdicts: list[bool]) -> list[int]:
    """Up to 2 'yes' + 1 'no' completions for verdict diversity, else first 3."""
    yes = [i for i, v in enumerate(verdicts) if v]
    no = [i for i, v in enumerate(verdicts) if not v]
    picked = yes[:2] + no[:1]
    for i in range(len(verdicts)):
        if len(picked) >= N_COMPLETIONS_SHOWN:
            break
        if i not in picked:
            picked.append(i)
    return sorted(picked[:N_COMPLETIONS_SHOWN])


def render_unit(r: dict, comps: dict, kind: str) -> list[str]:
    bank_model = MODEL_NAME[r["bank_run"]]
    graded_model = MODEL_NAME[r["graded_run"]]
    self_rate = r["stored_k_baseline"] / r["stored_n_baseline"]
    cross_rate = r["k"] / r["n"]
    lines = [
        f"## [{kind}] {r['direction']} / {r['prompt_id']} — "
        f"{bank_model} {r['stored_k_baseline']}/30 vs {graded_model} {r['k']}/30 "
        f"(Δ = {(cross_rate - self_rate) * 100:+.0f}pp)",
        "",
        f"**Classifier question ({bank_model} bank).** {r['behavior_question']}",
        "",
        "**Conversation.**",
        "",
    ]
    for m in comps[r["bank_run"]][r["prompt_id"]]["messages"]:
        lines += [f"[{m['role'].upper()}]:", "", m["content"], ""]

    graded = comps[r["graded_run"]][r["prompt_id"]]
    lines += [f"### {graded_model} completions (cross-graded, {r['k']}/30 yes)", ""]
    for i in pick_indices(r["verdicts"]):
        verdict = "YES" if r["verdicts"][i] else "NO"
        fr = graded["finish_reasons"][i]
        lines += [f"**Completion {i + 1}/30 — grader verdict: {verdict} (finish_reason={fr})**", "",
                  graded["completions"][i], ""]

    own = comps[r["bank_run"]][r["prompt_id"]]
    lines += [
        f"### {bank_model} completions (bank subject, stored rate "
        f"{r['stored_k_baseline']}/30; per-completion verdicts not stored)",
        "",
    ]
    for i in range(N_COMPLETIONS_SHOWN):
        fr = own["finish_reasons"][i]
        lines += [f"**Completion {i + 1}/30 (finish_reason={fr})**", "", own["completions"][i], ""]
    lines.append("---")
    return lines


def main() -> None:
    rows = [json.loads(line) for line in open(VERDICTS_PATH)]
    by_dir: dict[str, list[dict]] = {}
    for r in rows:
        if r["direction"] in CROSS_DIRECTIONS:
            by_dir.setdefault(r["direction"], []).append(r)

    selected: list[tuple[dict, str]] = []
    for d in CROSS_DIRECTIONS:
        ranked = sorted(
            by_dir[d],
            key=lambda r: abs(r["k"] / r["n"] - r["stored_k_baseline"] / r["stored_n_baseline"]),
            reverse=True,
        )
        selected += [(r, "DISAGREE") for r in ranked[:N_TOP_DISAGREE]]
        selected += [(r, "AGREE") for r in ranked[-N_AGREE:]]

    pids = {r["prompt_id"] for r, _ in selected}
    comps = load_completions(pids)

    lines = [
        "# Spot check: largest cross-model behavior disagreements (+ near-agreement controls)",
        "",
        f"Top {N_TOP_DISAGREE} |Δ| units and {N_AGREE} smallest-|Δ| units per direction.",
        "Full text throughout — review that the grader verdicts on the OTHER model's",
        "completions are sensible (not degenerate 'no' from formatting/reference mismatches).",
        "",
    ]
    for r, kind in selected:
        lines += render_unit(r, comps, kind)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH} ({len(selected)} units)")


if __name__ == "__main__":
    main()
