"""Run the full CHIVE pipeline (stages 0-5) on an arbitrary prompt dataset.

Converts any .jsonl / .json / .csv file into the pipeline's prompt-pool schema
(see chive/pipeline/pool_common.py), writes a run config, and invokes
chive/pipeline/run_pipeline.py on it. Re-running the same command resumes.

Each input record is either
    {"messages": [{"role": "system"|"user"|"assistant", "content": ...}, ...]}
or a single prompt under one of TEXT_FIELDS (override with --text-field), plus
an optional "system" field. An "id" field is kept if present.

Optional "screening": a dict of prewritten stage-2 fields (behavior_summary,
question, second_person_question, classifier_question, hypotheses,
counterfactual). If every record has one, the stage-2 judge is skipped and these
are used as-is, so the grader's measurement is fixed by the researcher.

Usage:
    # offline smoke test, no keys needed
    .venv/bin/python introspection/run_on_dataset.py \
        --input introspection/dummy_prompts.jsonl --run-name dummy_smoke --mock

    # real run: role specs from flags or CHIVE_*_LLM in .env (see .env.example)
    .venv/bin/python introspection/run_on_dataset.py \
        --input path/to/introspection.jsonl --run-name introspection_v1 \
        --set n_completions=30 --set stage3_thorough=true
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chive import paths
from chive.pipeline.config import DEFAULTS
from chive.pipeline.pool_common import content_hash, read_jsonl, write_jsonl

# ---------------------------------------------------------------------------
# Environment. Put keys and role specs in REPO_ROOT/.env (copy .env.example);
# anything already exported in the shell wins.
# ---------------------------------------------------------------------------
load_dotenv(REPO_ROOT / ".env")

# role -> (config key, env var). Spec forms: anthropic:MODEL, openrouter:MODEL,
# vllm:MODEL@http://host:port, openai:MODEL@https://host/v1, mock:NAME.
ROLES = {
    "target": ("stage1_llm", "CHIVE_TARGET_LLM"),             # the model being investigated
    "investigator": ("stage3_llm", "CHIVE_INVESTIGATOR_LLM"),  # stage-3 agent
    "judge": ("judge_llm", "CHIVE_JUDGE_LLM"),                 # stage-2 screening + stage-4 verification
    "grader": ("grader_llm", "CHIVE_GRADER_LLM"),              # per-completion yes/no grader
    "synth": ("stage5_llm", "CHIVE_SYNTH_LLM"),                # stage-5 synthetic training data
}

TEXT_FIELDS = ("prompt", "user_message", "question", "text", "input")


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        assert isinstance(data, list), f"{path}: expected a JSON list of records"
        return data
    if path.suffix == ".csv":
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input format: {path.suffix} (use .jsonl, .json or .csv)")


def to_messages(rec: dict, text_field: str | None) -> list[dict]:
    if rec.get("messages"):
        messages = rec["messages"]
        if isinstance(messages, str):  # CSV cell holding JSON
            messages = json.loads(messages)
    else:
        fields = (text_field,) if text_field else TEXT_FIELDS
        text = next((rec[f] for f in fields if rec.get(f)), None)
        assert text, f"record has no 'messages' and none of {fields}: {str(rec)[:200]}"
        messages = [{"role": "user", "content": text}]
        if rec.get("system"):
            messages.insert(0, {"role": "system", "content": rec["system"]})
    messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    # The pipeline samples the assistant reply to the final user turn.
    while messages and messages[-1]["role"] != "user":
        messages.pop()
    assert messages, f"record has no user turn: {str(rec)[:200]}"
    return messages


def build_pool(records: list[dict], source: str, text_field: str | None) -> list[dict]:
    pool = []
    for i, rec in enumerate(records):
        messages = to_messages(rec, text_field)
        pool.append({
            "id": str(rec.get("id") or f"prompt_{i:06d}"),
            "source": source,
            "conversation_hash": content_hash(messages),
            "messages": messages,
            "user_message": messages[-1]["content"],
            "n_turns": sum(m["role"] == "user" for m in messages),
            "total_chars": sum(len(m["content"]) for m in messages),
        })
    ids = [r["id"] for r in pool]
    assert len(ids) == len(set(ids)), "duplicate ids in input"
    return pool


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, raw = item.partition("=")
        assert key in DEFAULTS, f"--set {key}: not a pipeline knob (see chive/pipeline/config.py)"
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help=".jsonl / .json / .csv prompt file")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, help=f"default: {paths.RUNS_DIR}/<run-name>")
    parser.add_argument("--text-field", help=f"prompt field for records without 'messages' (default: first of {TEXT_FIELDS})")
    parser.add_argument("--limit", type=int, help="only use the first N records")
    for role, (_, env) in ROLES.items():
        parser.add_argument(f"--{role}", help=f"backend spec (default: ${env})")
    parser.add_argument("--mock", action="store_true", help="use the offline mock backend for every role")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any pipeline knob, e.g. --set n_completions=30 (repeatable)")
    parser.add_argument("--fresh", action="store_true", help="delete this run dir first instead of resuming")
    args = parser.parse_args()

    run_dir = (args.run_dir or paths.RUNS_DIR / args.run_name).resolve()
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input)[: args.limit]
    pool = build_pool(records, source=args.input.stem, text_field=args.text_field)
    pool_path = run_dir / "pool.jsonl"
    write_jsonl(pool, pool_path)

    config = {
        "run_name": args.run_name,
        "prompt_pool": str(pool_path),
        "stage1_n_prompts": len(pool),
        "stage2_n": None,
        "concurrency": 20,
    }

    n_prewritten = sum(bool(r.get("screening")) for r in records)
    assert n_prewritten in (0, len(records)), (
        f"{n_prewritten}/{len(records)} records have a 'screening' field: give it to all or none")
    if n_prewritten:
        prewritten_path = run_dir / "screening_prewritten.jsonl"
        write_jsonl([{"prompt_id": p["id"], **r["screening"]} for p, r in zip(pool, records)], prewritten_path)
        config["screening_file"] = str(prewritten_path)
    missing = []
    for role, (key, env) in ROLES.items():
        spec = "mock:" + role if args.mock else getattr(args, role) or os.environ.get(env)
        if spec:
            config[key] = spec
        elif DEFAULTS[key] is None:
            missing.append(f"--{role} / {env}")
    # Counterfactual target defaults to the stage-1 model so originals and edits share one distribution.
    config["target_llm"] = config.get("stage1_llm")
    config.update(parse_set(args.set))
    if config.get("skip_stage5_synthetic") and "--synth / CHIVE_SYNTH_LLM" in missing:
        missing.remove("--synth / CHIVE_SYNTH_LLM")
    assert not missing, f"no backend spec for: {', '.join(missing)} (set them in .env or pass --mock)"

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(pool)} prompts + config -> {run_dir}")

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    cmd = [sys.executable, str(REPO_ROOT / "chive" / "pipeline" / "run_pipeline.py"), "--config", str(config_path)]
    sys.exit(subprocess.call(cmd, cwd=REPO_ROOT, env=env))


if __name__ == "__main__":
    main()
