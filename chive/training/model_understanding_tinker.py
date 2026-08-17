"""Model-understanding SFT trainer that runs on the Tinker API.

Tinker attribution: every `tinker.ServiceClient(...)` call passes
`project_id=...` and `user_metadata={"owner": ...}`. Set the `TINKER_PROJECT_ID`
and `TINKER_OWNER` env vars (e.g. in `.env`) — the same convention as the
`tinker:` inference backend in chive/pipeline/llm_client.py — or set the
`project_id` / `owner` fields in the training config.

What this script does:
Reads one or more model-understanding pipeline run directories (each containing
`investigations.jsonl`, `screening.jsonl`, `verification.jsonl`), joins them by
prompt_id, and assembles a 4-turn SFT conversation per qualifying record:

    [system+user from investigations.messages]
    -> assistant: chosen completion from investigations.completions[
                      screening.behavior_completion_indices[0] - 1]
    -> user:     investigations.second_person_question
    -> assistant: investigations.structured_findings.structured_answer

Tokenizes with the project's Qwen chat template (`enable_thinking=False`),
masks loss to only the final assistant turn, and trains via Tinker's
`forward_backward` / `optim_step` API.

This mirrors `chive/training/model_understanding.py` but runs the actual
training on Tinker instead of local GPUs. Same data semantics, same loss
mask, same hyperparameter defaults as the existing local 32B run.

How to run
----------
Tinker runs server-side, so this script is just a CPU-only orchestrator. Run
it in tmux so it survives terminal disconnects, harness crashes, etc. If the
process dies, just re-run the same command — `resume.json` in `log_path`
makes it pick up at the last rolling save.

    tmux new -s tinker_train
    # inside tmux, from the repo root:
    source .venv/bin/activate && source .env
    python -m chive.training.model_understanding_tinker \
      --config data/training_configs/tinker/<your_config>.json

    # Ctrl-b d to detach. `tmux attach -t tinker_train` to reattach later.
    # `tmux capture-pane -t tinker_train -p | tail -30` to peek without attaching.

Add `--dry-run` to load + tokenize + print the pre-flight summary without
making any Tinker API calls.
"""

import argparse
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import tinker
import wandb
from transformers import AutoTokenizer

from chive.training import (
    ChatMessage,
    apply_chat_template_ids,
    render_chat_template,
)
from chive.training.utils.common import load_tokenizer, set_seed

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "model_understanding_tinker_config.json"
RESUME_FILENAME = "resume.json"
ROLLING_STATE_NAME = "rolling"  # rolling Tinker checkpoint slot — always overwritten
FINAL_STATE_NAME = "final"
# v2: read raw pipeline run-dirs (investigations + screening + verification join)
# instead of the now-removed structured_data.jsonl pre-generation step.
SCHEMA_VERSION = 2

# Tinker attribution defaults; a config's explicit `owner` / `project_id`
# fields take precedence. validate() fails loudly if neither is set.
OWNER_NAME = os.environ.get("TINKER_OWNER", "")
DEFAULT_PROJECT_ID = os.environ.get("TINKER_PROJECT_ID", "")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredExample:
    prompt_id: str
    interest_score: int
    verification_score: float
    messages: tuple[ChatMessage, ...]
    source: str = "investigation"  # one of: "investigation", "synthetic"


@dataclass(frozen=True)
class TokenizedExample:
    prompt_id: str
    input_ids: list[int]
    num_supervised_tokens: int
    prefix_len: int


@dataclass
class ModelUnderstandingTinkerConfig:
    # Data
    # One or more model-understanding pipeline run directories. Each must contain
    # `investigations.jsonl`, `screening.jsonl`, and `verification.jsonl`. The
    # 4-turn SFT conversation is built by joining those three files per prompt_id:
    #   - messages       <- investigations.jsonl[i]["messages"]
    #   - chosen_completion <- investigations.jsonl[i]["completions"][
    #                               screening.jsonl[i]["behavior_completion_indices"][0] - 1
    #                          ]    # 1-indexed → 0-indexed
    #   - question       <- investigations.jsonl[i]["second_person_question"]
    #   - structured_answer <- investigations.jsonl[i]["structured_findings"]["structured_answer"]
    #   - verification_score <- verification.jsonl[i]["score"]   (5-judge ensemble mean)
    # When more than one run_dir is given, prompt_ids are prefixed with the
    # run dir name to keep cross-run duplicates distinct.
    run_dirs: list[str]
    model_name: str  # HF tokenizer name (e.g. "Qwen/Qwen3.5-9B")
    base_model: str  # Tinker base model name (often same as model_name)
    log_path: str    # local dir for prepared debug dumps, config copy, wandb logs

    # When True, also load `<run_dir>/synthetic_data.jsonl` from each run_dir
    # and append the counterfactual_prediction examples to the training set.
    # We deliberately skip behavior_prediction entries — they use a synthetic
    # "[Stop. Do not respond to the user message above]" trigger format that
    # doesn't appear in real eval prompts, and they're ungrounded (the model
    # introspects about a hypothetical response with no actual completion to
    # reference). Same min_verification_score filter applies.
    include_synthetic: bool = False

    # Synthetic-only training (no freeform): one or more {metadata, results} JSON files in the
    # local trainer's synthetic_data_paths format (from balance_and_export.py). Mirrors
    # model_understanding.py: counterfactual_prediction rows are read directly (messages +
    # chosen_completion + question + answer); no interest_score gate (the file is already
    # balanced/filtered upstream). run_dirs may be empty when this is set, giving a pure
    # binary-counterfactual-ceiling run identical to the local 8B setup.
    synthetic_data_paths: list[str] | None = None

    # Tinker / training
    lora_rank: int = 64
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 5e-5
    max_seq_len: int = 4096
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_steps: int | None = None
    # Pipelining depth: number of EXTRA batches (beyond the one being awaited)
    # to keep submitted to Tinker. 0 = old serialized behavior (submit-wait-
    # submit-wait); 1 = cookbook default (one batch always queued ahead, so
    # Tinker never sits idle while client logs/submits next). Higher values
    # add diminishing returns and more GPU-side queueing.
    submit_ahead: int = 1

    # Filtering (match the local SFT pipeline)
    min_interest_score: int = 3
    min_verification_score: int = 7
    max_train_examples: int | None = None

    # Tinker infrastructure
    project_id: str = DEFAULT_PROJECT_ID
    owner: str = OWNER_NAME
    base_url: str | None = None
    # Rolling resume checkpoint: every N steps, overwrites a single "rolling" slot
    # on Tinker (so storage stays bounded at one adapter). 0 = no rolling save
    # (final checkpoint only — no resume support).
    save_every: int = 50
    # TTL on the rolling slot. The slot is overwritten on every save, but a TTL
    # is a safety net if the local script dies for >ttl and is never restarted.
    # Default 1 day.
    rolling_ttl_seconds: int | None = 24 * 3600

    # Wandb
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # Human-readable name stamped onto the Tinker training run via user_metadata.
    # Shows up in `rest.list_training_runs()`, so you can find a run by name
    # without remembering its UUID. If not set, falls back to:
    #   wandb_run_name -> Path(log_path).name -> "unnamed_run"
    # Suggested convention: mirror the local checkpoints_text_sft naming
    # (e.g. "mu_qwen35_9b_structured_e3_kl0").
    run_name: str | None = None

    # Optional path to a local trainer's saved training_config.json (e.g. the
    # 32B run's checkpoints_text_sft/.../final/training_config.json). When set,
    # the pre-flight summary prints a side-by-side diff of overlapping
    # hyperparameter fields so it's obvious whether the Tinker run matches the
    # local reference. Used for assurance only; does not affect training.
    reference_local_config_path: str | None = None

    # Misc
    seed: int = 42
    debug_num_examples_to_dump: int = 2
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        assert self.schema_version == SCHEMA_VERSION, (
            f"Unsupported schema_version={self.schema_version}; expected {SCHEMA_VERSION}"
        )
        assert self.run_dirs or self.synthetic_data_paths, (
            "must set run_dirs (freeform) and/or synthetic_data_paths (synthetic-only)"
        )
        for d in self.run_dirs:
            run_path = Path(d)
            assert run_path.exists(), f"run_dirs entry does not exist: {d}"
            required_files = ["investigations.jsonl", "screening.jsonl", "verification.jsonl"]
            if self.include_synthetic:
                required_files.append("synthetic_data.jsonl")
            for required in required_files:
                assert (run_path / required).exists(), (
                    f"run_dirs entry {d} is missing required file: {required}"
                )
        for p in self.synthetic_data_paths or []:
            assert Path(p).exists(), f"synthetic_data_paths entry does not exist: {p}"
        assert self.batch_size > 0
        assert self.num_epochs > 0
        assert self.learning_rate > 0
        assert self.max_seq_len > 0
        assert self.lora_rank > 0
        assert 0.0 <= self.warmup_ratio < 1.0
        assert self.max_grad_norm > 0
        assert self.min_interest_score > 0
        assert self.min_verification_score > 0
        if self.max_steps is not None:
            assert self.max_steps > 0
        if self.max_train_examples is not None:
            assert self.max_train_examples > 0
        assert self.submit_ahead >= 0, "submit_ahead must be >= 0"
        assert self.owner, "owner is required (Tinker policy)"
        assert self.project_id, "project_id is required (Tinker policy)"


def read_config(path: str | Path) -> ModelUnderstandingTinkerConfig:
    payload = json.loads(Path(path).read_text())
    # Strip underscore-prefixed keys (e.g. "_comment") so configs can carry
    # human notes alongside the real fields.
    payload = {k: v for k, v in payload.items() if not k.startswith("_")}
    cfg = ModelUnderstandingTinkerConfig(**payload)
    cfg.validate()
    return cfg


def write_config(log_dir: str | Path, cfg: ModelUnderstandingTinkerConfig) -> None:
    save_path = Path(log_dir) / CONFIG_FILENAME
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(asdict(cfg), indent=2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _run_prefix(run_dir: str, all_run_dirs: list[str]) -> str:
    """Disambiguate prompt_ids across multiple run dirs. When only one run dir
    is in use, no prefix (keeps debug filenames clean). With multiple, prefix
    with the run dir's own name (e.g. 'qwen3_32b_run_2_train_')."""
    if len(all_run_dirs) <= 1:
        return ""
    return Path(run_dir).name + "_"


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_structured_examples(cfg: ModelUnderstandingTinkerConfig) -> list[StructuredExample]:
    """Join investigations + screening + verification per run dir, build 4-turn convos.

    Skip-and-warn convention (per AGENTS.md): malformed/incomplete entries are
    counted and skipped; we hard-fail only if the skip rate exceeds 1% of all
    investigations that pass the score thresholds — at that point, something is
    structurally wrong with the upstream pipeline.
    """
    examples: list[StructuredExample] = []
    n_qualifying_total = 0   # investigations that pass both score thresholds
    n_skipped_total = 0      # of those, dropped due to missing/malformed fields

    for run_dir_str in cfg.run_dirs:
        run_path = Path(run_dir_str)
        prefix = _run_prefix(run_dir_str, cfg.run_dirs)

        inv_records = _read_jsonl(run_path / "investigations.jsonl")
        scr_records = _read_jsonl(run_path / "screening.jsonl")
        ver_records = _read_jsonl(run_path / "verification.jsonl")

        scr_bci = {d["prompt_id"]: (d.get("behavior_completion_indices") or []) for d in scr_records}
        ver_score = {d["prompt_id"]: d["score"] for d in ver_records}

        n_inv = len(inv_records)
        n_qualifying = 0
        n_skipped_missing_sa = 0
        n_skipped_missing_bci = 0
        n_skipped_bad_msgs = 0
        n_skipped_bad_completion_idx = 0
        n_kept = 0

        for entry in inv_records:
            pid = entry["prompt_id"]
            interest_score = entry["interest_score"]
            verification_score = ver_score.get(pid)
            if verification_score is None:
                # No verification record exists for this investigation. Not a
                # filter — just not eligible for SFT.
                continue
            if interest_score < cfg.min_interest_score:
                continue
            if verification_score < cfg.min_verification_score:
                continue
            n_qualifying += 1

            sf = entry.get("structured_findings") or {}
            structured_answer = sf.get("structured_answer")
            if not structured_answer:
                n_skipped_missing_sa += 1
                continue

            raw_messages = entry.get("messages") or []
            question = entry.get("second_person_question") or ""
            if not raw_messages or not question or raw_messages[-1]["role"] != "user":
                n_skipped_bad_msgs += 1
                continue

            bci = scr_bci.get(pid) or []
            if not bci:
                n_skipped_missing_bci += 1
                continue
            # screening behavior_completion_indices are 1-based. Occasional
            # flaky LLM tool output emits string indices instead of ints;
            # skip those rather than crash the loader (AGENTS.md convention).
            if not isinstance(bci[0], int):
                n_skipped_bad_completion_idx += 1
                continue
            chosen_idx = bci[0] - 1
            completions = entry.get("completions") or []
            if not (0 <= chosen_idx < len(completions)):
                n_skipped_bad_completion_idx += 1
                continue
            chosen_completion = completions[chosen_idx]
            if not chosen_completion:
                n_skipped_bad_completion_idx += 1
                continue

            messages = [ChatMessage(role=m["role"], content=m["content"]) for m in raw_messages]
            messages.append(ChatMessage(role="assistant", content=chosen_completion))
            messages.append(ChatMessage(role="user", content=question))
            messages.append(ChatMessage(role="assistant", content=structured_answer))

            examples.append(
                StructuredExample(
                    prompt_id=prefix + pid,
                    interest_score=interest_score,
                    verification_score=verification_score,
                    messages=tuple(messages),
                    source="investigation",
                )
            )
            n_kept += 1

        n_skipped = (
            n_skipped_missing_sa + n_skipped_missing_bci
            + n_skipped_bad_msgs + n_skipped_bad_completion_idx
        )
        skip_rate = n_skipped / max(n_qualifying, 1)
        if n_skipped:
            logger.warning(
                "%s: dropped %d/%d qualifying investigations (%.2f%%) "
                "[missing_structured_answer=%d, missing_behavior_completion_indices=%d, "
                "bad_messages=%d, bad_completion_idx=%d]",
                run_path, n_skipped, n_qualifying, 100 * skip_rate,
                n_skipped_missing_sa, n_skipped_missing_bci,
                n_skipped_bad_msgs, n_skipped_bad_completion_idx,
            )
        assert skip_rate < 0.01, (
            f"Too many skipped qualifying investigations in {run_path}: "
            f"{n_skipped}/{n_qualifying} ({skip_rate:.1%}). Inspect the upstream pipeline."
        )

        logger.info(
            "Loaded %d examples from %s (investigations=%d, qualifying_after_scores=%d, kept=%d, skipped=%d)",
            n_kept, run_path, n_inv, n_qualifying, n_kept, n_skipped,
        )
        n_qualifying_total += n_qualifying
        n_skipped_total += n_skipped

        if cfg.include_synthetic:
            _, n_syn_qualifying, n_syn_skipped = _load_synthetic_examples(
                run_path, prefix, cfg, examples
            )
            n_qualifying_total += n_syn_qualifying
            n_skipped_total += n_syn_skipped

    for syn_path in cfg.synthetic_data_paths or []:
        n_syn = _load_synthetic_path_examples(Path(syn_path), cfg, examples)
        logger.info("Loaded %d counterfactual_prediction examples from %s", n_syn, syn_path)

    logger.info(
        "TOTAL across %d run dir(s) + %d synthetic file(s): %d examples "
        "(qualifying_after_scores=%d, skipped=%d)",
        len(cfg.run_dirs), len(cfg.synthetic_data_paths or []), len(examples),
        n_qualifying_total, n_skipped_total,
    )
    assert examples, "No examples survived filtering"
    return examples


def _load_synthetic_path_examples(
    path: Path,
    cfg: ModelUnderstandingTinkerConfig,
    examples: list[StructuredExample],
) -> int:
    """Append counterfactual_prediction examples from a {metadata, results} synthetic JSON
    (balance_and_export.py output / the local trainer's synthetic_data_paths format). Mirrors
    model_understanding.load_synthetic_data: the 4-turn conversation is messages -> chosen_completion
    -> question -> answer, read directly. No interest_score gate (the file is already balanced and
    filtered upstream); the verification_score floor still applies. Returns n_kept."""
    data = json.loads(path.read_text())
    n_kept = 0
    for entry in data["results"]:
        if entry["example_type"] != "counterfactual_prediction":
            continue
        if entry["verification_score"] < cfg.min_verification_score:
            continue
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in entry["messages"]]
        messages.append(ChatMessage(role="assistant", content=entry["chosen_completion"]))
        messages.append(ChatMessage(role="user", content=entry["question"]))
        messages.append(ChatMessage(role="assistant", content=entry["answer"]))
        examples.append(
            StructuredExample(
                prompt_id=f"synth_{entry['prompt_id']}__cf_{entry.get('experiment_index', 0)}",
                interest_score=cfg.min_interest_score,  # not gated; file is pre-filtered
                verification_score=entry["verification_score"],
                messages=tuple(messages),
                source="synthetic_path",
            )
        )
        n_kept += 1
    return n_kept


def _load_synthetic_examples(
    run_path: Path,
    prefix: str,
    cfg: ModelUnderstandingTinkerConfig,
    examples: list[StructuredExample],
) -> tuple[int, int, int]:
    """Append synthetic_data.jsonl counterfactual_prediction examples from
    `run_path` to `examples`. Returns (n_kept, n_qualifying, n_skipped).

    behavior_prediction entries in synthetic_data.jsonl are intentionally
    skipped (not counted in qualifying or skipped) — their synthetic
    "[Stop. Do not respond to the user message above]" trigger format doesn't
    appear in real eval prompts and they're ungrounded.

    Counterfactual format is the same 4-turn structure as investigations:
        user(prompt) -> assistant(chosen_completion)
        -> user(question) -> assistant(answer)
    """
    syn_path = run_path / "synthetic_data.jsonl"
    assert syn_path.exists(), (
        f"include_synthetic=True but synthetic_data.jsonl is missing: {syn_path}"
    )

    n_qualifying = 0
    n_kept = 0
    n_skipped_bad_msgs = 0
    n_skipped_missing_text = 0
    n_skipped_missing_chosen = 0
    n_dropped_behavior_prediction = 0

    for entry in _read_jsonl(syn_path):
        if entry["example_type"] != "counterfactual_prediction":
            n_dropped_behavior_prediction += 1
            continue
        interest_score = entry["interest_score"]
        verification_score = entry["verification_score"]
        if interest_score < cfg.min_interest_score:
            continue
        if verification_score < cfg.min_verification_score:
            continue
        n_qualifying += 1

        msgs = entry.get("messages") or []
        question = entry.get("question") or ""
        answer = entry.get("answer") or ""
        if not msgs or msgs[-1]["role"] != "user":
            n_skipped_bad_msgs += 1
            continue
        if not question or not answer:
            n_skipped_missing_text += 1
            continue
        chosen = entry.get("chosen_completion")
        if not chosen:
            n_skipped_missing_chosen += 1
            continue

        custom_id = entry.get("_custom_id") or (
            f"{entry['prompt_id']}__cf_{entry.get('experiment_index', 0)}"
        )

        new_msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in msgs]
        new_msgs.append(ChatMessage(role="assistant", content=chosen))
        new_msgs.append(ChatMessage(role="user", content=question))
        new_msgs.append(ChatMessage(role="assistant", content=answer))

        examples.append(
            StructuredExample(
                prompt_id=prefix + custom_id,
                interest_score=interest_score,
                verification_score=verification_score,
                messages=tuple(new_msgs),
                source="synthetic",
            )
        )
        n_kept += 1

    n_skipped = n_skipped_bad_msgs + n_skipped_missing_text + n_skipped_missing_chosen
    skip_rate = n_skipped / max(n_qualifying, 1)
    if n_skipped:
        logger.warning(
            "%s: dropped %d/%d qualifying synthetic counterfactual examples (%.2f%%) "
            "[bad_messages=%d, missing_question_or_answer=%d, missing_chosen_completion=%d]",
            syn_path, n_skipped, n_qualifying, 100 * skip_rate,
            n_skipped_bad_msgs, n_skipped_missing_text, n_skipped_missing_chosen,
        )
    assert skip_rate < 0.01, (
        f"Too many skipped qualifying synthetic counterfactual examples in {syn_path}: "
        f"{n_skipped}/{n_qualifying} ({skip_rate:.1%}). Inspect the upstream pipeline."
    )
    logger.info(
        "Loaded %d synthetic counterfactual examples from %s "
        "(qualifying=%d, kept=%d, skipped=%d, behavior_prediction_filtered_out=%d)",
        n_kept, syn_path, n_qualifying, n_kept, n_skipped,
        n_dropped_behavior_prediction,
    )
    return n_kept, n_qualifying, n_skipped


# ---------------------------------------------------------------------------
# Tokenization (loss only on the final assistant turn)
# ---------------------------------------------------------------------------


def tokenize_example(
    example: StructuredExample,
    tokenizer: AutoTokenizer,
    cfg: ModelUnderstandingTinkerConfig,
) -> TokenizedExample | None:
    """Tokenize one 4-turn example. Returns None if it cannot fit in max_seq_len."""
    prefix_messages = example.messages[:-1]
    prefix_ids = apply_chat_template_ids(
        tokenizer, prefix_messages, add_generation_prompt=True
    )
    full_ids = apply_chat_template_ids(
        tokenizer, example.messages, add_generation_prompt=False
    )
    assert full_ids[: len(prefix_ids)] == prefix_ids, (
        f"Prefix/full tokenization mismatch for {example.prompt_id}: "
        f"len(prefix)={len(prefix_ids)} len(full)={len(full_ids)}"
    )
    if len(full_ids) > cfg.max_seq_len:
        return None
    num_supervised = len(full_ids) - len(prefix_ids)
    assert num_supervised > 0, f"No supervised tokens for {example.prompt_id}"
    return TokenizedExample(
        prompt_id=example.prompt_id,
        input_ids=full_ids,
        num_supervised_tokens=num_supervised,
        prefix_len=len(prefix_ids),
    )


def write_debug_example(
    cfg: ModelUnderstandingTinkerConfig,
    raw_example: StructuredExample,
    tok_example: TokenizedExample,
    tokenizer: AutoTokenizer,
    debug_index: int,
) -> None:
    debug_dir = Path(cfg.log_path) / "prepare_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{debug_index:02d}_{raw_example.prompt_id}.txt"

    lines: list[str] = []
    lines.append(f"prompt_id: {raw_example.prompt_id}")
    lines.append(f"interest_score: {raw_example.interest_score}")
    lines.append(f"verification_score: {raw_example.verification_score}")
    lines.append(f"prefix_len: {tok_example.prefix_len}")
    lines.append(f"num_supervised_tokens: {tok_example.num_supervised_tokens}")
    lines.append("")
    lines.append("MESSAGES")
    lines.append("--------")
    for i, msg in enumerate(raw_example.messages):
        lines.append(f"[{i}] role={msg.role}")
        lines.append(msg.content)
        lines.append("")

    lines.append("RENDERED CHAT TEMPLATE")
    lines.append("----------------------")
    lines.append(render_chat_template(tokenizer, raw_example.messages))
    lines.append("")
    lines.append("TOKEN TABLE")
    lines.append("-----------")
    for idx, tid in enumerate(tok_example.input_ids):
        tok_text = tokenizer.decode([tid], skip_special_tokens=False)
        loss_flag = 1 if idx >= tok_example.prefix_len else 0
        lines.append(f"{idx:05d} loss={loss_flag} token_id={tid:>8} token={tok_text!r}")

    path.write_text("\n".join(lines))
    logger.info("Wrote debug tokenization to %s", path)


def tokenize_all(
    cfg: ModelUnderstandingTinkerConfig,
    examples: list[StructuredExample],
    tokenizer: AutoTokenizer,
) -> list[TokenizedExample]:
    tokenized: list[TokenizedExample] = []
    n_dropped_too_long = 0
    n_dropped_tok_error = 0
    n_debug_dumped = 0

    for ex in examples:
        try:
            tok = tokenize_example(ex, tokenizer, cfg)
        except AssertionError:
            # Rare cases where the chat template's tokenization of prefix vs full
            # diverges (e.g. content contains literal think tags). Match the local
            # model_understanding.py behavior: count and skip, hard-fail if too many.
            n_dropped_tok_error += 1
            continue
        if tok is None:
            n_dropped_too_long += 1
            continue
        tokenized.append(tok)
        if n_debug_dumped < cfg.debug_num_examples_to_dump:
            write_debug_example(cfg, ex, tok, tokenizer, n_debug_dumped)
            n_debug_dumped += 1

    max_allowed_errors = max(1, len(examples) // 5000)
    assert n_dropped_tok_error <= max_allowed_errors, (
        f"Too many tokenization errors: {n_dropped_tok_error}/{len(examples)} "
        f"(threshold: {max_allowed_errors})"
    )
    assert tokenized, "All examples were dropped during tokenization (max_seq_len too small?)"
    lengths = [t.prefix_len + t.num_supervised_tokens for t in tokenized]
    logger.info(
        "Tokenized %d examples (dropped_too_long=%d, dropped_tok_error=%d). "
        "seq_len min=%d median=%.0f max=%d",
        len(tokenized), n_dropped_too_long, n_dropped_tok_error,
        min(lengths), median(lengths), max(lengths),
    )
    return tokenized


# ---------------------------------------------------------------------------
# Convert to tinker.Datum
# ---------------------------------------------------------------------------


def example_to_datum(tok: TokenizedExample) -> tinker.Datum:
    """Build a tinker.Datum with input tokens, target tokens, and per-token weights.

    Next-token-prediction layout:
        input_tokens  = full_ids[:-1]           (the model sees these positions)
        target_tokens = full_ids[1:]            (what should come next at each position)
        weights       = 1 / N where target falls inside the supervised span else 0,
                        with N = num_supervised_tokens; this gives a per-example mean loss.

    The supervised span runs over indices [prefix_len, len(full_ids)) in the original
    sequence. After the left-shift on targets (drop index 0), that span lands at
    indices [prefix_len - 1, len(full_ids) - 1) in the target sequence.
    """
    full_ids = tok.input_ids
    seq_len = len(full_ids)
    assert seq_len >= 2, f"Sequence too short for next-token training: {seq_len}"

    input_tokens = full_ids[:-1]
    target_tokens = full_ids[1:]
    n_tgt = len(target_tokens)

    weights = np.zeros(n_tgt, dtype=np.float32)
    # Target index i corresponds to predicting original index i+1.
    # Supervised positions in the original sequence are i+1 >= prefix_len  =>  i >= prefix_len - 1.
    start = max(0, tok.prefix_len - 1)
    end = n_tgt  # original index n_tgt corresponds to full_ids[n_tgt] which is the last token
    n_supervised = end - start
    assert n_supervised == tok.num_supervised_tokens, (
        f"Supervised token count mismatch for {tok.prompt_id}: "
        f"slice={n_supervised} expected={tok.num_supervised_tokens}"
    )
    weights[start:end] = 1.0 / float(n_supervised)

    model_input = tinker.ModelInput.from_ints(input_tokens)
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(
                np.asarray(target_tokens, dtype=np.int32)
            ),
            "weights": tinker.TensorData.from_numpy(weights),
        },
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchMetrics:
    mean_nll: float            # weighted mean NLL (averaged across supervised tokens per example, then across batch)
    num_examples: int
    num_supervised_tokens: int  # actual count of positions with non-zero loss weight
    num_input_tokens: int       # total input tokens fed to the model in this batch


@dataclass(frozen=True)
class SourceFileFingerprint:
    path: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class ResumeFingerprint:
    run_dirs: tuple[str, ...]
    source_files: tuple[SourceFileFingerprint, ...]
    model_name: str
    base_model: str
    lora_rank: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    max_seq_len: int
    warmup_ratio: float
    max_grad_norm: float
    max_steps: int | None
    min_interest_score: int
    min_verification_score: int
    max_train_examples: int | None
    submit_ahead: int
    seed: int
    num_tokenized_examples: int
    n_full_batches_per_epoch: int
    total_steps: int


def compute_batch_metrics(
    fwd_bwd_output: tinker.ForwardBackwardOutput,
    batch: list[tinker.Datum],
) -> BatchMetrics:
    total_weighted_logp = 0.0
    total_weight = 0.0
    total_input_tokens = 0
    total_supervised_positions = 0
    for datum, loss_out in zip(batch, fwd_bwd_output.loss_fn_outputs, strict=True):
        logprobs = loss_out["logprobs"].to_torch()
        weights = datum.loss_fn_inputs["weights"].to_torch()
        total_weighted_logp += float(logprobs.dot(weights))
        total_weight += float(weights.sum())
        total_input_tokens += datum.model_input.length
        total_supervised_positions += int((weights > 0).sum().item())
    mean_nll = float("nan") if total_weight == 0 else -total_weighted_logp / total_weight
    return BatchMetrics(
        mean_nll=mean_nll,
        num_examples=len(batch),
        num_supervised_tokens=total_supervised_positions,
        num_input_tokens=total_input_tokens,
    )


@dataclass(frozen=True)
class ResumeState:
    state_path: str
    global_step: int        # number of completed optim_steps
    epoch: int              # epoch the next batch belongs to
    batch_idx_in_epoch: int # next batch to run within `epoch`
    total_steps: int        # captured from the original schedule; resume must match
    fingerprint: ResumeFingerprint


def read_resume_state(log_dir: Path) -> ResumeState | None:
    path = log_dir / RESUME_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    fingerprint_payload = payload["fingerprint"]
    source_files = tuple(
        SourceFileFingerprint(**source_file)
        for source_file in fingerprint_payload["source_files"]
    )
    fingerprint = ResumeFingerprint(
        **{
            **fingerprint_payload,
            "run_dirs": tuple(fingerprint_payload["run_dirs"]),
            "source_files": source_files,
        }
    )
    return ResumeState(
        state_path=payload["state_path"],
        global_step=payload["global_step"],
        epoch=payload["epoch"],
        batch_idx_in_epoch=payload["batch_idx_in_epoch"],
        total_steps=payload["total_steps"],
        fingerprint=fingerprint,
    )


def write_resume_state(log_dir: Path, state: ResumeState) -> None:
    path = log_dir / RESUME_FILENAME
    # Write to tmp + rename so the file is never half-written if we crash mid-write.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(path)


def shuffle_indices_for_epoch(
    num_examples: int, seed: int, epoch: int
) -> list[int]:
    """Per-epoch deterministic shuffle so resume can reproduce the exact same
    batch ordering given only (seed, epoch)."""
    rng = random.Random(seed * 1_000_003 + epoch)
    indices = list(range(num_examples))
    rng.shuffle(indices)
    return indices


def build_resume_fingerprint(
    cfg: ModelUnderstandingTinkerConfig,
    tokenized: list[TokenizedExample],
    n_full_batches_per_epoch: int,
    total_steps: int,
) -> ResumeFingerprint:
    source_files = []
    for run_dir in cfg.run_dirs:
        run_path = Path(run_dir)
        filenames = ["investigations.jsonl", "screening.jsonl", "verification.jsonl"]
        if cfg.include_synthetic:
            filenames.append("synthetic_data.jsonl")
        for filename in filenames:
            path = run_path / filename
            stat = path.stat()
            source_files.append(
                SourceFileFingerprint(
                    path=str(path.resolve()),
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
    for syn_path in cfg.synthetic_data_paths or []:
        stat = Path(syn_path).stat()
        source_files.append(
            SourceFileFingerprint(
                path=str(Path(syn_path).resolve()),
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return ResumeFingerprint(
        run_dirs=tuple(str(Path(p).resolve()) for p in cfg.run_dirs),
        source_files=tuple(source_files),
        model_name=cfg.model_name,
        base_model=cfg.base_model,
        lora_rank=cfg.lora_rank,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        max_seq_len=cfg.max_seq_len,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        max_steps=cfg.max_steps,
        min_interest_score=cfg.min_interest_score,
        min_verification_score=cfg.min_verification_score,
        max_train_examples=cfg.max_train_examples,
        submit_ahead=cfg.submit_ahead,
        seed=cfg.seed,
        num_tokenized_examples=len(tokenized),
        n_full_batches_per_epoch=n_full_batches_per_epoch,
        total_steps=total_steps,
    )


# ---------------------------------------------------------------------------
# Pre-flight summary — printed before any Tinker spend so the user can sanity-
# check what's about to run. Includes a config diff vs a reference local
# training_config.json when one is supplied.
# ---------------------------------------------------------------------------


# Fields compared between the Tinker config and a reference local
# trainer's training_config.json. Tinker fields are the keys; local fields are
# the values. Names match where they line up, differ where they don't.
_TINKER_TO_LOCAL_FIELD_MAP = {
    "min_interest_score": "min_interest_score",
    "min_verification_score": "min_verification_score",
    "batch_size": "global_train_batch_size",
    "num_epochs": "num_epochs",
    "learning_rate": "lr",
    "max_seq_len": "max_seq_len",
    "warmup_ratio": "warmup_ratio",
    "max_grad_norm": "max_grad_norm",
    "lora_rank": "lora_r",
    "seed": "seed",
}


def print_preflight_summary(
    cfg: ModelUnderstandingTinkerConfig,
    tokenized: list[TokenizedExample],
    n_full_batches_per_epoch: int,
    total_steps: int,
) -> None:
    """Print a comprehensive pre-flight summary so the user can sanity-check
    the run before any Tinker compute is spent."""
    lengths = [t.prefix_len + t.num_supervised_tokens for t in tokenized]
    input_tok_per_epoch = sum(L - 1 for L in lengths)  # input == full_ids[:-1]
    supervised_tok_per_epoch = sum(t.num_supervised_tokens for t in tokenized)
    total_input_tok = input_tok_per_epoch * cfg.num_epochs
    total_supervised_tok = supervised_tok_per_epoch * cfg.num_epochs

    bar = "=" * 78
    lines: list[str] = []
    lines.append(bar)
    lines.append("TINKER SFT PRE-FLIGHT SUMMARY")
    lines.append(bar)
    lines.append(f"run_name          : {cfg.run_name or '(derived)'}")
    lines.append(f"log_path          : {cfg.log_path}")
    lines.append(f"base_model        : {cfg.base_model}")
    lines.append(f"tokenizer         : {cfg.model_name}")
    lines.append(f"project_id        : {cfg.project_id}")
    lines.append(f"owner             : {cfg.owner}")
    lines.append("")
    lines.append("DATA SOURCES (run dirs — joined investigations + screening + verification)")
    lines.append("-" * 78)
    for p in cfg.run_dirs:
        lines.append(f"  {p}")
    lines.append(f"include_synthetic  : {cfg.include_synthetic}")
    lines.append(
        f"Filter: interest_score >= {cfg.min_interest_score} "
        f"AND verification_score >= {cfg.min_verification_score}"
    )
    lines.append(f"Examples after filter + tokenization: {len(tokenized):,}")
    if cfg.max_train_examples is not None:
        lines.append(f"  (capped by max_train_examples = {cfg.max_train_examples})")
    lines.append(f"Sequence length stats: min={min(lengths)}, "
                 f"median={int(sorted(lengths)[len(lengths)//2])}, "
                 f"max={max(lengths)}")
    lines.append("")
    lines.append("TOKEN VOLUME")
    lines.append("-" * 78)
    lines.append(f"Input tokens / epoch     : {input_tok_per_epoch:>14,}")
    lines.append(f"Supervised tokens / epoch: {supervised_tok_per_epoch:>14,}")
    lines.append(f"Input tokens TOTAL ({cfg.num_epochs} ep)  : {total_input_tok:>14,}")
    lines.append(f"Supervised TOTAL ({cfg.num_epochs} ep)    : {total_supervised_tok:>14,}")
    lines.append("")
    lines.append("TRAINING SCHEDULE")
    lines.append("-" * 78)
    lines.append(f"batch_size        : {cfg.batch_size}")
    lines.append(f"num_epochs        : {cfg.num_epochs}")
    lines.append(f"batches per epoch : {n_full_batches_per_epoch}")
    lines.append(f"total optim steps : {total_steps}"
                 + (f" (capped from full by max_steps={cfg.max_steps})"
                    if cfg.max_steps is not None and cfg.max_steps < n_full_batches_per_epoch * cfg.num_epochs
                    else ""))
    lines.append(f"learning_rate     : {cfg.learning_rate}")
    lines.append(f"warmup_ratio      : {cfg.warmup_ratio} "
                 f"(={max(1, int(cfg.warmup_ratio * total_steps))} warmup steps)")
    lines.append(f"lora_rank         : {cfg.lora_rank}")
    lines.append(f"max_grad_norm     : {cfg.max_grad_norm}")
    lines.append(f"max_seq_len       : {cfg.max_seq_len}")
    lines.append(f"save_every (steps): {cfg.save_every}  "
                 f"(~{(total_steps // max(1, cfg.save_every))} rolling saves)")
    lines.append(f"submit_ahead      : {cfg.submit_ahead}  "
                 f"(pipeline depth: up to {1 + cfg.submit_ahead} batches in flight)")

    # Side-by-side diff against a reference local training_config.json.
    if cfg.reference_local_config_path is not None:
        ref_path = Path(cfg.reference_local_config_path)
        lines.append("")
        lines.append(f"DIFF VS LOCAL REFERENCE  ({ref_path})")
        lines.append("-" * 78)
        if not ref_path.exists():
            lines.append(f"  REFERENCE FILE NOT FOUND — cannot diff")
        else:
            ref = json.loads(ref_path.read_text())
            mismatches: list[str] = []
            lines.append(f"  {'tinker_field':<22} {'tinker':<18} {'local':<18} {'local_field'}")
            for tk_field, lc_field in _TINKER_TO_LOCAL_FIELD_MAP.items():
                tk_val = getattr(cfg, tk_field, None)
                lc_val = ref.get(lc_field, "<missing>")
                marker = " " if tk_val == lc_val else "!"
                lines.append(
                    f" {marker}{tk_field:<22} {str(tk_val):<18} {str(lc_val):<18} {lc_field}"
                )
                if tk_val != lc_val:
                    mismatches.append(f"{tk_field}={tk_val} vs local {lc_field}={lc_val}")
            ref_data_dirs = ref.get("run_dirs") or []
            lines.append("")
            lines.append(f"  data sources:")
            lines.append(f"    tinker run_dirs ({len(cfg.run_dirs)}):")
            for p in cfg.run_dirs:
                lines.append(f"      {p}")
            lines.append(f"    local trainer's run_dirs ({len(ref_data_dirs)}):")
            for d in ref_data_dirs:
                lines.append(f"      {d}")
            lines.append("")
            if mismatches:
                lines.append(f"  ⚠️  {len(mismatches)} hyperparameter mismatch(es) — REVIEW above before launching")
            else:
                lines.append(f"  ✓  All compared hyperparameters match the local reference")

    lines.append(bar)
    print("\n".join(lines))


def linear_warmup_decay_lr(step: int, total_steps: int, peak_lr: float, warmup_ratio: float) -> float:
    """Linear warmup from 0 -> peak_lr over `warmup_ratio * total_steps`,
    then linear decay to 0 by total_steps."""
    if total_steps <= 0:
        return peak_lr
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * max(0.0, 1.0 - progress)


def train(cfg: ModelUnderstandingTinkerConfig) -> None:
    set_seed(cfg.seed)

    log_dir = Path(cfg.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_config(log_dir, cfg)

    tokenizer = load_tokenizer(cfg.model_name)

    raw_examples = load_structured_examples(cfg)
    tokenized = tokenize_all(cfg, raw_examples, tokenizer)
    if cfg.max_train_examples is not None and len(tokenized) > cfg.max_train_examples:
        rng = random.Random(cfg.seed)
        rng.shuffle(tokenized)
        tokenized = tokenized[: cfg.max_train_examples]
        logger.info("Subsampled to %d examples (max_train_examples)", len(tokenized))

    # Determine training schedule
    n_full_batches_per_epoch = len(tokenized) // cfg.batch_size
    n_dropped = len(tokenized) % cfg.batch_size
    if n_dropped:
        logger.info(
            "Dropping last %d examples per epoch to keep batch_size=%d uniform",
            n_dropped, cfg.batch_size,
        )
    total_steps = n_full_batches_per_epoch * cfg.num_epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    assert total_steps > 0, (
        f"No training steps scheduled: examples={len(tokenized)} "
        f"batch_size={cfg.batch_size} num_epochs={cfg.num_epochs}"
    )
    logger.info(
        "Schedule: batches_per_epoch=%d, num_epochs=%d, total_steps=%d",
        n_full_batches_per_epoch, cfg.num_epochs, total_steps,
    )
    resume_fingerprint = build_resume_fingerprint(
        cfg, tokenized, n_full_batches_per_epoch, total_steps
    )

    # Pre-flight summary printed before ANY Tinker compute is spent so the user
    # has one last chance to ctrl-C if anything looks wrong.
    print_preflight_summary(cfg, tokenized, n_full_batches_per_epoch, total_steps)

    # Resolve a stable, human-readable run name for Tinker user_metadata.
    run_name = cfg.run_name or cfg.wandb_run_name or Path(cfg.log_path).name or "unnamed_run"
    tinker_metadata = {"owner": cfg.owner, "run_name": run_name}

    # Tinker service client (Tinker policy: owner + project_id)
    service_client = tinker.ServiceClient(
        base_url=cfg.base_url,
        user_metadata=tinker_metadata,
        project_id=cfg.project_id,
    )

    # Resume or start fresh
    resume = read_resume_state(log_dir)
    if resume is not None:
        assert resume.fingerprint == resume_fingerprint, (
            "Resume mismatch: saved training fingerprint differs from the current config/data. "
            f"Refusing to resume — config changed since last run."
        )
        logger.info(
            "Resuming from %s at epoch=%d batch_in_epoch=%d global_step=%d (run_name=%s)",
            resume.state_path, resume.epoch, resume.batch_idx_in_epoch, resume.global_step, run_name,
        )
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume.state_path, user_metadata=tinker_metadata
        )
        start_epoch = resume.epoch
        start_batch_idx = resume.batch_idx_in_epoch
        global_step = resume.global_step
    else:
        training_client = service_client.create_lora_training_client(
            base_model=cfg.base_model,
            rank=cfg.lora_rank,
            seed=cfg.seed,
            user_metadata=tinker_metadata,
        )
        logger.info(
            "Created LoRA training client: base_model=%s rank=%d",
            cfg.base_model, cfg.lora_rank,
        )
        start_epoch = 0
        start_batch_idx = 0
        global_step = 0

    # Wandb — initialize after resume decision so we can `resume="allow"` an existing run
    wandb_run = None
    if cfg.wandb_project is not None:
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=asdict(cfg),
            dir=str(log_dir),
            id=cfg.wandb_run_name,  # stable id so resume continues the same run
            resume="allow",
        )

    # ---- Pipelined train loop ---------------------------------------------
    # Tinker's sync `forward_backward(...)` returns an APIFuture immediately
    # without blocking on the server. So if we submit batch N+1 BEFORE awaiting
    # batch N's .result(), the server has work queued and never sits idle while
    # the client logs metrics / preps the next slice / does network RTT for the
    # next submit. cfg.submit_ahead controls how many batches are kept queued
    # beyond the one currently being awaited (default 1, matching the cookbook).
    # Rolling saves drain everything in flight first so the saved server-side
    # state corresponds to a clean, deterministic point in the schedule.

    @dataclass(frozen=True)
    class _Submitted:
        fwd_bwd_future: Any
        optim_future: Any
        batch: list[tinker.Datum]
        epoch: int
        batch_idx: int
        step: int          # global_step value this batch will be assigned when drained
        lr: float
        t_submitted: float

    pending: deque[_Submitted] = deque()

    def submit(epoch_: int, batch_idx_: int, indices_: list[int], step_for_batch: int) -> _Submitted:
        batch_slice = indices_[batch_idx_ * cfg.batch_size : (batch_idx_ + 1) * cfg.batch_size]
        batch_examples = [tokenized[i] for i in batch_slice]
        batch_ = [example_to_datum(t) for t in batch_examples]
        lr_ = linear_warmup_decay_lr(
            step=step_for_batch,
            total_steps=total_steps,
            peak_lr=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
        )
        adam_params = tinker.AdamParams(
            learning_rate=lr_, beta1=0.9, beta2=0.999, eps=1e-8,
            weight_decay=0.0, grad_clip_norm=cfg.max_grad_norm,
        )
        fwd_bwd_future = training_client.forward_backward(batch_, loss_fn="cross_entropy")
        optim_future = training_client.optim_step(adam_params)
        return _Submitted(
            fwd_bwd_future=fwd_bwd_future, optim_future=optim_future,
            batch=batch_, epoch=epoch_, batch_idx=batch_idx_,
            step=step_for_batch, lr=lr_, t_submitted=time.time(),
        )

    def finish(sb: _Submitted) -> None:
        nonlocal global_step
        fwd_bwd_result = sb.fwd_bwd_future.result()
        optim_result = sb.optim_future.result()
        bm = compute_batch_metrics(fwd_bwd_result, sb.batch)
        metrics = {
            "epoch": sb.epoch,
            "global_step": sb.step,
            "lr": sb.lr,
            "train_mean_nll": bm.mean_nll,
            "num_examples": bm.num_examples,
            "num_supervised_tokens": bm.num_supervised_tokens,
            "num_input_tokens": bm.num_input_tokens,
            "step_seconds": time.time() - sb.t_submitted,
        }
        if optim_result.metrics:
            metrics.update(optim_result.metrics)
        logger.info(
            "epoch=%d step=%d/%d lr=%.2e nll=%.4f supervised=%d input_tokens=%d (%.1fs)",
            sb.epoch, sb.step + 1, total_steps, sb.lr, bm.mean_nll,
            bm.num_supervised_tokens, bm.num_input_tokens, metrics["step_seconds"],
        )
        if wandb_run is not None:
            wandb_run.log(metrics, step=sb.step)
        global_step += 1

    def maybe_save_rolling(after_sb: _Submitted) -> None:
        """Save the rolling resume checkpoint if `global_step` lands on a boundary.

        Drains any still-in-flight batches first so the server-side state we save
        matches exactly the global_step we record in resume.json (no out-of-order
        optim_steps queued past the save point). Saves are infrequent (every
        cfg.save_every steps), so the brief loss of pipelining is negligible."""
        if cfg.save_every <= 0 or global_step % cfg.save_every != 0:
            return
        while pending:
            finish(pending.popleft())
        save_resp = training_client.save_state(
            name=ROLLING_STATE_NAME, ttl_seconds=cfg.rolling_ttl_seconds, overwrite=True,
        ).result()
        # `after_sb.batch_idx + 1` is the NEXT batch to run within this epoch.
        next_epoch_ = after_sb.epoch
        next_batch_idx_ = after_sb.batch_idx + 1
        if next_batch_idx_ >= n_full_batches_per_epoch:
            next_epoch_ += 1
            next_batch_idx_ = 0
        write_resume_state(log_dir, ResumeState(
            state_path=save_resp.path, global_step=global_step,
            epoch=next_epoch_, batch_idx_in_epoch=next_batch_idx_,
            total_steps=total_steps, fingerprint=resume_fingerprint,
        ))
        logger.info(
            "Saved rolling state (overwrite): %s  -> next epoch=%d batch=%d",
            save_resp.path, next_epoch_, next_batch_idx_,
        )

    next_step_to_assign = global_step   # the step value the NEXT submit will own
    for epoch in range(start_epoch, cfg.num_epochs):
        indices = shuffle_indices_for_epoch(len(tokenized), cfg.seed, epoch)
        batch_start = start_batch_idx if epoch == start_epoch else 0
        for batch_idx in range(batch_start, n_full_batches_per_epoch):
            if cfg.max_steps is not None and next_step_to_assign >= cfg.max_steps:
                break
            sb = submit(epoch, batch_idx, indices, next_step_to_assign)
            next_step_to_assign += 1
            pending.append(sb)
            # Keep at most (1 + submit_ahead) batches in flight at once: drain
            # the OLDEST whenever we exceed submit_ahead pending after this
            # submission.
            if len(pending) > cfg.submit_ahead:
                drained = pending.popleft()
                finish(drained)
                maybe_save_rolling(drained)
        # New epoch reached cleanly; reset the resume start_batch_idx so the next
        # iteration's `batch_start` calc is 0 (we already used start_batch_idx for
        # `epoch == start_epoch`).
        start_batch_idx = 0
        if cfg.max_steps is not None and next_step_to_assign >= cfg.max_steps:
            break

    # Drain anything still in flight at end-of-training.
    while pending:
        drained = pending.popleft()
        finish(drained)
        maybe_save_rolling(drained)

    # Final checkpoint: state (resumable) + sampler weights (usable).
    state_result = training_client.save_state(
        name=FINAL_STATE_NAME, ttl_seconds=None
    ).result()
    weights_result = training_client.save_weights_for_sampler(
        name=FINAL_STATE_NAME, ttl_seconds=None
    ).result()
    logger.info("Saved final state: %s", state_result.path)
    logger.info("Saved final sampling weights: %s", weights_result.path)

    # Persist checkpoint metadata next to the config so we can fetch later.
    (log_dir / "final_checkpoint.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "base_model": cfg.base_model,
                "state_path": state_result.path,
                "sampling_weights_path": weights_result.path,
            },
            indent=2,
        )
    )

    # Clean up the local resume marker after final artifacts are safely written.
    # The remote rolling checkpoint is left to expire via rolling_ttl_seconds.
    resume_path = log_dir / RESUME_FILENAME
    if resume_path.exists():
        resume_path.unlink()

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Model-understanding SFT trainer (Tinker backend)")
    parser.add_argument("--config", type=str, required=True, help="Path to ModelUnderstandingTinkerConfig JSON")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data, tokenize, dump debug examples; skip Tinker API calls",
    )
    args = parser.parse_args()

    cfg = read_config(args.config)

    if args.dry_run:
        Path(cfg.log_path).mkdir(parents=True, exist_ok=True)
        write_config(cfg.log_path, cfg)
        tokenizer = load_tokenizer(cfg.model_name)
        raw = load_structured_examples(cfg)
        tokenized = tokenize_all(cfg, raw, tokenizer)
        # Compute the schedule the same way train() does so the preflight matches.
        if cfg.max_train_examples is not None and len(tokenized) > cfg.max_train_examples:
            rng = random.Random(cfg.seed)
            rng.shuffle(tokenized)
            tokenized = tokenized[: cfg.max_train_examples]
        n_full_batches_per_epoch = len(tokenized) // cfg.batch_size
        total_steps = n_full_batches_per_epoch * cfg.num_epochs
        if cfg.max_steps is not None:
            total_steps = min(total_steps, cfg.max_steps)
        print_preflight_summary(cfg, tokenized, n_full_batches_per_epoch, total_steps)
        # Build one batch worth of Datums to verify the tinker conversion works.
        sample = tokenized[: min(cfg.batch_size, len(tokenized))]
        datums = [example_to_datum(t) for t in sample]
        total_w = sum(float(d.loss_fn_inputs["weights"].to_torch().sum()) for d in datums)
        logger.info(
            "Dry run OK: %d datums built, total weight=%.4f (expected %d), max input length=%d",
            len(datums), total_w, len(datums),
            max(d.model_input.length for d in datums),
        )
        return

    train(cfg)


if __name__ == "__main__":
    main()
