"""Shared MCQ helpers: task list, bank dir, and per-item prompt construction.

`build_task` turns each MCQ item into a record with a cue target + the rendered prompt for every
condition (neutral + each cue). `generate_until.py` is the entry point that calls this; there is no
standalone dataset-build step (generation owns the bank).
"""

import random
from dataclasses import asdict
from pathlib import Path

from chive import paths
from chive.hint_datasets.cot_hint.cues import CONDITION_BUILDERS, build_condition, letter
from chive.hint_datasets.cot_hint.datasets import load_task

TASKS = ["mmlu"]  # MMLU-only: GPQA/Law yield almost no inert claims, so a 3-task bank makes the
# supported/inert split a task-difficulty split (confound). MMLU-only keeps both cells same-distribution.
SETTING = "mcq_explain"  # the single MCQ setting (dir name == the unified `dataset` field)
BANKS_ROOT = paths.DATA_ROOT / "hint_banks" / "cot_hint"


def bank_dir() -> Path:
    """Where the genpool + emitted bank live: hint_banks/cot_hint/mcq_explain/."""
    return BANKS_ROOT / SETTING


def build_task(task: str, n: int, seed: int) -> list[dict]:
    items = load_task(task, n, seed)
    records = []
    for it in items:
        # Cue target = uniform random over ALL options (incl. gold), deterministic per item. Pointing
        # at gold sometimes decorrelates is_true from the model's competence on the item (cue→gold
        # supported = model was wrong; cue→wrong supported = model was right), so the cf eval can't be
        # solved by a difficulty prior — and it makes inert claims plentiful even for accurate models.
        rng = random.Random(f"cuetarget::{task}::{it.qid}")
        cue_idx = rng.randrange(len(it.options))
        prompts = {name: build_condition(name, it, cue_idx) for name in CONDITION_BUILDERS}
        records.append({
            **asdict(it),
            "gold_letter": letter(it.gold_idx),
            "cue_target_idx": cue_idx,
            "cue_target_letter": letter(cue_idx),
            "prompts": prompts,
        })
    return records
