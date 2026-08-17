"""Load and normalize MCQ tasks to a common format.

Each task is normalized to a list of `MCQItem`, so the cue-construction and eval
code is fully task-agnostic.
"""

import random
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class MCQItem:
    task: str
    qid: str
    question: str
    options: list[str]  # option texts, index 0..k-1
    gold_idx: int       # index of the correct option


def _load_mmlu(n: int, seed: int) -> list[MCQItem]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    items = []
    for i, row in enumerate(ds):
        items.append(MCQItem(
            task="mmlu", qid=f"{row['subject']}::{i}",
            question=row["question"].strip(),
            options=[c.strip() for c in row["choices"]],
            gold_idx=int(row["answer"]),
        ))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


LOADERS = {
    "mmlu": _load_mmlu,
}


def load_task(task: str, n: int, seed: int = 42) -> list[MCQItem]:
    # Returns up to n items (a task may have fewer than requested).
    items = LOADERS[task](n, seed)
    if len(items) < n:
        print(f"  {task}: only {len(items)} items available (< requested {n}); using all")
    print(f"  {task}: loaded {len(items)} items "
          f"(option counts: {sorted(set(len(it.options) for it in items))})")
    return items
