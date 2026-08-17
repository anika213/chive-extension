"""Shared record schema + JSONL helpers for the model-understanding prompt pool.

Single source of truth for the pool record shape and JSONL I/O used by both
`build_prompt_pool.py` (WildChat base) and `build_agentic_pool.py` (agentic
sources + final merge), so the schema and the reader can't drift between them.

Canonical pool record — one JSON object per line in `prompts.jsonl`:

    {
      "id":                "prompt_000123",   # assigned at (re)build time
      "source":            "wildchat" | "hermes" | "toolace" | "systemchat2",
      "conversation_hash": "<WildChat hash; '' for agentic rows>",
      "messages":          [{"role": ..., "content": ...}, ...],  # ends with user
      "user_message":      "<messages[-1].content>",
      "n_turns":           <int: number of user turns>,
      "total_chars":       <int: sum of message content lengths>,
    }
"""

import hashlib
import json
from pathlib import Path


def content_hash(messages: list[dict]) -> str:
    """Deterministic 16-hex hash of a record's messages array (for dedup)."""
    content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file, splitting ONLY on "\\n".

    str.splitlines() also breaks on other Unicode line separators (VT, FF,
    FS/GS/RS, NEL, U+2028, U+2029) that appear literally inside some prompt text
    and would split one JSON object across two "lines". json.dumps never emits a
    bare "\\n" inside a string, so newline-split is the correct delimiter.
    """
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write records as one JSON object per line (ensure_ascii=False)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records)} records -> {path}")
