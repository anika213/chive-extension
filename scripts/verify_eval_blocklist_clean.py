"""Independent post-filter verification: prove data/ contains no blocklisted prompt.

Deliberately uses a DIFFERENT detection path than the filter script:
  1. raw-text scan: every prompt_XXXXXX-shaped token in every .jsonl/.json file is
     checked against the blocklist ids (pool-id list for the source pool file) —
     catches records the filter's identity-field logic might have missed;
  2. content-hash scan: every record with a "messages" field (any file, any
     namespace, including PETRI / hint banks / smoke tests / training data) is
     hashed with the audit normalization (also minus a trailing assistant turn)
     and checked against the blocklisted conversation hashes;
  3. transcripts: no blocklisted <prompt_id>_transcript.json file remains.

Whitelisted (documented provenance, expected to still mention blocklisted ids):
  eval_blocklist.json, eval_blocklist_filter_manifest.json, train_overlap_*.json,
  qwen_bank_train_text_overlap.json, *.meta.json (selfexpl cache exclusion lists).
Whitelisted files still get scan #2 (content must be clean even there).

Exit code 0 + "CLEAN" only if nothing is found.

Run under Slurm (CPU-only, streams ~40 GB):
    srun -p dev,overflow --qos=dev --cpus-per-task=4 --mem=16G --time=2:00:00 \
        --job-name=D_adamk_blverify .venv/bin/python scripts/verify_eval_blocklist_clean.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path("/workspace-vast/adamk/chive")
DATA = REPO / "data"
BLOCKLIST = DATA / "eval_blocklist.json"
POOL_FILE = DATA / "runs/wildchat_plus_agentic/prompts.jsonl"

ID_RE = re.compile(r"prompt_\d{6}")


def content_hash(messages: list) -> str:
    normalized = [(m["role"], m["content"]) for m in messages]
    blob = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def is_whitelisted(path: Path) -> bool:
    return (
        path.name in ("eval_blocklist.json", "eval_blocklist_filter_manifest.json",
                      "qwen_bank_train_text_overlap.json")
        or path.name.startswith("train_overlap_")
        or path.name.endswith(".meta.json")
    )


def iter_records(path: Path):
    if path.suffix == ".jsonl":
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    yield line, json.loads(line)
    else:
        text = path.read_text()
        obj = json.loads(text)
        recs = obj.get("results") if isinstance(obj, dict) else obj
        if isinstance(recs, list):
            for r in recs:
                if isinstance(r, dict):
                    yield None, r
        # raw-text scan handles the rest of the structure


def main() -> None:
    bl = json.loads(BLOCKLIST.read_text())
    run_ids = set(bl["prompt_ids"])
    pool_ids = set(bl["source_pool_ids"])
    hashes = set(bl["content_hashes"])

    id_hits: dict[str, int] = {}
    hash_hits: dict[str, int] = {}
    transcript_hits: list[str] = []
    n_files = n_records_hashed = 0

    for path in sorted(DATA.rglob("*")):
        if not path.is_file() or path.suffix not in (".jsonl", ".json"):
            continue
        rel = str(path.relative_to(DATA))
        if path.parent.name == "transcripts" and path.name.endswith("_transcript.json"):
            pid = path.name[: -len("_transcript.json")]
            if pid in run_ids:
                transcript_hits.append(rel)
            continue
        n_files += 1
        ids = pool_ids if path == POOL_FILE else run_ids
        whitelisted = is_whitelisted(path)

        # scan 1: raw text (skip for whitelisted provenance files)
        if not whitelisted:
            n = 0
            with open(path) as fh:
                for chunk_line in fh:
                    for tok in ID_RE.findall(chunk_line):
                        if tok in ids:
                            n += 1
            if n:
                id_hits[rel] = n

        # scan 2: content hashes of messages-bearing records (all files)
        try:
            for _, rec in iter_records(path):
                msgs = rec.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    continue
                n_records_hashed += 1
                h = content_hash(msgs)
                h2 = content_hash(msgs[:-1]) if msgs[-1].get("role") == "assistant" and len(msgs) > 1 else None
                if h in hashes or (h2 is not None and h2 in hashes):
                    hash_hits[rel] = hash_hits.get(rel, 0) + 1
        except json.JSONDecodeError:
            print(f"  [warn] unparseable, raw-scan only: {rel}")

    print(f"scanned {n_files} files, hashed {n_records_hashed} messages-bearing records")
    ok = not id_hits and not hash_hits and not transcript_hits
    if id_hits:
        print("ID HITS (blocklisted ids still referenced):")
        for k, v in sorted(id_hits.items()):
            print(f"  {v:>6}  {k}")
    if hash_hits:
        print("CONTENT-HASH HITS (blocklisted conversations still present):")
        for k, v in sorted(hash_hits.items()):
            print(f"  {v:>6}  {k}")
    if transcript_hits:
        print(f"TRANSCRIPTS REMAINING: {transcript_hits}")
    print("CLEAN" if ok else "DIRTY")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
