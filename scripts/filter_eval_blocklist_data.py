"""One-time migration: remove every blocklisted (train/eval-leaked) prompt from data/.

Reads data/eval_blocklist.json (built by scripts/build_eval_blocklist.py) and rewrites,
in place (tmp file + atomic rename), every data file that contains records for a
blocklisted prompt:

  - every *.jsonl under data/ — record identity is prompt_id, else claim_id prefix
    (prompt_XXXXXX_cN -> prompt_XXXXXX), else id. Namespace safety: PETRI ids
    (petri_*) and hint-bank ids (mmlu_*/aita_*) can never match the blocklist.
  - data/runs/wildchat_plus_agentic/prompts.jsonl uses the source-pool id list
    (same prompt_XXXXXX format, DIFFERENT numbering than the run slice).
  - data/runs/*/transcripts/<prompt_id>_transcript.json — deleted when blocklisted.
  - training-data record containers (data/training_data/*.json and
    experiments/**/results/training_data/*.json, {"metadata":..., "results": [...]}):
    filtered by conversation content hash (train-side ids are a different namespace).
  - JSON files that are a bare list of claim-id strings (nla_failure id lists).

Deliberately untouched (recorded in the manifest):
  - provenance/metadata files: train_overlap_*.json, qwen_bank_train_text_overlap.json,
    *.meta.json, build_manifest.json, config.json, status.json, capture_*.json —
    these document runs/overlap as they happened.  NOTE status.json score
    distributions describe the PRE-filter run and are stale for funnel counts.
  - data/runs/smoke_test_* (dev fixtures; content-hash-verified clean, but their id
    namespace is unverified so id-filtering could false-positive) and data/scratch/.

Safety: refuses to modify any file that is not already present with identical size in
the backup tree (BACKUP_ROOT).

Run under Slurm (CPU-only, streams ~40 GB):
    srun -p dev,overflow --qos=dev --cpus-per-task=4 --mem=16G --time=2:00:00 \
        --job-name=D_adamk_blfilter .venv/bin/python scripts/filter_eval_blocklist_data.py
"""

import hashlib
import json
import os
import re
from pathlib import Path

REPO = Path("/workspace-vast/adamk/chive")
DATA = REPO / "data"
BACKUP_ROOT = Path("/workspace-vast/adamk/counterfactual-investigations_prefilter_backup_2026-07-28/data")
BLOCKLIST = DATA / "eval_blocklist.json"
MANIFEST_OUT = DATA / "eval_blocklist_filter_manifest.json"
POOL_FILE = DATA / "runs/wildchat_plus_agentic/prompts.jsonl"

ID_RE = re.compile(r"prompt_\d{6}")

SKIP_DIR_PARTS = ("smoke_test_qwen3_8b", "smoke_test_qwen3_8b_jsonl", "scratch")
PROVENANCE_BASENAMES = ("config.json", "status.json", "build_manifest.json")
PROVENANCE_PREFIXES = ("train_overlap_", "capture_")
PROVENANCE_EXACT = ("qwen_bank_train_text_overlap.json",)


def content_hash(messages: list) -> str:
    """Same normalization as the audit / build_eval_blocklist.py."""
    normalized = [(m["role"], m["content"]) for m in messages]
    blob = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def train_record_hash(messages: list) -> str:
    """Train-bank records append the assistant answer turn; hash the conversation only
    (identical to the audit's load_train_bank)."""
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    return content_hash(messages)


def record_pid(rec: dict, path: Path) -> str | None:
    if "prompt_id" in rec:
        return rec["prompt_id"]
    if "claim_id" in rec:
        return rec["claim_id"].rsplit("_c", 1)[0]
    if "id" in rec:
        return rec["id"]
    raise AssertionError(f"{path}: record has no prompt_id/claim_id/id — add a rule")


def assert_backed_up(path: Path) -> None:
    rel = path.relative_to(DATA)
    b = BACKUP_ROOT / rel
    if not b.exists() or b.stat().st_size != path.stat().st_size:
        raise AssertionError(f"refusing to modify {path}: no identical-size backup at {b}")


def atomic_rewrite(path: Path, lines_or_text) -> None:
    tmp = path.with_name(path.name + ".blfilter_tmp")
    with open(tmp, "w") as fh:
        if isinstance(lines_or_text, str):
            fh.write(lines_or_text)
        else:
            fh.writelines(lines_or_text)
    os.replace(tmp, path)


def is_provenance(path: Path) -> bool:
    return (
        path.name in PROVENANCE_BASENAMES
        or path.name in PROVENANCE_EXACT
        or any(path.name.startswith(p) for p in PROVENANCE_PREFIXES)
        or path.name.endswith(".meta.json")
    )


def filter_jsonl(path: Path, ids: set[str], stats: dict) -> None:
    """Stream-filter a JSONL file. Fast path: skip lines with no prompt_XXXXXX candidate."""
    kept: list[str] = []
    n = dropped = 0
    with open(path) as fh:
        for line in fh:
            n += 1
            if not line.strip():
                continue
            cands = set(ID_RE.findall(line))
            if cands & ids:
                pid = record_pid(json.loads(line), path)
                if pid in ids:
                    dropped += 1
                    continue
            kept.append(line)
    if dropped:
        assert_backed_up(path)
        atomic_rewrite(path, kept)
    stats[str(path.relative_to(REPO))] = {"n_records": n, "dropped": dropped}


def filter_train_json(path: Path, hashes: set[str], stats: dict) -> None:
    obj = json.loads(path.read_text())
    results = obj["results"]
    kept = [r for r in results if train_record_hash(r["messages"]) not in hashes]
    dropped = len(results) - len(kept)
    if dropped:
        assert_backed_up(path)
        obj["results"] = kept
        meta = obj.setdefault("metadata", {})
        meta["evalpool_blocklist_filter"] = {
            "date": "2026-07-28",
            "blocklist": str(BLOCKLIST.relative_to(REPO)),
            "n_dropped": dropped,
        }
        atomic_rewrite(path, json.dumps(obj, indent=1) + "\n")
    stats[str(path.relative_to(REPO))] = {"n_records": len(results), "dropped": dropped}


def filter_claimid_list_json(path: Path, ids: set[str], stats: dict) -> None:
    obj = json.loads(path.read_text())
    kept = [c for c in obj if c.rsplit("_c", 1)[0] not in ids]
    dropped = len(obj) - len(kept)
    if dropped:
        assert_backed_up(path)
        atomic_rewrite(path, json.dumps(kept, indent=1) + "\n")
    stats[str(path.relative_to(REPO))] = {"n_records": len(obj), "dropped": dropped}


def main() -> None:
    bl = json.loads(BLOCKLIST.read_text())
    run_ids = set(bl["prompt_ids"])
    pool_ids = set(bl["source_pool_ids"])
    hashes = set(bl["content_hashes"])
    print(f"blocklist: {len(run_ids)} run ids, {len(pool_ids)} pool ids, {len(hashes)} hashes")
    if not BACKUP_ROOT.is_dir():
        raise AssertionError(f"backup tree missing: {BACKUP_ROOT}")

    stats: dict = {}
    skipped: dict[str, list[str]] = {"skip_dir": [], "provenance": [], "no_rule": []}
    deleted_transcripts: list[str] = []

    for path in sorted(DATA.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(DATA))
        if path == BLOCKLIST or path == MANIFEST_OUT:
            continue
        if any(part in path.parts for part in SKIP_DIR_PARTS):
            skipped["skip_dir"].append(rel)
            continue

        if path.suffix == ".jsonl":
            filter_jsonl(path, pool_ids if path == POOL_FILE else run_ids, stats)
        elif path.parent.name == "transcripts" and path.name.endswith("_transcript.json"):
            pid = path.name[: -len("_transcript.json")]
            if pid in run_ids:
                assert_backed_up(path)
                path.unlink()
                deleted_transcripts.append(rel)
        elif path.suffix == ".json":
            if is_provenance(path):
                skipped["provenance"].append(rel)
                continue
            obj = json.loads(path.read_text())
            if isinstance(obj, dict) and isinstance(obj.get("results"), list) and "training_data" in path.parts:
                filter_train_json(path, hashes, stats)
            elif isinstance(obj, list) and obj and all(isinstance(c, str) and ID_RE.match(c) for c in obj):
                filter_claimid_list_json(path, run_ids, stats)
            else:
                skipped["no_rule"].append(rel)
        else:
            skipped["no_rule"].append(rel)

    touched = {k: v for k, v in stats.items() if v["dropped"]}
    total_dropped = sum(v["dropped"] for v in stats.values())
    manifest = {
        "date": "2026-07-28",
        "blocklist": {"n_run_ids": len(run_ids), "n_pool_ids": len(pool_ids)},
        "total_records_dropped": total_dropped,
        "n_files_scanned": len(stats),
        "n_files_modified": len(touched),
        "n_transcripts_deleted": len(deleted_transcripts),
        "deleted_transcripts": deleted_transcripts,
        "files": stats,
        "skipped": {k: sorted(v) for k, v in skipped.items()},
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"scanned {len(stats)} record files; modified {len(touched)}; "
          f"dropped {total_dropped} records; deleted {len(deleted_transcripts)} transcripts")
    print(f"skipped: { {k: len(v) for k, v in skipped.items()} }")
    print(f"wrote {MANIFEST_OUT}")
    for k, v in sorted(touched.items()):
        print(f"  {v['dropped']:>6} / {v['n_records']:>7}  {k}")


if __name__ == "__main__":
    main()
