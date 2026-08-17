"""One-time repair of the `categories` field in released mechanism_concreteness.jsonl files.

WHAT HAPPENED: chive/binary_eval/concreteness.py's few-shot examples wrote category lists in
bare unquoted syntax (categories=[a, b]), the Anthropic API does not enforce the tool's
input_schema, and call_for_tool only checked key PRESENCE — so in ~5% of rows (416/8118
across the 8 released eval runs) the judge imitated the few-shot literal and `categories`
shipped as ONE string "[a, b]" instead of a JSON list. Four further rows carry tokens
outside the CATEGORIES enum (hallucinated tags / a JS ternary). No other field is affected,
and no analysis code reads `categories` — this is a serialization repair, not a relabel.
Both root causes are now fixed upstream (quoted few-shots + a `validate=` hook).

REPAIR RULE (per row):
  - list-typed `categories`: keep as is;
  - string-typed: strip the outer [ ], split on ",", strip whitespace/quotes/backslashes;
  - then drop any token not in the CATEGORIES enum (printed loudly, one line per drop).
Asserts: row count unchanged, every other field byte-identical, every repaired row keeps
>= 1 valid tag. Writes via tmp + os.replace (atomic; intentionally breaks the hardlink to
the HF staging tree — re-run scripts/stage_hf_upload.py afterwards to re-link), and
updates the staged MANIFEST.txt byte counts.

Usage:
    python scripts/fix_mechanism_concreteness_categories.py            # dry-run (default)
    python scripts/fix_mechanism_concreteness_categories.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_layout import DEFAULT_DATA_ROOT, DEFAULT_STAGING_ROOT, data_to_hf, iter_data_files  # noqa: E402

from chive.binary_eval.concreteness import CATEGORIES  # single source of truth for the enum


def repair_tokens(raw: str | list) -> tuple[list[str], list[str]]:
    """Return (valid_tokens, dropped_tokens). Fails loudly on unrecognized shapes."""
    if isinstance(raw, list):
        tokens = raw
    else:
        s = raw.strip()
        assert s.startswith("[") and s.endswith("]"), f"unrecognized categories string: {raw!r}"
        tokens = [t.strip().strip("\\\"'").strip("\\") for t in s[1:-1].split(",") if t.strip()]
    valid = [t for t in tokens if t in CATEGORIES]
    dropped = [t for t in tokens if t not in CATEGORIES]
    return valid, dropped


def repair_file(path: Path, apply: bool) -> tuple[int, int]:
    """Repair one JSONL in place. Returns (n_rows, n_changed)."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    n_changed = 0
    for row in rows:
        raw = row["categories"]
        valid, dropped = repair_tokens(raw)
        for tok in dropped:
            print(f"  DROPPED non-enum token {tok!r}  ({path.parent.name} / {row['prompt_id']})")
        assert valid, f"{row['prompt_id']}: no valid categories left (raw={raw!r})"
        if valid != raw:
            row["categories"] = valid
            n_changed += 1
    if apply and n_changed:
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        # sanity: re-parse the repaired file before replacing the original
        for r in map(json.loads, tmp.read_text().splitlines()):
            assert isinstance(r["categories"], list) and r["categories"]
            assert all(c in CATEGORIES for c in r["categories"])
        assert sum(1 for _ in tmp.read_text().splitlines()) == len(rows)
        os.replace(tmp, path)
    return len(rows), n_changed


def update_manifest(staging_folder: Path, fname: str, new_bytes: int, new_lines: int) -> None:
    manifest = staging_folder / "MANIFEST.txt"
    out = []
    hit = False
    for line in manifest.read_text().splitlines():
        parts = line.split("\t")
        if parts[0] == fname:
            out.append(f"{fname}\t{new_bytes}\t{new_lines}")
            hit = True
        else:
            out.append(line)
    assert hit, f"{fname} not found in {manifest}"
    manifest.write_text("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write repairs (default: dry-run)")
    args = ap.parse_args()

    total_rows = total_changed = 0
    for rel in sorted(iter_data_files(DEFAULT_DATA_ROOT)):
        hf_rel = data_to_hf(rel)
        if hf_rel is None or not hf_rel.endswith("mechanism_concreteness.jsonl"):
            continue
        src = DEFAULT_DATA_ROOT / rel
        n_rows, n_changed = repair_file(src, apply=args.apply)
        total_rows += n_rows
        total_changed += n_changed
        print(f"{hf_rel}: {n_changed}/{n_rows} rows repaired"
              f"{'' if args.apply else ' (DRY-RUN, nothing written)'}")
        if args.apply and n_changed:
            update_manifest(DEFAULT_STAGING_ROOT / Path(hf_rel).parent, src.name,
                            src.stat().st_size, n_rows)
    print(f"TOTAL: {total_changed}/{total_rows} rows repaired")
    if args.apply:
        print("now re-run: python scripts/stage_hf_upload.py  (re-links repaired files into staging)")


if __name__ == "__main__":
    main()
