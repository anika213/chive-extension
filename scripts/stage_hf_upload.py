"""(Re)build the local HF-upload staging tree from <repo>/data/.

Materializes the HF dataset repo layout (see scripts/hf_layout.py) under a staging
root using HARDLINKS (zero extra disk when data/ and the staging root share a
filesystem; falls back to a plain copy per-file when hardlinking fails).

The HF release is DATASETS ONLY: whitelisted files of the released investigation
runs plus the two prompt pools. Everything else under data/ (experiment results,
caches, non-whitelisted run files) is skipped — see hf_layout.py for the table.

Curated release metadata already present in the staging tree (README.md and the
per-folder MANIFEST.txt files, written once by the release build) is left in place;
this script never deletes anything.

This script only STAGES — it never talks to HuggingFace. The actual upload
(`hf upload-large-folder` / `HfApi.upload_folder`) is a separate, gated step.

Usage:
    python scripts/stage_hf_upload.py                       # data/ -> <repo>/hf_dataset/
    python scripts/stage_hf_upload.py --staging-root /path  # custom destination
    python scripts/stage_hf_upload.py --dry-run             # print the plan only
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_layout import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_STAGING_ROOT,
    data_to_hf,
    iter_data_files,
)


def place(src: Path, dest: Path) -> str:
    """Hardlink src -> dest (fallback: copy). Returns 'link' | 'copy' | 'skip'."""
    if dest.exists():
        if dest.stat().st_ino == src.stat().st_ino:
            return "skip"  # already the same hardlinked file
        dest.unlink()  # stale staging entry (staging tree is generated output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return "link"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    ap.add_argument("--dry-run", action="store_true", help="print the mapping, write nothing")
    args = ap.parse_args()

    counts = {"link": 0, "copy": 0, "skip": 0}
    n_not_released = 0
    bytes_by_top: dict[str, int] = {}

    for rel in iter_data_files(args.data_root):
        hf_rel = data_to_hf(rel)
        if hf_rel is None:
            n_not_released += 1
            continue
        src = args.data_root / rel
        dest = args.staging_root / hf_rel
        if args.dry_run:
            print(f"{rel} -> {hf_rel}")
            continue
        action = place(src, dest)
        counts[action] += 1
        top = hf_rel.split("/", 1)[0]
        bytes_by_top[top] = bytes_by_top.get(top, 0) + src.stat().st_size

    if args.dry_run:
        return
    print(f"staged into {args.staging_root}")
    print(f"  hardlinked {counts['link']}, copied {counts['copy']}, already-current {counts['skip']}")
    print(f"  not part of the dataset release (skipped): {n_not_released}")
    if counts["copy"]:
        print("  WARNING: some files were COPIED (hardlink failed — cross-filesystem?)")
    total = 0
    for top, b in sorted(bytes_by_top.items()):
        total += b
        print(f"  {top:<16} {b / 1e9:8.2f} GB")
    print(f"  {'TOTAL':<16} {total / 1e9:8.2f} GB (hardlinks: no extra disk used)")


if __name__ == "__main__":
    main()
