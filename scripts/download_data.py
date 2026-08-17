"""Download the paper datasets from HuggingFace and materialize <repo>/data/.

Pulls the HF dataset repo (adamkarvonen/chive-data) via
snapshot_download and maps every file back into the local data/runs/ layout that
the pipeline scripts expect (internal run-dir names reconstructed from the public
datasets/ + prompt_sources/ folder names). The mapping is the exact inverse of
scripts/stage_hf_upload.py — both share the rule table in scripts/hf_layout.py.

The HF repo is DATASETS ONLY (investigation datasets + prompt pools). It does not
contain experiment results / binary_evals / analyses; the figure-reproduction data
for the paper is distributed separately (kept on the analysis machine), so this
script cannot materialize data/experiments/ etc.

Files are hardlinked out of the HF cache when possible (no extra disk), else copied.

Usage:
    python scripts/download_data.py                          # everything (~93 GB)
    python scripts/download_data.py --only prompt_sources    # just some top-level dirs
    python scripts/download_data.py --from-dir /path/to/snapshot --dry-run   # offline mapping test

--only accepts a comma-separated subset of:
    datasets, prompt_sources
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
    HF_DATASET_REPO,
    HF_TOP_LEVEL_DIRS,
    hf_to_data,
)


def iter_snapshot_files(snapshot_root: Path):
    """Yield snapshot-relative POSIX paths of regular files (skips .cache etc.)."""
    for p in sorted(snapshot_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(snapshot_root).as_posix()
        if rel.startswith(".cache/") or rel.startswith(".git/"):
            continue
        yield rel


def place(src: Path, dest: Path, overwrite: bool) -> str:
    """Hardlink src -> dest (fallback: copy). Returns 'link' | 'copy' | 'skip'."""
    if dest.exists():
        if not overwrite or dest.stat().st_ino == src.stat().st_ino:
            return "skip"
        dest.unlink()
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
    ap.add_argument(
        "--only",
        default=None,
        help=f"comma-separated top-level dirs to fetch (default: all). Choices: {', '.join(HF_TOP_LEVEL_DIRS)}",
    )
    ap.add_argument("--revision", default=None, help="HF revision (branch/tag/sha)")
    ap.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="use a local snapshot dir instead of downloading (offline / mapping test)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print src -> dest mapping, write nothing")
    ap.add_argument("--overwrite", action="store_true", help="replace existing files in data/ (default: skip)")
    args = ap.parse_args()

    only: list[str] | None = None
    if args.only:
        only = [t.strip() for t in args.only.split(",") if t.strip()]
        bad = [t for t in only if t not in HF_TOP_LEVEL_DIRS]
        if bad:
            ap.error(f"unknown --only values {bad}; choices: {', '.join(HF_TOP_LEVEL_DIRS)}")

    if args.from_dir is not None:
        snapshot_root = args.from_dir
    else:
        from huggingface_hub import snapshot_download

        allow_patterns = None if only is None else [f"{t}/**" for t in only]
        snapshot_root = Path(
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                revision=args.revision,
                allow_patterns=allow_patterns,
            )
        )
        print(f"snapshot at {snapshot_root}")

    counts = {"link": 0, "copy": 0, "skip": 0}
    n_meta = 0
    for rel in iter_snapshot_files(snapshot_root):
        if only is not None and rel.split("/", 1)[0] not in only:
            continue
        data_rel = hf_to_data(rel)
        if data_rel is None:  # release metadata: README.md / .gitattributes / MANIFEST.txt
            n_meta += 1
            continue
        src = snapshot_root / rel
        dest = args.data_root / data_rel
        if args.dry_run:
            print(f"{rel} -> data/{data_rel}")
            continue
        counts[place(src, dest, overwrite=args.overwrite)] += 1

    if args.dry_run:
        return
    print(f"materialized into {args.data_root}")
    print(
        f"  hardlinked {counts['link']}, copied {counts['copy']}, "
        f"already-present (skipped) {counts['skip']}, repo-metadata ignored {n_meta}"
    )


if __name__ == "__main__":
    main()
