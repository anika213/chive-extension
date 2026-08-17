"""Shared mapping between the local ``data/`` tree and the HF dataset repo layout.

This module is the single source of truth for how files in ``<repo>/data/`` map onto
the HuggingFace dataset repo ``adamkarvonen/chive-data``, and back.
Both ``scripts/stage_hf_upload.py`` (data/ -> staging tree) and ``scripts/download_data.py``
(HF snapshot -> data/) use the same rule table, so the two directions cannot drift.

The HF release is DATASETS ONLY (investigation datasets + prompt pools). Experiment
results, binary_evals, caches, and analyses are not uploaded; figure-reproduction data
is distributed separately.

Layout summary (HF side):

    datasets/<public_name>/        one folder per released run, JSONL-only serialization.
      eval runs (n=30):            investigations/screening/verification/completions/
                                   prompts .jsonl, config.json,
                                   binary_counterfactual_questions.jsonl (+ .build_meta.json),
                                   eval_set_*.jsonl, mechanism_concreteness.jsonl
                                   (some runs lack some files; see each folder's MANIFEST.txt)
      train runs (n=10):           investigations/screening/verification .jsonl (converted
                                   once from the consolidated .json; *.conversion_note.txt
                                   records counts + source sha256), metadata.json (preserved
                                   consolidated-JSON headers), binary_counterfactual_questions
                                   .jsonl (+ .build_meta.json), provenance_config*.json
                                   (archived original pipeline run config(s))
    prompt_sources/<public_name>/  prompts.jsonl prompt pools

Public naming: ``{target}_{source}_{split}`` where ``_thinking`` marks TARGET reasoning
mode and ``self_investigate`` marks the target investigating itself (the investigator is
otherwise Claude Opus 4.6; recorded per-folder in config.json / provenance_config*.json).
Internal run-dir names (under data/runs/) are unchanged; renaming happens only on the HF
side via this table.

README.md and the per-folder MANIFEST.txt files are release metadata curated in the
staging tree (default <repo>/hf_dataset/, gitignored); they do not map back into data/.

Run ``python scripts/hf_layout.py --self-test`` to verify the mapping round-trips every
releasable file currently present under data/.
"""

from __future__ import annotations

from pathlib import Path

HF_DATASET_REPO = "adamkarvonen/chive-data"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_STAGING_ROOT = REPO_ROOT / "hf_dataset"

# Internal run dir (under data/runs/) -> public HF dataset name.
DATASET_SLUGS: dict[str, str] = {
    # eval runs (n=30, Opus 4.6 investigator)
    "opus_investigates_gemma3_27b_it_5000_thorough_n30": "gemma3_27b_it_wildchat_eval",
    "opus_investigates_qwen3_8b_5000_thorough_n30": "qwen3_8b_wildchat_eval",
    "opus_investigates_qwen3_8b_petri_thorough_n30": "qwen3_8b_petri_eval",
    "opus_investigates_qwen3p5_397b_5000_thorough_n30": "qwen3p5_397b_wildchat_eval",
    "opus_investigates_qwen3p5_397b_petri_thorough_n30": "qwen3p5_397b_petri_eval",
    "opus_investigates_qwen3_32b_5000_thorough_n30": "qwen3_32b_wildchat_eval",
    "opus_investigates_llama3_1_8b_5000_thorough_n30": "llama3_1_8b_wildchat_eval",
    "opus_investigates_qwen3_8b_think1024_5000_thorough_n30": "qwen3_8b_wildchat_thinking_eval",
    "opus_investigates_qwen3p5_397b_think1024_5000_thorough_n30": "qwen3p5_397b_wildchat_thinking_eval",
    # training runs (n=10)
    "qwen3_8b_50k_train_structured": "qwen3_8b_wildchat_train",
    "qwen3p5_397b_110k_thinking_structured": "qwen3p5_397b_wildchat_self_investigate_train",
    "qwen3_32b_combined_train_structured": "qwen3_32b_wildchat_train",
    "llama3_1_8b_instruct_50k_train_structured": "llama3_1_8b_wildchat_train",
    # post-release addition: Qwen3.6-35B-A3B self-investigation (investigator = judge = target)
    "qwen3p6_35b_a3b_combined_train": "qwen3p6_35b_a3b_wildchat_self_investigate_train",
}

# Prompt pool dir (under data/runs/) -> public HF prompt-source name.
PROMPT_POOL_SLUGS: dict[str, str] = {
    "wildchat_plus_agentic": "prompts_wildchat_agentic",
    "wildchat48m_clean_172k": "prompts_wildchat_172k",
}

SLUG_DATASETS: dict[str, str] = {v: k for k, v in DATASET_SLUGS.items()}
SLUG_POOLS: dict[str, str] = {v: k for k, v in PROMPT_POOL_SLUGS.items()}
assert len(SLUG_DATASETS) == len(DATASET_SLUGS), "dataset slugs must be unique"
assert len(SLUG_POOLS) == len(PROMPT_POOL_SLUGS), "prompt pool slugs must be unique"

# Top-level dirs of the HF dataset repo (valid values for download_data.py --only).
HF_TOP_LEVEL_DIRS = ("datasets", "prompt_sources")

# Top-level provenance files released as-is (data/<name> -> <name> in the HF repo):
# the train/eval-leakage blocklist and the record of what it removed (2026-07-28).
HF_TOP_FILES = frozenset({"eval_blocklist.json", "eval_blocklist_filter_manifest.json"})

# Per-folder file whitelists (everything else in a run dir is NOT released:
# caches, archives, binary_evals, logs, scratch, dropped/leakage sidecars,
# the duplicate consolidated-.json serialization, ...).
_EVAL_FILES = frozenset(
    {
        "investigations.jsonl",
        "screening.jsonl",
        "verification.jsonl",
        "completions.jsonl",
        "prompts.jsonl",
        "config.json",
        "binary_counterfactual_questions.jsonl",
        "binary_counterfactual_questions.build_meta.json",
        "mechanism_concreteness.jsonl",
    }
)
_EVAL_PREFIXES = ("eval_set_",)

_TRAIN_FILES = frozenset(
    {
        "investigations.jsonl",
        "screening.jsonl",
        "verification.jsonl",
        "investigations.conversion_note.txt",
        "screening.conversion_note.txt",
        "verification.conversion_note.txt",
        "metadata.json",
        "binary_counterfactual_questions.jsonl",
        "binary_counterfactual_questions.build_meta.json",
        "provenance_config.json",
        "provenance_config_run1.json",
        "provenance_config_run2.json",
    }
)

_POOL_FILES = frozenset({"prompts.jsonl"})

# HF-side release metadata: curated in the staging tree, never mapped back into data/.
HF_METADATA_FILES = frozenset({"README.md", ".gitattributes"})


def _is_train(public_name: str) -> bool:
    return public_name.endswith("_train")


def _run_file_released(public_name: str, rest: str) -> bool:
    """Is data/runs/<run>/<rest> part of the release for this run?"""
    if _is_train(public_name):
        return rest in _TRAIN_FILES
    return rest in _EVAL_FILES or rest.startswith(_EVAL_PREFIXES)


def data_to_hf(data_rel: str) -> str | None:
    """Map a data/-relative POSIX path to its HF-repo-relative path.

    Returns None for files that are not part of the dataset release (the HF repo is
    datasets-only: experiment results, caches, non-whitelisted run files, and every
    non-run data/ subtree stay local).
    """
    if data_rel in HF_TOP_FILES:
        return data_rel
    parts = data_rel.split("/")
    if parts[0] != "runs" or len(parts) < 3:
        return None
    run, rest = parts[1], "/".join(parts[2:])
    if run in DATASET_SLUGS:
        public = DATASET_SLUGS[run]
        if _run_file_released(public, rest):
            return f"datasets/{public}/{rest}"
        return None
    if run in PROMPT_POOL_SLUGS:
        if rest in _POOL_FILES:
            return f"prompt_sources/{PROMPT_POOL_SLUGS[run]}/{rest}"
        return None
    return None


def hf_to_data(hf_rel: str) -> str | None:
    """Map an HF-repo-relative POSIX path back to its data/-relative path.

    Returns None for release metadata (README.md, .gitattributes, MANIFEST.txt).
    Raises on unmapped paths — every data file in the HF repo must be covered
    (fail loudly, no silent drops).
    """
    if hf_rel in HF_METADATA_FILES or hf_rel.split("/")[-1] == "MANIFEST.txt":
        return None
    if hf_rel in HF_TOP_FILES:
        return hf_rel
    parts = hf_rel.split("/")
    if len(parts) >= 3:
        top, public, rest = parts[0], parts[1], "/".join(parts[2:])
        if top == "datasets" and public in SLUG_DATASETS:
            if not _run_file_released(public, rest):
                raise ValueError(f"HF file outside the release whitelist: {hf_rel!r}")
            return f"runs/{SLUG_DATASETS[public]}/{rest}"
        if top == "prompt_sources" and public in SLUG_POOLS:
            if rest not in _POOL_FILES:
                raise ValueError(f"HF file outside the release whitelist: {hf_rel!r}")
            return f"runs/{SLUG_POOLS[public]}/{rest}"
    raise ValueError(f"no data mapping rule covers HF path: {hf_rel!r}")


def iter_data_files(data_root: Path):
    """Yield data/-relative POSIX paths of every regular file under data_root."""
    for p in sorted(data_root.rglob("*")):
        if p.is_file():
            yield p.relative_to(data_root).as_posix()


def _self_test(data_root: Path) -> None:
    n_mapped = 0
    skipped: dict[str, int] = {}  # reason-key -> count
    hf_seen: dict[str, str] = {}
    per_top: dict[str, int] = {}
    for rel in iter_data_files(data_root):
        hf = data_to_hf(rel)
        if hf is None:
            parts = rel.split("/")
            if parts[0] == "runs" and len(parts) > 1 and (
                parts[1] in DATASET_SLUGS or parts[1] in PROMPT_POOL_SLUGS
            ):
                key = f"runs/{parts[1]} (non-whitelisted files)"
            else:
                key = "/".join(parts[:1]) + "/ (not part of the dataset release)"
            skipped[key] = skipped.get(key, 0) + 1
            continue
        # injectivity
        if hf in hf_seen:
            raise AssertionError(f"collision: {rel!r} and {hf_seen[hf]!r} both map to {hf!r}")
        hf_seen[hf] = rel
        # round trip
        back = hf_to_data(hf)
        if back != rel:
            raise AssertionError(f"round-trip failure: {rel!r} -> {hf!r} -> {back!r}")
        top = hf.split("/", 1)[0]
        if top not in HF_TOP_LEVEL_DIRS and hf not in HF_TOP_FILES:
            raise AssertionError(f"unexpected top-level HF dir {top!r} for {rel!r}")
        per_top[top] = per_top.get(top, 0) + 1
        n_mapped += 1
    # static sanity: whitelisted release paths round-trip even if absent locally
    for run, public in DATASET_SLUGS.items():
        probe = "investigations.jsonl"
        hf = data_to_hf(f"runs/{run}/{probe}")
        assert hf == f"datasets/{public}/{probe}" and hf_to_data(hf) == f"runs/{run}/{probe}"
    for pool, public in PROMPT_POOL_SLUGS.items():
        hf = data_to_hf(f"runs/{pool}/prompts.jsonl")
        assert hf == f"prompt_sources/{public}/prompts.jsonl"
        assert hf_to_data(hf) == f"runs/{pool}/prompts.jsonl"
    assert hf_to_data("README.md") is None
    assert hf_to_data("datasets/qwen3_8b_wildchat_eval/MANIFEST.txt") is None
    print(f"self-test OK: {n_mapped} files round-trip exactly; static slug checks pass")
    for top in HF_TOP_LEVEL_DIRS:
        print(f"  {top:<16} {per_top.get(top, 0):>6} files")
    for key, n in sorted(skipped.items()):
        print(f"  not uploaded: {key} ({n} files)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="verify the mapping round-trips every file under data/")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = ap.parse_args()
    if args.self_test:
        _self_test(args.data_root)
    else:
        ap.print_help()
