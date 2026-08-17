"""Build the unfiltered-complement eval sets (see README.md).

Per WildChat run, the complement = full cf7 bank
  − claims already in the conc3 eval set (already scored by every arm)
  − claims whose ORIGINAL conversation content-hashes into the run's trained-model
    training data (the trained models saw those conversations; gemma has no trained arm)
  − investigations that fail the transcript-segmentation hard requirement (expected 0).

The analysis population "no filters" := conc3 eval set ∪ complement (the train-overlap
claim_ids are subtracted there too — on 8B that removes 8 claims from the conc3 part).
Rows are byte-identical bank rows (pass-through); concreteness is joined at analysis time
from <run>/mechanism_concreteness.jsonl.

Run (CPU-only):
    PYTHONPATH=. python experiments/unfiltered_evaluation/build_unfiltered_complement.py

Outputs (all inside this experiment's data dir):
    eval_sets/<slug>_complement.jsonl
    results/train_overlap_<slug>.json   (claim_ids in the FULL bank that overlap training data)
    results/build_manifest.json
"""

import json
from pathlib import Path

from chive import paths
from chive.binary_eval.agent.transcript_segments import segment_transcript
from chive.training.balance_and_export import _conv_hash

DATA = paths.experiment_data_dir("unfiltered_evaluation")
RUNS = paths.RUNS_DIR
TRAIN_DATA = paths.DATA_ROOT / "training_data"

# slug -> (run_dir, target model / tokenizer, training-data json or None)
CONFIG = {
    "gemma": (
        RUNS / "opus_investigates_gemma3_27b_it_5000_thorough_n30",
        "google/gemma-3-27b-it",
        None,  # no trained arm on gemma -> no train-overlap exclusion
    ),
    "qwen8b": (
        RUNS / "opus_investigates_qwen3_8b_5000_thorough_n30",
        "Qwen/Qwen3-8B",
        TRAIN_DATA / "binary_counterfactual_training_data.json",
    ),
    "q397": (
        RUNS / "opus_investigates_qwen3p5_397b_5000_thorough_n30",
        "Qwen/Qwen3.5-397B-A17B-FP8",
        TRAIN_DATA / "binary_counterfactual_training_data_qwen3p5_397b.json",
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def build_one(slug: str, run_dir: Path, base_model: str, train_json: Path | None) -> dict:
    bank = load_jsonl(run_dir / "binary_counterfactual_questions.jsonl")
    conc = {r["prompt_id"]: r["mechanism_concreteness"]
            for r in load_jsonl(run_dir / "mechanism_concreteness.jsonl")}
    eval_set_ids = {r["claim_id"] for r in load_jsonl(run_dir / "eval_set_conc3_n2000.jsonl")}

    ids = [r["claim_id"] for r in bank]
    assert len(ids) == len(set(ids)), f"{slug}: duplicate claim_id in bank"
    unrated = [r["prompt_id"] for r in bank if r["prompt_id"] not in conc]
    assert not unrated, f"{slug}: {len(unrated)} bank prompts missing a concreteness rating"

    # Train-overlap over the FULL bank (the analysis subtracts these from the conc3 part too).
    overlap_ids: list[str] = []
    if train_json is not None:
        train_hashes = {_conv_hash(r["messages"]) for r in json.load(open(train_json))["results"]}
        overlap_ids = [r["claim_id"] for r in bank if _conv_hash(r["messages"]) in train_hashes]
    overlap_set = set(overlap_ids)

    complement = [r for r in bank
                  if r["claim_id"] not in eval_set_ids and r["claim_id"] not in overlap_set]

    # Segmentation: HARD requirement (same as build_eval_set.py), checked once per prompt_id.
    # Expected 0 failures; anything else stops the build for a human decision.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    failed: dict[str, str] = {}
    checked: set[str] = set()
    for r in complement:
        pid = r["prompt_id"]
        if pid in checked:
            continue
        checked.add(pid)
        try:
            segment_transcript(tokenizer, r["messages"], r["chosen_completion"])
        except AssertionError as e:
            failed[pid] = str(e)
    if failed:
        sample = "\n".join(f"  {pid}: {msg}" for pid, msg in list(failed.items())[:5])
        raise RuntimeError(
            f"{slug}: segmentation failed for {len(failed)}/{len(checked)} complement "
            f"investigations over {base_model} — these can't be scored by the agent arms. "
            f"Decide before proceeding.\n{sample}")

    out_path = DATA / "eval_sets" / f"{slug}_complement.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in complement:
            fh.write(json.dumps(r) + "\n")

    (DATA / "results").mkdir(parents=True, exist_ok=True)
    with open(DATA / "results" / f"train_overlap_{slug}.json", "w") as fh:
        json.dump({"training_data": str(train_json), "n_overlap": len(overlap_ids),
                   "claim_ids": sorted(overlap_ids)}, fh, indent=1)

    n_true = sum(r["is_true"] for r in complement)
    by_conc: dict[int, int] = {}
    for r in complement:
        by_conc[conc[r["prompt_id"]]] = by_conc.get(conc[r["prompt_id"]], 0) + 1
    manifest = {
        "run_dir": str(run_dir),
        "bank_claims": len(bank),
        "in_conc3_eval_set": len(eval_set_ids),
        "train_overlap_total": len(overlap_ids),
        "train_overlap_in_eval_set": len(overlap_set & eval_set_ids),
        "segmentation_checked": len(checked),
        "segmentation_failed": 0,
        "complement_n": len(complement),
        "complement_true_false": f"{n_true}T/{len(complement) - n_true}F",
        "complement_by_concreteness": {str(k): v for k, v in sorted(by_conc.items())},
        "out": str(out_path),
    }
    print(f"[{slug}] " + json.dumps(manifest, indent=1))
    return manifest


def main():
    manifests = {slug: build_one(slug, *cfg) for slug, cfg in CONFIG.items()}
    with open(DATA / "results" / "build_manifest.json", "w") as fh:
        json.dump(manifests, fh, indent=1)
    print(f"\nwrote {DATA / 'results' / 'build_manifest.json'}")


if __name__ == "__main__":
    main()
