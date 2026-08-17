"""Size-matched training data for the traditional privileged-access test.

The Qwen-generated training data (36,824 examples) is 54% larger than the Llama-generated
data (23,944). The traditional test trains the self and cross predictors of BOTH directions
on the same amount of data, so this subsamples the Qwen data to EXACTLY the Llama data's
size and per-cell balance (11,972 per cell for cells [supported, inert]), seeded, changing
nothing else. The two Qwen-target predictors train on this subsample.

Run (CPU): python experiments/privileged_access/make_sub24k_data.py
Writes results/training_data/binary_counterfactual_training_data_qwen_sub24k.json.
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from chive import paths

HERE = Path(__file__).resolve().parent
SRC = paths.DATA_ROOT / "training_data" / "binary_counterfactual_training_data.json"
REF = paths.DATA_ROOT / "training_data" / "binary_counterfactual_training_data_llama31_8b.json"
DST = (paths.experiment_data_dir("privileged_access")
       / "results" / "training_data" / "binary_counterfactual_training_data_qwen_sub24k.json")


def main():
    src = json.loads(SRC.read_text())
    ref_meta = json.loads(REF.read_text())["metadata"]
    per_cell_target = ref_meta["per_cell"]                       # 11,972
    cells = ref_meta["cells"]                                    # ["supported", "inert"]

    by_cell = collections.defaultdict(list)
    for r in src["results"]:
        by_cell[r["claim_type"]].append(r)
    assert sorted(by_cell) == sorted(cells), f"cell mismatch: {sorted(by_cell)} vs {cells}"

    rng = random.Random(0)
    picked = []
    for cell in cells:
        pool = by_cell[cell]
        assert len(pool) >= per_cell_target, f"{cell}: {len(pool)} < {per_cell_target}"
        picked.extend(rng.sample(pool, per_cell_target))

    out = {
        "metadata": {
            **src["metadata"],
            "n_examples": len(picked),
            "per_cell": per_cell_target,
            "subsampled_from": str(SRC.relative_to(paths.DATA_ROOT)),
            "size_matched_to": str(REF.relative_to(paths.DATA_ROOT)),
            "subsample_seed": 0,
            "note": "size-matched control: Qwen data subsampled to the Llama data's exact "
                    "size and per-cell balance; nothing else changed",
        },
        "results": picked,
    }
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(out))
    counts = collections.Counter(r["claim_type"] for r in picked)
    print(f"wrote {DST}: n={len(picked)} per-cell={dict(counts)} (target {per_cell_target}/cell)")


if __name__ == "__main__":
    main()
