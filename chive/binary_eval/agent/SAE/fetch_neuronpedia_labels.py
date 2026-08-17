"""Download Gemma Scope 2 autointerp labels for one Neuronpedia source (bulk, via S3).

Neuronpedia's per-feature API is deprecated for bulk reads (it now 400s with a pointer to
the public dataset dump). The dump lives at
    https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1/{model}/{source}/explanations/batch-{i}.jsonl.gz
with 256 features per batch (256 batches for a 65k-width SAE). Each row carries the
autointerp label (`description`), the pipeline (`typeName`, e.g. np_acts-logits-general),
the explainer model (`explanationModelName`, e.g. gemini-2.5-flash-lite), and an embedding
vector we drop.

Output: one row per explanation, sorted by feature index ->
    <DATA_ROOT>/sae_labels/{model}_{source}.jsonl
    {"index": int, "description": str, "type": str, "explainer": str}

A feature can have zero explanations (not yet labeled — emitted as no row) or several
(e.g. an extra claude-generated label). Coverage stats print at the end; the SAE tool
shows unlabeled features as "(unlabeled)".

CPU-only, ~256 small downloads. Run from the repo root:
    python -m chive.binary_eval.agent.SAE.fetch_neuronpedia_labels
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from chive import paths

LABELS_DIR = paths.DATA_ROOT / "sae_labels"
S3_BASE = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"


def list_batch_keys(model: str, source: str) -> list[str]:
    """All explanations/batch-*.jsonl.gz keys for the source (paginated S3 listing)."""
    keys: list[str] = []
    token = None
    while True:
        url = (f"{S3_BASE}/?list-type=2&prefix=v1/{model}/{source}/explanations/"
               + (f"&continuation-token={requests.utils.quote(token)}" if token else ""))
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]+\.jsonl\.gz)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if m is None:
            return keys
        token = m.group(1)


def fetch_batch(key: str) -> list[dict]:
    r = requests.get(f"{S3_BASE}/{key}", timeout=120)
    r.raise_for_status()
    rows = []
    with gzip.open(io.BytesIO(r.content), "rt") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "index": int(d["index"]),
                "description": d["description"].strip(),
                "type": d["typeName"],
                "explainer": d["explanationModelName"],
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-3-27b-it")
    p.add_argument("--source", default="40-gemmascope-2-res-65k",
                   help="Neuronpedia source id; must match the SAE used by the cache")
    p.add_argument("--n-features", type=int, default=65536,
                   help="SAE width, for the coverage report")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    keys = list_batch_keys(args.model, args.source)
    assert keys, f"no explanation batches found for {args.model}/{args.source}"
    print(f"[labels] {args.model}/{args.source}: {len(keys)} batch files")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        batches = list(ex.map(fetch_batch, keys))
    rows = sorted((r for b in batches for r in b), key=lambda r: r["index"])

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABELS_DIR / f"{args.model}_{args.source}.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    covered = {r["index"] for r in rows}
    multi = len(rows) - len(covered)
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[f"{r['type']}/{r['explainer']}"] = by_type.get(f"{r['type']}/{r['explainer']}", 0) + 1
    print(f"[labels] wrote {len(rows)} explanations for {len(covered)}/{args.n_features} features "
          f"({len(covered)/args.n_features:.1%} coverage; {multi} features have >1 label) -> {out_path}")
    for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"[labels]   {k}: {v}")
    missing = args.n_features - len(covered)
    if missing / args.n_features > 0.05:
        print(f"*** WARNING: {missing} features ({missing/args.n_features:.1%}) have NO label ***")


if __name__ == "__main__":
    main()
