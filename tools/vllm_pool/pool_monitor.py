"""Compact monitor for a vLLM pool proxy.

Prints the proxy's integrated /pool_metrics view plus optional JSONL progress
rate. This is intentionally small: the proxy owns backend instrumentation, and
this script only formats repeated snapshots for iteration.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fmt_float(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://localhost:8999")
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    previous_time: float | None = None
    previous_count: int | None = None

    while True:
        now = time.time()
        metrics = fetch_json(args.proxy.rstrip("/") + "/pool_metrics")
        totals = metrics["totals"]

        progress = ""
        if args.progress_jsonl is not None:
            count = count_lines(args.progress_jsonl)
            if previous_time is not None and previous_count is not None:
                elapsed_min = (now - previous_time) / 60.0
                rate = (count - previous_count) / elapsed_min if elapsed_min > 0 else 0.0
                progress = f" progress={count} rate={rate:.1f}/min"
            else:
                progress = f" progress={count} rate=n/a"
            previous_time = now
            previous_count = count

        print(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            f"name={metrics['proxy_name']}",
            f"healthy={metrics['n_healthy']}/{metrics['n_upstreams']}",
            f"cap={metrics['max_seqs_per_upstream']}",
            f"proxy_inflight={totals['proxy_in_flight']:.0f}",
            f"run={totals['vllm_running']:.0f}",
            f"wait={totals['vllm_waiting']:.0f}",
            f"preempt/s={fmt_float(totals['preemptions_per_sec'], 3)}",
            f"kv_avg={fmt_float(totals['kv_cache_usage_avg'], 3)}",
            f"kv_max={fmt_float(totals['kv_cache_usage_max'], 3)}",
            f"kv_ge95={totals['kv_cache_ge_95']}",
            f"prefix_hit={fmt_float(totals['prefix_cache_hit_rate'], 3)}",
            f"fallback={fmt_float(totals['fallback_route_rate'], 3)}",
            progress,
            flush=True,
        )

        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
