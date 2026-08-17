#!/usr/bin/env python3
"""Benchmark OpenAI-compatible chat endpoints with full response capture."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import aiohttp


@dataclass(frozen=True)
class RequestResult:
    endpoint: str
    request_id: int
    ok: bool
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int
    status: int | None
    response: dict | None
    error: str | None


@dataclass(frozen=True)
class EndpointSummary:
    endpoint: str
    url: str
    n_requests: int
    ok: int
    failed: int
    wall_s: float
    req_per_s: float
    prompt_tok_per_s: float
    completion_tok_per_s: float
    p50_s: float
    p95_s: float


def parse_endpoint(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"endpoint must be NAME=URL, got {raw!r}")
    name, url = raw.split("=", 1)
    name = name.strip()
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    if not name:
        raise ValueError(f"empty endpoint name in {raw!r}")
    return name, url


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int((len(ordered) - 1) * q))
    return ordered[idx]


async def one_request(
    session: aiohttp.ClientSession,
    endpoint_name: str,
    base_url: str,
    request_id: int,
    model: str,
    max_tokens: int,
) -> RequestResult:
    prompt = (
        "Write a compact technical note about measuring vLLM throughput. "
        f"Use request id {request_id}. Include exactly three short paragraphs."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    start = time.perf_counter()
    status = None
    try:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
            status = resp.status
            body = await resp.json()
        elapsed = time.perf_counter() - start
        usage = body["usage"]
        if status != 200:
            return RequestResult(
                endpoint=endpoint_name,
                request_id=request_id,
                ok=False,
                elapsed_s=elapsed,
                prompt_tokens=0,
                completion_tokens=0,
                status=status,
                response=body,
                error=f"HTTP {status}",
            )
        return RequestResult(
            endpoint=endpoint_name,
            request_id=request_id,
            ok=True,
            elapsed_s=elapsed,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            status=status,
            response=body,
            error=None,
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return RequestResult(
            endpoint=endpoint_name,
            request_id=request_id,
            ok=False,
            elapsed_s=elapsed,
            prompt_tokens=0,
            completion_tokens=0,
            status=status,
            response=None,
            error=f"{type(e).__name__}: {e}",
        )


async def bench_endpoint(
    endpoint_name: str,
    base_url: str,
    n_requests: int,
    concurrency: int,
    model: str,
    max_tokens: int,
) -> tuple[EndpointSummary, list[RequestResult]]:
    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 100))
    sem = asyncio.Semaphore(concurrency)
    start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def guarded(i: int) -> RequestResult:
            async with sem:
                return await one_request(session, endpoint_name, base_url, i, model, max_tokens)

        results = await asyncio.gather(*(guarded(i) for i in range(n_requests)))
    wall = time.perf_counter() - start
    ok_results = [r for r in results if r.ok]
    latencies = [r.elapsed_s for r in ok_results]
    prompt_tokens = sum(r.prompt_tokens for r in ok_results)
    completion_tokens = sum(r.completion_tokens for r in ok_results)
    summary = EndpointSummary(
        endpoint=endpoint_name,
        url=base_url,
        n_requests=n_requests,
        ok=len(ok_results),
        failed=n_requests - len(ok_results),
        wall_s=wall,
        req_per_s=len(ok_results) / wall if wall else 0.0,
        prompt_tok_per_s=prompt_tokens / wall if wall else 0.0,
        completion_tok_per_s=completion_tokens / wall if wall else 0.0,
        p50_s=median(latencies) if latencies else 0.0,
        p95_s=percentile(latencies, 0.95),
    )
    return summary, results


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True, help="NAME=URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    endpoints = [parse_endpoint(raw) for raw in args.endpoint]
    summaries: list[EndpointSummary] = []
    with args.out.open("w") as f:
        for name, url in endpoints:
            summary, results = await bench_endpoint(
                endpoint_name=name,
                base_url=url,
                n_requests=args.n_requests,
                concurrency=args.concurrency,
                model=args.model,
                max_tokens=args.max_tokens,
            )
            summaries.append(summary)
            for result in results:
                f.write(json.dumps(asdict(result)) + "\n")
            print(
                f"{summary.endpoint}: ok={summary.ok}/{summary.n_requests} "
                f"wall={summary.wall_s:.1f}s req/s={summary.req_per_s:.2f} "
                f"prompt_tok/s={summary.prompt_tok_per_s:.0f} "
                f"completion_tok/s={summary.completion_tok_per_s:.0f} "
                f"p50={summary.p50_s:.1f}s p95={summary.p95_s:.1f}s",
                flush=True,
            )

    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps([asdict(s) for s in summaries], indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
