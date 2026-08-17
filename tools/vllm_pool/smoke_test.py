"""Smoke tests for the vLLM pool proxy.

Six checks; first is offline, the rest need a live proxy + pool:

  consistency   OFFLINE unit test of the config/model quarantine guard — no network.
                Verifies a drifted-config member is quarantined and the majority kept.
  route         LIVE: burst N concurrent n=10 requests through the proxy; assert all
                succeed with 10 choices each, and per-server in-flight never exceeds
                the cap (run the proxy with a low --max-seqs-per-upstream to also
                exercise backpressure).
  churn         LIVE: start a sustained burst, scancel ONE pool task mid-flight, and
                assert zero failed requests (retry-on-another + the proxy's health
                eviction make the preemption invisible). Also resubmits a replacement.
  affinity      LIVE: send N identical prompts; verify they all land on the SAME
                backend (affinity routing pins same-prompt traffic for prefix-cache
                locality). Fallback only if the preferred backend is saturated.
  cap_control   LIVE: GET /control, POST /control to change the cap, verify it took
                effect in /pool, persisted to disk, then restored.
  metrics_basic LIVE: GET /pool_metrics, verify shape (required keys present, at
                least 1 healthy backend, all backend rows reported ok).

Usage:
  python tools/vllm_pool/smoke_test.py --tests consistency
  python tools/vllm_pool/smoke_test.py --tests route,affinity,cap_control,metrics_basic --proxy-url http://localhost:8999
  python tools/vllm_pool/smoke_test.py --tests churn  --proxy-url http://localhost:8999
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
import types
from pathlib import Path

import aiohttp

import pool_proxy


def test_consistency() -> bool:
    """Offline: the guard quarantines drifted config_hash / model, keeps the majority."""
    args = types.SimpleNamespace(per_upstream_concurrency=48, model=None)
    pool = pool_proxy.Pool(args)

    def add(base, healthy, model, chash):
        u = pool_proxy.Upstream(base, "test", chash)
        u.healthy, u.model = healthy, model
        pool.upstreams[base] = u

    # 3 members agree (hashA), 1 drifted (hashB), 1 already-unhealthy drifted.
    add("http://a", True, "M", "hashA")
    add("http://b", True, "M", "hashA")
    add("http://c", True, "M", "hashA")
    add("http://d", True, "M", "hashB")          # drifted -> must be quarantined
    add("http://e", False, "M", "hashB")         # already down -> stays down

    pool.enforce_consistency()
    ok = True
    for base in ("http://a", "http://b", "http://c"):
        if not pool.upstreams[base].healthy:
            print(f"  FAIL: majority member {base} was wrongly quarantined"); ok = False
    if pool.upstreams["http://d"].healthy:
        print("  FAIL: drifted-config member d was NOT quarantined"); ok = False
    print(f"  consistency: {'PASS' if ok else 'FAIL'} "
          f"(healthy after guard: {sorted(b for b,u in pool.upstreams.items() if u.healthy)})")
    return ok


async def _healthy_bases(session, proxy_url):
    async with session.get(f"{proxy_url}/pool") as r:
        d = await r.json()
    return [u["base"] for u in d["upstreams"] if u["healthy"]], d


async def _one(session, proxy_url, model, max_tokens):
    payload = {"model": model, "messages": [{"role": "user", "content": "Name one fruit."}],
               "n": 10, "max_tokens": max_tokens, "temperature": 1.0}
    async with session.post(f"{proxy_url}/v1/chat/completions", json=payload) as r:
        body = await r.json()
        return r.status, len(body.get("choices", []))


async def _sample_inflight(session, proxy_url, stop, peak):
    while not stop.is_set():
        try:
            async with session.get(f"{proxy_url}/pool") as r:
                d = await r.json()
            for u in d["upstreams"]:
                peak[u["base"]] = max(peak.get(u["base"], 0), u["in_flight"])
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(0.3)


async def test_route(proxy_url, model, n_requests, cap, max_tokens) -> bool:
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        bases, _ = await _healthy_bases(s, proxy_url)
        print(f"  {len(bases)} healthy upstreams")
        if not bases:
            print("  FAIL: no healthy upstreams"); return False
        stop = asyncio.Event(); peak: dict[str, int] = {}
        sampler = asyncio.create_task(_sample_inflight(s, proxy_url, stop, peak))
        t = time.time()
        res = await asyncio.gather(*[_one(s, proxy_url, model, max_tokens) for _ in range(n_requests)],
                                   return_exceptions=True)
        stop.set(); await sampler
    errs = [r for r in res if isinstance(r, Exception)]
    bad = [r for r in res if not isinstance(r, Exception) and (r[0] != 200 or r[1] != 10)]
    peak_max = max(peak.values()) if peak else 0
    over = {b: v for b, v in peak.items() if v > cap}
    ok = not errs and not bad and not over
    print(f"  route: {'PASS' if ok else 'FAIL'} — {n_requests-len(errs)-len(bad)}/{n_requests} good "
          f"in {time.time()-t:.1f}s; peak in-flight/server={peak_max} (cap {cap})")
    if errs: print(f"    errors: {len(errs)} e.g. {repr(errs[0])[:160]}")
    if bad: print(f"    bad responses (status/choices): {bad[:5]}")
    if over: print(f"    OVER CAP: {over}")
    return ok


async def test_churn(proxy_url, model, n_requests, max_tokens, registry_dir) -> bool:
    # Sacrifice a server that is ACTUALLY in the pool (registered + healthy), so the
    # test exercises a real mid-flight preemption rather than killing a still-loading
    # task the proxy never routed to. PREFER a HIGH-prio victim: high re-lands almost
    # immediately on resubmit, so the test self-heals the pool; cancelling a low one
    # can leave it stuck pending under contention. The preemption mechanism the test
    # validates (retry-on-another + health eviction) is identical regardless of qos.
    import glob
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        bases, _ = await _healthy_bases(s, proxy_url)
    healthy = set(bases)
    candidates = []  # (qos, job_id, base)
    for f in glob.glob(f"{registry_dir}/*.json"):
        try:
            rec = json.loads(open(f).read())
        except (json.JSONDecodeError, OSError):
            continue
        if pool_proxy._base(rec["url"]) in healthy:
            candidates.append((rec.get("qos"), rec["job_id"], pool_proxy._base(rec["url"])))
    candidates.sort(key=lambda c: 0 if c[0] == "high" else 1)  # high first
    if not candidates:
        print("  churn: SKIP — no healthy upstream registered yet"); return True
    victim_qos, victim, victim_base = candidates[0]
    script = "tools/vllm_pool/submit_qwen3.6_35b.sh"
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        bases, _ = await _healthy_bases(s, proxy_url)
        print(f"  churn: {len(bases)} healthy upstreams; will scancel {victim_qos} job {victim} "
              f"({victim_base}) mid-burst")
        async def burst():
            return await asyncio.gather(*[_one(s, proxy_url, model, max_tokens) for _ in range(n_requests)],
                                        return_exceptions=True)
        task = asyncio.create_task(burst())
        await asyncio.sleep(3)
        print(f"  churn: scancel {victim} (stated explicitly; it is my mu_pool_{victim_qos[:2]} task)")
        subprocess.run(["scancel", victim])
        res = await task
    # Self-heal: resubmit one replacement of the same qos so the pool recovers.
    print(f"  churn: resubmitting one {victim_qos} replacement")
    subprocess.run(["sbatch", f"--qos={victim_qos}", "--partition=general,overflow",
                    f"--job-name=mu_pool_{victim_qos[:2]}", script])
    errs = [r for r in res if isinstance(r, Exception)]
    bad = [r for r in res if not isinstance(r, Exception) and (r[0] != 200 or r[1] != 10)]
    ok = not errs and not bad
    print(f"  churn: {'PASS' if ok else 'FAIL'} — {n_requests-len(errs)-len(bad)}/{n_requests} "
          f"requests survived the preemption")
    if errs: print(f"    errors: {len(errs)} e.g. {repr(errs[0])[:160]}")
    if bad: print(f"    bad responses: {bad[:5]}")
    return ok


async def test_affinity(proxy_url: str, model: str, n_requests: int, max_tokens: int) -> bool:
    """Same prompt N times in parallel; verify all hit the SAME backend (route_count delta)."""
    payload = {"model": model, "messages": [{"role": "user", "content": "Say 'hi' and stop."}],
               "n": 1, "max_tokens": max_tokens, "temperature": 0.7}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        async with s.get(f"{proxy_url}/pool") as r:
            before = await r.json()
        before_routes = {u["base"]: u["route_count"] for u in before["upstreams"]}
        async def one():
            async with s.post(f"{proxy_url}/v1/chat/completions", json=payload) as r:
                return r.status, await r.text()
        t = time.time()
        results = await asyncio.gather(*[one() for _ in range(n_requests)], return_exceptions=True)
        async with s.get(f"{proxy_url}/pool") as r:
            after = await r.json()
    deltas = {u["base"]: u["route_count"] - before_routes.get(u["base"], 0)
              for u in after["upstreams"] if u["route_count"] - before_routes.get(u["base"], 0) > 0}
    errs = [r for r in results if isinstance(r, Exception)]
    bad = [r for r in results if not isinstance(r, Exception) and r[0] != 200]
    # Pass if 1 backend received all routes (perfect affinity) and 0 errors.
    ok = len(deltas) == 1 and not errs and not bad
    print(f"  affinity: {'PASS' if ok else 'FAIL'} — {n_requests-len(errs)-len(bad)}/{n_requests} good in "
          f"{time.time()-t:.1f}s; routed_to={list(deltas)}")
    if errs: print(f"    errors: {len(errs)} e.g. {repr(errs[0])[:160]}")
    if bad: print(f"    bad responses (status): {[r[0] for r in bad[:5]]}")
    if len(deltas) > 1: print(f"    routes scattered: {deltas}")
    return ok


async def test_cap_control(proxy_url: str) -> bool:
    """GET /control, POST a new cap, verify /pool reflects it + control file persisted, then restore."""
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        async with s.get(f"{proxy_url}/control") as r:
            before = await r.json()
        old_cap = before["max_seqs_per_upstream"]
        new_cap = max(1, old_cap // 2)
        async with s.post(f"{proxy_url}/control", json={"max_seqs_per_upstream": new_cap}) as r:
            after = await r.json()
        async with s.get(f"{proxy_url}/pool") as r:
            pool = await r.json()
        pass_pool = pool["max_seqs_per_upstream"] == new_cap
        # Restore the old cap so subsequent tests get the original concurrency budget.
        async with s.post(f"{proxy_url}/control", json={"max_seqs_per_upstream": old_cap}) as r:
            restored = await r.json()
    control_file = Path(after["control_file"])
    pass_persist = control_file.exists() and json.loads(
        control_file.read_text())["max_seqs_per_upstream"] == old_cap
    ok = (after["max_seqs_per_upstream"] == new_cap and pass_pool
          and restored["max_seqs_per_upstream"] == old_cap and pass_persist)
    print(f"  cap_control: {'PASS' if ok else 'FAIL'} — old={old_cap} new={new_cap} restored={restored['max_seqs_per_upstream']} "
          f"pool_reflected_new={pass_pool} control_file_persisted={pass_persist}")
    return ok


async def test_metrics_basic(proxy_url: str) -> bool:
    """GET /pool_metrics; verify required keys + ≥1 healthy + all backend rows ok."""
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        async with s.get(f"{proxy_url}/pool_metrics") as r:
            m = await r.json()
    required_top = {"proxy_name", "n_upstreams", "n_healthy", "max_seqs_per_upstream", "totals", "backends"}
    required_totals = {"proxy_in_flight", "vllm_running", "vllm_waiting", "preemptions_total",
                       "kv_cache_usage_avg", "prefix_cache_hits_total", "prefix_cache_queries_total",
                       "affinity_routes", "fallback_routes", "total_routes"}
    missing_top = required_top - set(m)
    missing_totals = required_totals - set(m.get("totals", {}))
    backends_ok = all(b.get("ok") for b in m.get("backends", []))
    n_healthy = m.get("n_healthy", 0)
    ok = not missing_top and not missing_totals and backends_ok and n_healthy >= 1
    print(f"  metrics_basic: {'PASS' if ok else 'FAIL'} — n_healthy={n_healthy} "
          f"missing_top={sorted(missing_top)} missing_totals={sorted(missing_totals)} all_backends_ok={backends_ok}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", default="consistency",
                   help="comma list: consistency,route,churn,affinity,cap_control,metrics_basic")
    p.add_argument("--proxy-url", default="http://localhost:8999")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument("--n-requests", type=int, default=40)
    p.add_argument("--cap", type=int, default=160, help="must match the proxy's --max-seqs-per-upstream (sequences; route fires n=10 so ~160=16 reqs/server)")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--n-affinity", type=int, default=10,
                   help="affinity test: number of identical-prompt requests to fire")
    p.add_argument("--registry-dir", default=f"/workspace-vast/{os.environ['USER']}/vllm_pool")
    args = p.parse_args()

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    results: dict[str, bool] = {}
    if "consistency" in tests:
        print("[consistency]"); results["consistency"] = test_consistency()
    if "route" in tests:
        print("[route]"); results["route"] = asyncio.run(
            test_route(args.proxy_url, args.model, args.n_requests, args.cap, args.max_tokens))
    if "churn" in tests:
        print("[churn]"); results["churn"] = asyncio.run(
            test_churn(args.proxy_url, args.model, args.n_requests, args.max_tokens, args.registry_dir))
    if "affinity" in tests:
        print("[affinity]"); results["affinity"] = asyncio.run(
            test_affinity(args.proxy_url, args.model, args.n_affinity, args.max_tokens))
    if "cap_control" in tests:
        print("[cap_control]"); results["cap_control"] = asyncio.run(test_cap_control(args.proxy_url))
    if "metrics_basic" in tests:
        print("[metrics_basic]"); results["metrics_basic"] = asyncio.run(test_metrics_basic(args.proxy_url))

    print("\n=== summary ===")
    for t, ok in results.items():
        print(f"  {t}: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
