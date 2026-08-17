"""Round-robin front proxy for a pool of identical vLLM servers.

Presents ONE stable OpenAI-compatible endpoint while the set of backing servers
churns underneath it. Point the pipeline at `vllm:MODEL@http://<proxy-host>:PORT`
and never restart the proxy — add servers by submitting more Slurm jobs (they
self-register), and preempted servers drop out on their own.

Pool membership comes from one of two sources:
  --registry-dir DIR   (default) each server writes DIR/<jobid>.json once it is
                       serving; this proxy re-reads the dir every --refresh-interval
                       seconds. Vanished files (preempted/ended jobs) drop out.
  --upstreams URL,...  explicit static list (e.g. for testing against servers that
                       are already running and didn't self-register).

Routing is least-in-flight across healthy upstreams, with retry-on-another when a
server errors or was preempted mid-request, so churn is invisible to the caller.

CORRECTNESS GUARD: every upstream must serve the same model. The health check
reads each server's /v1/models; if --model is given, mismatches are marked
unhealthy, and if healthy upstreams ever disagree on the model id the proxy warns
loudly (mixing configs behind one endpoint would silently blend different
behavior into one dataset).

Run it co-located with the pipeline driver (CPU-only, no GPU):
    python tools/vllm_pool/pool_proxy.py --port 8999 --model Qwen/Qwen3.6-27B-FP8
    # then: stage1_llm = "vllm:Qwen/Qwen3.6-27B-FP8@http://localhost:8999"
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import aiohttp
from aiohttp import web

print = functools.partial(print, flush=True)


def _base(url: str) -> str:
    """Normalize a registry/CLI url to a scheme://host:port base (no trailing /v1)."""
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    if not url.startswith("http"):
        url = "http://" + url
    return url


class Upstream:
    __slots__ = (
        "base", "in_flight", "healthy", "blacklisted", "model", "config_hash", "source",
        "api_key",
        "route_count", "affinity_route_count", "fallback_route_count",
        "error_count", "last_error", "last_error_at", "failure_kind", "total_weight",
    )

    def __init__(
        self,
        base: str,
        source: str,
        config_hash: str | None = None,
        api_key: str | None = None,
    ):
        self.base = base
        self.in_flight = 0
        self.healthy = False
        # Sticky: once True, only cleared by the upstream being removed from the
        # registry (i.e. its Slurm job ends and a fresh one self-registers under a
        # different <jobid>.json). Refresh / health-check code MUST NOT clear it —
        # see _maybe_blacklist for the policy.
        self.blacklisted = False
        self.model: str | None = None
        self.config_hash = config_hash  # declared by the server in its registry file
        self.source = source  # registry filename or "cli"
        self.api_key = api_key
        self.route_count = 0
        self.affinity_route_count = 0
        self.fallback_route_count = 0
        # Consecutive backend failures. Reset on any successful upstream response.
        # This is intentionally not lifetime error count: under heavy steady-state
        # load, even good backends eventually accumulate a few transient failures.
        self.error_count = 0
        self.last_error: str | None = None
        self.last_error_at: float | None = None
        self.failure_kind: str | None = None
        self.total_weight = 0


class Pool:
    def __init__(self, args):
        self.args = args
        self.upstreams: dict[str, Upstream] = {}
        self._session: aiohttp.ClientSession | None = None
        # Notified whenever a per-upstream slot frees or the pool changes, so
        # requests waiting for capacity (backpressure) re-evaluate promptly.
        self._cond: asyncio.Condition | None = None
        # Set True during graceful shutdown so we reject new traffic with 503
        # while waiting for in-flight to drain (prevents vLLM-side orphans when
        # swapping proxies — see STRANDED_REQUESTS_INVESTIGATION.md).
        self.draining: bool = False
        self.metrics_prev: dict[str, tuple[float, dict[str, float]]] = {}
        self.waiting_acquires = 0
        self.acquire_wait_events = 0
        self.adaptive_prev_preemptions: dict[str, float] = {}
        self.adaptive_stable_intervals = 0
        self.adaptive_cooldown_until = 0.0
        self.control_path: Path | None = None

    @property
    def cond(self) -> asyncio.Condition:
        assert self._cond is not None, "cond not started"
        return self._cond

    @property
    def session(self) -> aiohttp.ClientSession:
        assert self._session is not None, "session not started"
        return self._session

    # --- membership ---------------------------------------------------------
    def _discover_candidates(self) -> dict[str, tuple[str, str | None, str | None]]:
        """Return {base_url: (source, config_hash, api_key)} for declared servers."""
        if self.args.upstreams:
            # Explicit list (testing): no declared config_hash to enforce.
            return {_base(u): ("cli", None, None) for u in self.args.upstreams.split(",") if u.strip()}
        candidates: dict[str, tuple[str, str | None, str | None]] = {}
        reg = Path(self.args.registry_dir)
        if not reg.is_dir():
            return candidates
        for f in reg.glob("*.json"):
            try:
                rec = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue  # half-written file; picked up next tick
            url = rec.get("url")
            if url:
                candidates[_base(url)] = (f.name, rec.get("config_hash"), rec.get("api_key"))
        return candidates

    def _headers_for(self, up: Upstream, *, content_type: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = "application/json"
        if up.api_key:
            headers["Authorization"] = f"Bearer {up.api_key}"
        return headers

    async def _check_one(self, up: Upstream) -> None:
        try:
            async with self.session.get(
                f"{up.base}/v1/models",
                headers=self._headers_for(up),
                timeout=aiohttp.ClientTimeout(total=self.args.health_timeout),
            ) as resp:
                if resp.status != 200:
                    up.healthy = False
                    return
                data = await resp.json()
            model_id = data["data"][0]["id"]
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError, ValueError):
            up.healthy = False
            return
        if self.args.model and model_id != self.args.model:
            if up.healthy or up.model != model_id:
                print(f"  [pool] REJECT {up.base}: serves {model_id!r}, expected {self.args.model!r}")
            up.healthy = False
            up.model = model_id
            return
        up.model = model_id
        up.healthy = True

    async def refresh(self) -> None:
        candidates = self._discover_candidates()
        # add new
        for base, (source, config_hash, api_key) in candidates.items():
            if base not in self.upstreams:
                self.upstreams[base] = Upstream(base, source, config_hash, api_key)
                print(f"  [pool] + discovered {base} (from {source}, config={config_hash})")
            else:
                up = self.upstreams[base]
                up.source = source
                up.config_hash = config_hash
                up.api_key = api_key
        # drop servers whose registration is gone (preempted/ended)
        for base in list(self.upstreams):
            if base not in candidates:
                print(f"  [pool] - deregistered {base}")
                del self.upstreams[base]
        # health-check all in parallel
        before = {b: u.healthy for b, u in self.upstreams.items()}
        await asyncio.gather(*(self._check_one(u) for u in self.upstreams.values()))
        for base, up in self.upstreams.items():
            if before.get(base) != up.healthy:
                print(f"  [pool] {base} -> {'HEALTHY' if up.healthy else 'unhealthy'}")
        self.enforce_consistency()
        n_healthy = sum(u.healthy for u in self.upstreams.values())
        cap = self.args.max_seqs_per_upstream
        print(f"  [pool] {n_healthy}/{len(self.upstreams)} healthy "
              f"(seq admit cap {n_healthy * cap} = {n_healthy} x {cap}; "
              f"in-flight seqs {sum(u.in_flight for u in self.upstreams.values())})")
        # Wake waiters: new/healthy servers may have added capacity.
        async with self.cond:
            self.cond.notify_all()

    def enforce_consistency(self) -> None:
        """Warn (without quarantining) when healthy upstreams disagree.

        The submit script hashes only the OUTPUT-affecting flag subset
        (model / parsers / max-model-len), so a config_hash mismatch should be
        rare and indicates a real semantic drift worth flagging. Perf knobs
        (prefix-caching, gpu-memory-utilization, gdn-prefill-backend, ...) are
        not in the hash, so a flag tweak does not force a pool rotation. We
        warn loudly instead of quarantining so the operator sees the drift
        without losing capacity during transient rollouts.
        """
        models = {u.model for u in self.upstreams.values() if u.healthy and u.model}
        if len(models) > 1:
            print(f"  [pool] !! WARNING: healthy upstreams disagree on model: {models}")
        healthy_hashes = [u.config_hash for u in self.upstreams.values() if u.healthy and u.config_hash]
        if healthy_hashes:
            modal = Counter(healthy_hashes).most_common(1)[0][0]
            for u in self.upstreams.values():
                if u.healthy and u.config_hash and u.config_hash != modal:
                    print(f"  [pool] !! WARNING {u.base}: config_hash {u.config_hash} "
                          f"!= pool majority {modal} (semantic flag drift — investigate)")

    async def refresh_loop(self) -> None:
        while True:
            try:
                await self.refresh()
            except Exception as e:  # never let the refresher die
                print(f"  [pool] refresh error: {type(e).__name__}: {e}")
            await asyncio.sleep(self.args.refresh_interval)

    # --- routing ------------------------------------------------------------
    async def acquire(
        self,
        tried: set[str],
        deadline: float,
        weight: int,
        affinity_key: bytes | None,
    ) -> tuple[Upstream, bool] | None:
        """Reserve `weight` sequence-slots on a healthy upstream.

        `in_flight` is counted in SEQUENCES, and a request with `n=K` reserves K
        (one slot per sampled sequence) — so the cap means the same server load
        whether the stage uses n=10 (generation/target) or n=1 (judges/agent).
        The first choice is affinity-routed when a stable key is available. If
        the preferred backend is saturated or unhealthy, the request falls back
        to least-loaded routing so one hot key cannot deadlock the pool.
        """
        cap = self.args.max_seqs_per_upstream
        async with self.cond:
            while True:
                untried = [u for u in self.upstreams.values()
                           if u.healthy and not u.blacklisted and u.base not in tried]
                if not untried:
                    return None
                if affinity_key is not None:
                    sorted_ups = sorted(untried, key=lambda u: u.base)
                    digest = hashlib.sha256(affinity_key).digest()
                    idx = int.from_bytes(digest[:4], "big") % len(sorted_ups)
                    preferred = sorted_ups[idx]
                    if preferred.in_flight < cap:
                        preferred.in_flight += weight
                        preferred.route_count += 1
                        preferred.affinity_route_count += 1
                        preferred.total_weight += weight
                        return preferred, False

                # `in_flight < cap` (not `+ weight <= cap`) so a single request whose
                # n exceeds the cap can still run somewhere instead of deadlocking;
                # overshoot is bounded by one request's n.
                avail = [u for u in untried if u.in_flight < cap]
                if avail:
                    up = min(avail, key=lambda u: u.in_flight)
                    up.in_flight += weight
                    up.route_count += 1
                    up.total_weight += weight
                    if affinity_key is not None:
                        up.fallback_route_count += 1
                    return up, affinity_key is not None
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return None  # backpressure timeout -> 503
                try:
                    self.waiting_acquires += 1
                    self.acquire_wait_events += 1
                    await asyncio.wait_for(self.cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None
                finally:
                    self.waiting_acquires -= 1

    async def release(self, up: Upstream, weight: int) -> None:
        async with self.cond:
            up.in_flight -= weight
            self.cond.notify(1)

    def set_max_seqs_per_upstream(self, value: int, reason: str) -> None:
        old = self.args.max_seqs_per_upstream
        if value == old:
            return
        self.args.max_seqs_per_upstream = value
        if self.control_path is not None:
            self.control_path.write_text(json.dumps({"max_seqs_per_upstream": value}, indent=2) + "\n")
        print(f"[pool] adaptive cap {old} -> {value}: {reason}")


async def _requeue_backend(jobid: str) -> None:
    """Fire-and-forget `scontrol requeue` of a wedged backend's job. Unlike
    bare `scancel` (which terminates permanently), requeue puts the job back
    into PENDING so the array slot is reclaimed and a fresh vLLM process
    eventually relaunches on whatever node Slurm schedules. The current vLLM
    process receives SIGTERM as part of the requeue, so submit_pool_server.sh's
    cleanup trap still runs and the registry file is removed; the next refresh
    tick (~15s) drops the upstream from the proxy's map.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "scontrol", "requeue", jobid,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            print(f"[pool] scontrol requeue {jobid} failed (rc={proc.returncode}): {stderr.decode().strip()}")
    except Exception as e:
        print(f"[pool] scontrol requeue {jobid} raised {type(e).__name__}: {e}")


def _maybe_blacklist(up: Upstream, threshold: int) -> None:
    """Transition an upstream into blacklisted state when its error_count crosses
    threshold so the proxy stops routing to it. We do NOT auto-requeue the
    Slurm job here — under heavy load (high KV, admit-timeouts) backends
    can return 5xx in bursts that look identical to a real wedge, and
    auto-requeueing then cycles the whole pool into a death spiral. A human
    (or a periodic monitor) handles requeueing blacklisted jobs manually.
    Idempotent: only the first crossing prints.
    """
    if up.blacklisted or up.error_count < threshold:
        return
    up.blacklisted = True
    jobid = up.source.removesuffix(".json")
    print(f"[pool] !! BLACKLIST {up.base} after {up.error_count} errors; job {jobid} left for manual review (no auto-requeue)")


def _affinity_key_from_payload(payload: object) -> bytes | None:
    """Stable key for routing all turns of one investigation to one backend."""
    if not isinstance(payload, dict):
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content[:512].encode("utf-8", errors="replace")
        if isinstance(content, list):
            encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
            return encoded[:2048].encode("utf-8", errors="replace")
        return None
    return None


async def proxy_request(request: web.Request) -> web.Response:
    pool: Pool = request.app["pool"]
    if pool.draining:
        return web.json_response(
            {"error": {"message": "pool_proxy: draining for shutdown, retry against the new proxy",
                       "type": "pool_proxy_draining"}},
            status=503,
        )
    body = await request.read()

    # We buffer and forward non-streaming JSON only. Reject stream=true explicitly
    # so a caller doesn't silently get mis-framed SSE through the JSON path.
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict) and payload.get("stream"):
        return web.json_response(
            {"error": {"message": "pool_proxy: streaming is not supported (set stream=false)",
                       "type": "pool_proxy_error"}},
            status=400,
        )

    # Reserve one sequence-slot per sampled sequence (request n), so the cap means
    # the same server load for n=10 generation and n=1 judge/agent calls alike.
    weight = 1
    if isinstance(payload, dict):
        n = payload.get("n", 1)
        weight = n if isinstance(n, int) and n > 0 else 1

    tried: set[str] = set()
    last_err = "no healthy upstreams in pool"
    deadline = asyncio.get_running_loop().time() + pool.args.acquire_timeout
    affinity_key = _affinity_key_from_payload(payload)

    # Retry on a *different* upstream when one errors/was preempted mid-request.
    for _ in range(pool.args.max_retries):
        acquired = await pool.acquire(tried, deadline, weight, affinity_key)
        if acquired is None:
            break
        up, _used_fallback = acquired
        try:
            async with pool.session.post(
                f"{up.base}{request.path}",
                data=body,
                headers=pool._headers_for(up, content_type=True),
                timeout=aiohttp.ClientTimeout(total=pool.args.request_timeout),
            ) as resp:
                raw = await resp.read()
                if resp.status >= 500:
                    # Server-side failure: demote and retry elsewhere, but do not
                    # count toward sticky blacklisting. vLLM emits 5xx under load,
                    # and treating overload noise as a wedge removes capacity just
                    # when the pool needs it most.
                    tried.add(up.base)
                    last_err = f"{up.base} returned {resp.status}"
                    up.last_error = last_err
                    up.last_error_at = time.time()
                    up.failure_kind = "5xx"
                    up.healthy = False
                    continue
                # 2xx and 4xx (e.g. genuine bad request) pass straight back.
                # Any response proves the backend is not wedged.
                up.error_count = 0
                return web.Response(
                    body=raw, status=resp.status,
                    content_type="application/json",
                )
        except asyncio.TimeoutError:
            # A timeout is stronger than a 5xx, but under heavy load a backend can
            # still time out one request while continuing to serve others. Require
            # a consecutive streak before sticky blacklisting.
            tried.add(up.base)
            last_err = f"{up.base}: timeout after {pool.args.request_timeout}s"
            up.error_count += 1
            up.last_error = last_err
            up.last_error_at = time.time()
            up.failure_kind = "timeout"
            up.healthy = False
            _maybe_blacklist(up, threshold=pool.args.blacklist_after)
        except aiohttp.ClientError as e:
            # Network/connection errors can indicate a dead/preempted backend.
            # Require a consecutive streak before sticky blacklisting.
            tried.add(up.base)
            last_err = f"{up.base}: {type(e).__name__}: {e}"
            up.error_count += 1
            up.last_error = last_err
            up.last_error_at = time.time()
            up.failure_kind = "client_error"
            up.healthy = False
            _maybe_blacklist(up, threshold=pool.args.blacklist_after)
        finally:
            await pool.release(up, weight)
        affinity_key = None

    return web.json_response(
        {"error": {"message": f"pool_proxy: no upstream served the request ({last_err})",
                   "type": "pool_proxy_error"}},
        status=503,
    )


async def list_models(request: web.Request) -> web.Response:
    pool: Pool = request.app["pool"]
    models = sorted({u.model for u in pool.upstreams.values() if u.healthy and u.model})
    if not models and pool.args.model:
        models = [pool.args.model]
    return web.json_response({
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "pool_proxy"} for m in models],
    })


PROM_METRIC_RE = re.compile(r"^(vllm:[^\s{]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", re.MULTILINE)


def _sum_metric(metrics: dict[str, list[float]], name: str) -> float:
    return sum(metrics[name]) if name in metrics else 0.0


def _max_metric(metrics: dict[str, list[float]], name: str) -> float:
    return max(metrics[name]) if name in metrics and metrics[name] else 0.0


async def _fetch_backend_metrics(pool: Pool, up: Upstream) -> dict:
    try:
        async with pool.session.get(
            f"{up.base}/metrics",
            headers=pool._headers_for(up),
            timeout=aiohttp.ClientTimeout(total=pool.args.metrics_timeout),
        ) as resp:
            text = await resp.text()
            status = resp.status
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return {
            "base": up.base,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "proxy_in_flight": up.in_flight,
        }
    if status != 200:
        return {
            "base": up.base,
            "ok": False,
            "error": f"HTTP {status}",
            "proxy_in_flight": up.in_flight,
        }
    raw: dict[str, list[float]] = {}
    for name, value in PROM_METRIC_RE.findall(text):
        raw.setdefault(name, []).append(float(value))
    return {
        "base": up.base,
        "ok": True,
        "proxy_in_flight": up.in_flight,
        "healthy": up.healthy,
        "blacklisted": up.blacklisted,
        "model": up.model,
        "source": up.source,
        "route_count": up.route_count,
        "affinity_route_count": up.affinity_route_count,
        "fallback_route_count": up.fallback_route_count,
        "error_count": up.error_count,
        "last_error": up.last_error,
        "last_error_at": up.last_error_at,
        "failure_kind": up.failure_kind,
        "total_weight": up.total_weight,
        "vllm_running": _sum_metric(raw, "vllm:num_requests_running"),
        "vllm_waiting": _sum_metric(raw, "vllm:num_requests_waiting"),
        "preemptions_total": _sum_metric(raw, "vllm:num_preemptions_total"),
        "kv_cache_usage": _max_metric(raw, "vllm:kv_cache_usage_perc"),
        "prefix_cache_hits_total": _sum_metric(raw, "vllm:prefix_cache_hits_total"),
        "prefix_cache_queries_total": _sum_metric(raw, "vllm:prefix_cache_queries_total"),
        "prompt_tokens_cached_total": _sum_metric(raw, "vllm:prompt_tokens_cached_total"),
    }


def _delta_rate(
    pool: Pool,
    now: float,
    row: dict,
    metric: str,
) -> float | None:
    prev = pool.metrics_prev.get(row["base"])
    if prev is None:
        return None
    prev_time, prev_metrics = prev
    elapsed = now - prev_time
    if elapsed <= 0:
        return None
    return (row[metric] - prev_metrics[metric]) / elapsed


def _reduced_cap(cap: int, factor: float, floor: int) -> int:
    new_cap = max(floor, int(cap * factor))
    if cap > floor and new_cap >= cap:
        new_cap = cap - 1
    return new_cap


def _adaptive_cap_status(pool: Pool) -> dict:
    now = asyncio.get_running_loop().time()
    return {
        "enabled": pool.args.adaptive_cap,
        "min": pool.args.adaptive_cap_min,
        "max": pool.args.adaptive_cap_max,
        "increase_step": pool.args.adaptive_cap_increase_step,
        "low_kv_threshold": pool.args.adaptive_cap_low_kv,
        "high_kv_threshold": pool.args.adaptive_cap_high_kv,
        "stable_intervals": pool.adaptive_stable_intervals,
        "stable_intervals_required": pool.args.adaptive_cap_stable_intervals,
        "cooldown_remaining_secs": max(0.0, pool.adaptive_cooldown_until - now),
        "waiting_acquires": pool.waiting_acquires,
        "acquire_wait_events": pool.acquire_wait_events,
    }


async def adaptive_cap_loop(pool: Pool) -> None:
    last_wait_events = pool.acquire_wait_events
    print(
        "[pool] adaptive cap enabled: "
        f"min={pool.args.adaptive_cap_min} max={pool.args.adaptive_cap_max} "
        f"low_kv={pool.args.adaptive_cap_low_kv} high_kv={pool.args.adaptive_cap_high_kv} "
        f"interval={pool.args.adaptive_cap_interval}s"
    )
    while True:
        await asyncio.sleep(pool.args.adaptive_cap_interval)
        try:
            rows = await asyncio.gather(
                *(_fetch_backend_metrics(pool, up) for up in pool.upstreams.values())
            )
            ok_rows = [r for r in rows if r["ok"]]
            if not ok_rows:
                print("[pool] adaptive cap hold: no backend metrics available")
                continue

            wait_events = pool.acquire_wait_events
            wait_delta = wait_events - last_wait_events
            last_wait_events = wait_events
            pressure = pool.waiting_acquires > 0 or wait_delta > 0

            kv_values = [r["kv_cache_usage"] for r in ok_rows]
            kv_max = max(kv_values)
            kv_avg = sum(kv_values) / len(kv_values)
            preemptions_delta = 0.0
            for row in ok_rows:
                base = row["base"]
                current = row["preemptions_total"]
                previous = pool.adaptive_prev_preemptions.get(base)
                if previous is not None:
                    preemptions_delta += max(0.0, current - previous)
                pool.adaptive_prev_preemptions[base] = current

            now = asyncio.get_running_loop().time()
            cap = pool.args.max_seqs_per_upstream
            reason_prefix = (
                f"pressure={pressure} wait_delta={wait_delta} waiting={pool.waiting_acquires} "
                f"kv_max={kv_max:.3f} kv_avg={kv_avg:.3f} "
                f"preemptions_delta={preemptions_delta:g}"
            )

            if preemptions_delta > 0:
                new_cap = _reduced_cap(
                    cap,
                    pool.args.adaptive_cap_preemption_factor,
                    pool.args.adaptive_cap_min,
                )
                pool.adaptive_stable_intervals = 0
                pool.adaptive_cooldown_until = now + pool.args.adaptive_cap_cooldown
                pool.set_max_seqs_per_upstream(new_cap, f"preemptions observed; {reason_prefix}")
                continue

            if kv_max >= pool.args.adaptive_cap_high_kv:
                new_cap = _reduced_cap(
                    cap,
                    pool.args.adaptive_cap_high_kv_factor,
                    pool.args.adaptive_cap_min,
                )
                pool.adaptive_stable_intervals = 0
                pool.adaptive_cooldown_until = now + pool.args.adaptive_cap_cooldown
                pool.set_max_seqs_per_upstream(new_cap, f"high KV usage; {reason_prefix}")
                continue

            if now < pool.adaptive_cooldown_until:
                pool.adaptive_stable_intervals = 0
                continue

            if pressure and kv_max < pool.args.adaptive_cap_low_kv:
                pool.adaptive_stable_intervals += 1
                if pool.adaptive_stable_intervals >= pool.args.adaptive_cap_stable_intervals:
                    new_cap = min(
                        pool.args.adaptive_cap_max,
                        cap + pool.args.adaptive_cap_increase_step,
                    )
                    pool.adaptive_stable_intervals = 0
                    pool.set_max_seqs_per_upstream(
                        new_cap,
                        f"sustained safe queue pressure; {reason_prefix}",
                    )
                continue

            pool.adaptive_stable_intervals = 0
        except Exception as e:  # never let the controller die
            print(f"[pool] adaptive cap error: {type(e).__name__}: {e}")


async def pool_metrics(request: web.Request) -> web.Response:
    pool: Pool = request.app["pool"]
    now = time.time()
    rows = await asyncio.gather(
        *(_fetch_backend_metrics(pool, up) for up in sorted(pool.upstreams.values(), key=lambda u: u.base))
    )
    ok_rows = [r for r in rows if r["ok"]]
    for row in ok_rows:
        row["preemptions_per_sec"] = _delta_rate(pool, now, row, "preemptions_total")
        row["prefix_cache_hits_per_sec"] = _delta_rate(pool, now, row, "prefix_cache_hits_total")
        row["prefix_cache_queries_per_sec"] = _delta_rate(pool, now, row, "prefix_cache_queries_total")
        pool.metrics_prev[row["base"]] = (
            now,
            {
                "preemptions_total": row["preemptions_total"],
                "prefix_cache_hits_total": row["prefix_cache_hits_total"],
                "prefix_cache_queries_total": row["prefix_cache_queries_total"],
            },
        )

    prefix_hits = sum(r["prefix_cache_hits_total"] for r in ok_rows)
    prefix_queries = sum(r["prefix_cache_queries_total"] for r in ok_rows)
    fallback_routes = sum(r["fallback_route_count"] for r in ok_rows)
    affinity_routes = sum(r["affinity_route_count"] for r in ok_rows)
    total_routes = sum(r["route_count"] for r in ok_rows)
    return web.json_response({
        "proxy_name": pool.args.proxy_name,
        "port": pool.args.port,
        "n_upstreams": len(pool.upstreams),
        "n_healthy": sum(u.healthy for u in pool.upstreams.values()),
        "draining": pool.draining,
        "max_seqs_per_upstream": pool.args.max_seqs_per_upstream,
        "waiting_acquires": pool.waiting_acquires,
        "acquire_wait_events": pool.acquire_wait_events,
        "adaptive_cap": _adaptive_cap_status(pool),
        "totals": {
            "proxy_in_flight": sum(u.in_flight for u in pool.upstreams.values()),
            "vllm_running": sum(r["vllm_running"] for r in ok_rows),
            "vllm_waiting": sum(r["vllm_waiting"] for r in ok_rows),
            "preemptions_total": sum(r["preemptions_total"] for r in ok_rows),
            "preemptions_per_sec": sum(
                r["preemptions_per_sec"] for r in ok_rows
                if r["preemptions_per_sec"] is not None
            ),
            "kv_cache_usage_avg": (
                sum(r["kv_cache_usage"] for r in ok_rows) / len(ok_rows)
                if ok_rows else 0.0
            ),
            "kv_cache_usage_max": max((r["kv_cache_usage"] for r in ok_rows), default=0.0),
            "kv_cache_ge_95": sum(r["kv_cache_usage"] >= 0.95 for r in ok_rows),
            "prefix_cache_hits_total": prefix_hits,
            "prefix_cache_queries_total": prefix_queries,
            "prefix_cache_hit_rate": prefix_hits / prefix_queries if prefix_queries else None,
            "affinity_routes": affinity_routes,
            "fallback_routes": fallback_routes,
            "total_routes": total_routes,
            "fallback_route_rate": fallback_routes / total_routes if total_routes else None,
        },
        "backends": rows,
    })


async def pool_status(request: web.Request) -> web.Response:
    pool: Pool = request.app["pool"]
    return web.json_response({
        "proxy_name": pool.args.proxy_name,
        "port": pool.args.port,
        "n_upstreams": len(pool.upstreams),
        "n_healthy": sum(u.healthy for u in pool.upstreams.values()),
        "n_blacklisted": sum(u.blacklisted for u in pool.upstreams.values()),
        "n_routable": sum(u.healthy and not u.blacklisted for u in pool.upstreams.values()),
        "draining": pool.draining,
        "max_seqs_per_upstream": pool.args.max_seqs_per_upstream,
        "waiting_acquires": pool.waiting_acquires,
        "acquire_wait_events": pool.acquire_wait_events,
        "adaptive_cap": _adaptive_cap_status(pool),
        "upstreams": [
            {"base": u.base, "healthy": u.healthy, "blacklisted": u.blacklisted,
             "in_flight": u.in_flight,
             "model": u.model, "config_hash": u.config_hash, "source": u.source,
             "route_count": u.route_count, "affinity_route_count": u.affinity_route_count,
             "fallback_route_count": u.fallback_route_count, "error_count": u.error_count,
             "last_error": u.last_error, "last_error_at": u.last_error_at,
             "failure_kind": u.failure_kind,
             "total_weight": u.total_weight}
            for u in sorted(pool.upstreams.values(), key=lambda u: u.base)
        ],
    })


async def control_status(request: web.Request) -> web.Response:
    pool: Pool = request.app["pool"]
    return web.json_response({
        "proxy_name": pool.args.proxy_name,
        "max_seqs_per_upstream": pool.args.max_seqs_per_upstream,
        "control_file": str(request.app["control_path"]),
    })


async def update_control(request: web.Request) -> web.Response:
    pool: Pool = request.app["pool"]
    payload = await request.json()
    if not isinstance(payload, dict) or set(payload) != {"max_seqs_per_upstream"}:
        return web.json_response(
            {"error": "expected JSON object with exactly max_seqs_per_upstream"},
            status=400,
        )
    value = payload["max_seqs_per_upstream"]
    if not isinstance(value, int) or value <= 0:
        return web.json_response(
            {"error": "max_seqs_per_upstream must be a positive integer"},
            status=400,
        )
    pool.args.max_seqs_per_upstream = value
    control_path: Path = request.app["control_path"]
    control_path.write_text(json.dumps({"max_seqs_per_upstream": value}, indent=2) + "\n")
    return await control_status(request)


async def on_startup(app: web.Application) -> None:
    pool: Pool = app["pool"]
    # limit=0: no client-side connection cap. aiohttp's DEFAULT is 100, which
    # would silently throttle the WHOLE pool to 100 in-flight regardless of how
    # many servers are up. We meter per-upstream ourselves (acquire/release), so
    # total connections are bounded by n_healthy x per_upstream_concurrency.
    pool._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
    pool._cond = asyncio.Condition()
    await pool.refresh()  # populate before we start accepting traffic
    app["refresher"] = asyncio.create_task(pool.refresh_loop())
    if pool.args.adaptive_cap:
        app["adaptive_cap"] = asyncio.create_task(adaptive_cap_loop(pool))


async def on_shutdown(app: web.Application) -> None:
    """Graceful drain: reject new requests with 503, wait for in-flight to finish.

    Without this, killing the proxy mid-flight closes the aiohttp ClientSession
    on requests vLLM has already accepted into its engine. vLLM doesn't always
    promptly abort those, and they pile up as "orphans" that compete for KV
    cache with whoever next holds the backends — causing the next driver's
    requests to be queued and preempted for many minutes.
    """
    pool: Pool = app["pool"]
    pool.draining = True
    print(f"[pool_proxy] draining: waiting up to {pool.args.drain_timeout}s for in-flight to finish")
    deadline = asyncio.get_running_loop().time() + pool.args.drain_timeout
    while True:
        in_flight = sum(u.in_flight for u in pool.upstreams.values())
        if in_flight == 0:
            print("[pool_proxy] drain complete: 0 in-flight")
            break
        if asyncio.get_running_loop().time() >= deadline:
            busy = [(u.base, u.in_flight) for u in pool.upstreams.values() if u.in_flight > 0]
            print(f"[pool_proxy] drain timed out with {in_flight} in-flight across {len(busy)} backends")
            for base, n in sorted(busy, key=lambda x: -x[1])[:10]:
                print(f"  still in-flight: {base}: {n}")
            break
        await asyncio.sleep(1.0)


async def on_cleanup(app: web.Application) -> None:
    app["refresher"].cancel()
    if "adaptive_cap" in app:
        app["adaptive_cap"].cancel()
    await app["pool"].session.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8999)
    p.add_argument("--proxy-name", default="pool",
                   help="name used for pid/control files so multiple proxies can share a registry safely")
    p.add_argument("--registry-dir", default=f"/workspace-vast/{os.environ['USER']}/vllm_pool")
    p.add_argument("--upstreams", default=None,
                   help="explicit comma-separated upstream urls; overrides --registry-dir")
    p.add_argument("--model", default=None,
                   help="expected model id; upstreams serving anything else are rejected")
    p.add_argument("--max-seqs-per-upstream", type=int, default=160,
                   help="max in-flight SEQUENCES per server (a request with n=K counts "
                        "as K). Default 160 is the starting cap for adaptive mode; "
                        "total admitted = n_healthy x this, auto-scaling as servers join/leave")
    p.add_argument("--refresh-interval", type=float, default=15.0)
    p.add_argument("--health-timeout", type=float, default=5.0)
    p.add_argument("--metrics-timeout", type=float, default=5.0)
    p.add_argument("--listen-backlog", type=int, default=4096,
                   help="TCP listen backlog for the proxy. Must be large enough "
                        "for high-concurrency localhost clients; aiohttp's "
                        "default is 128, which can reject connection waves.")
    p.add_argument("--request-timeout", type=float, default=900.0,
                   help="per-request HTTP timeout against an upstream. Hitting this "
                        "counts toward consecutive backend failures for sticky "
                        "blacklisting. This is a wedge guard for requests already "
                        "admitted to vLLM, not an overload detector; keep "
                        "--acquire-timeout short for admission backpressure.")
    p.add_argument("--acquire-timeout", type=float, default=3600.0,
                   help="max seconds a request waits for a per-upstream slot before "
                        "returning a retryable 503. Set long (1h) so the proxy "
                        "queue absorbs preempt-driven capacity dips instead of "
                        "turning them into 503s; the driver-side semaphore is the "
                        "real backpressure layer.")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--blacklist-after", type=int, default=5,
                   help="consecutive backend failures (timeout or ClientError; "
                        "5xx is retried/logged but not sticky-counted) "
                        "before an upstream is blacklisted. Success resets the count. "
                        "Once blacklisted, an upstream is only re-admitted via a fresh "
                        "Slurm job (new <jobid>.json).")
    p.add_argument("--adaptive-cap", action="store_true",
                   help="adapt max_seqs_per_upstream from proxy queue pressure and "
                        "vLLM KV/preemption metrics. Increases require proxy-side "
                        "admission waits and low KV usage; decreases happen on "
                        "preemptions or high max KV usage.")
    p.add_argument("--adaptive-cap-min", type=int, default=50,
                   help="lower bound for adaptive max_seqs_per_upstream")
    p.add_argument("--adaptive-cap-max", type=int, default=640,
                   help="upper bound for adaptive max_seqs_per_upstream")
    p.add_argument("--adaptive-cap-interval", type=float, default=30.0,
                   help="seconds between adaptive cap decisions")
    p.add_argument("--adaptive-cap-low-kv", type=float, default=0.70,
                   help="max backend KV usage must be below this to increase cap")
    p.add_argument("--adaptive-cap-high-kv", type=float, default=0.85,
                   help="max backend KV usage at or above this decreases cap")
    p.add_argument("--adaptive-cap-increase-step", type=int, default=10,
                   help="additive cap increase after sustained safe queue pressure")
    p.add_argument("--adaptive-cap-stable-intervals", type=int, default=2,
                   help="number of consecutive safe pressure intervals before increasing")
    p.add_argument("--adaptive-cap-cooldown", type=float, default=180.0,
                   help="seconds to hold after an adaptive decrease before increasing")
    p.add_argument("--adaptive-cap-preemption-factor", type=float, default=0.75,
                   help="multiplicative cap decrease when new vLLM preemptions appear")
    p.add_argument("--adaptive-cap-high-kv-factor", type=float, default=0.95,
                   help="multiplicative cap decrease when max KV usage is high")
    p.add_argument("--drain-timeout", type=float, default=180.0,
                   help="max seconds to wait for in-flight requests to finish during "
                        "graceful shutdown before forcing the session closed (and risking "
                        "vLLM-side orphans)")
    args = p.parse_args()
    if args.adaptive_cap_min <= 0:
        p.error("--adaptive-cap-min must be positive")
    if args.adaptive_cap_max < args.adaptive_cap_min:
        p.error("--adaptive-cap-max must be >= --adaptive-cap-min")
    if args.adaptive_cap and args.max_seqs_per_upstream > args.adaptive_cap_max:
        p.error("--max-seqs-per-upstream must be <= --adaptive-cap-max")
    if args.listen_backlog <= 0:
        p.error("--listen-backlog must be positive")
    if not 0.0 <= args.adaptive_cap_low_kv < args.adaptive_cap_high_kv <= 1.0:
        p.error("--adaptive-cap-low-kv and --adaptive-cap-high-kv must satisfy 0 <= low < high <= 1")
    if args.adaptive_cap_interval <= 0:
        p.error("--adaptive-cap-interval must be positive")
    if args.adaptive_cap_increase_step <= 0:
        p.error("--adaptive-cap-increase-step must be positive")
    if args.adaptive_cap_stable_intervals <= 0:
        p.error("--adaptive-cap-stable-intervals must be positive")
    if args.adaptive_cap_cooldown < 0:
        p.error("--adaptive-cap-cooldown must be non-negative")
    if not 0.0 < args.adaptive_cap_preemption_factor < 1.0:
        p.error("--adaptive-cap-preemption-factor must be between 0 and 1")
    if not 0.0 < args.adaptive_cap_high_kv_factor < 1.0:
        p.error("--adaptive-cap-high-kv-factor must be between 0 and 1")

    app = web.Application(client_max_size=256 * 1024 * 1024)  # verification prompts are large
    pool = Pool(args)
    runtime_prefix = f"proxy_{args.proxy_name}_{args.port}"
    control_path = Path(args.registry_dir) / f"{runtime_prefix}.control.json"
    pool.control_path = control_path
    app["pool"] = pool
    app["control_path"] = control_path
    app.router.add_get("/v1/models", list_models)
    app.router.add_get("/pool", pool_status)
    app.router.add_get("/pool_metrics", pool_metrics)
    app.router.add_get("/control", control_status)
    app.router.add_post("/control", update_control)
    app.router.add_get("/health", lambda r: web.Response(text="ok"))
    app.router.add_post("/v1/{tail:.*}", proxy_request)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.on_cleanup.append(on_cleanup)

    # Named pidfile for reliable teardown without collisions between dev/prod
    # proxies sharing a registry.
    pid_path = Path(args.registry_dir) / f"{runtime_prefix}.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    app["control_path"].write_text(
        json.dumps({"max_seqs_per_upstream": args.max_seqs_per_upstream}, indent=2) + "\n"
    )

    src = args.upstreams if args.upstreams else f"registry {args.registry_dir}"
    print(f"[pool_proxy] name={args.proxy_name} pid {os.getpid()} listening on :{args.port}  source={src}  "
          f"expected_model={args.model}  listen_backlog={args.listen_backlog}  pidfile={pid_path}")
    # shutdown_timeout must exceed drain_timeout, else aiohttp force-closes the
    # session while on_shutdown is still waiting for in-flight to drain.
    web.run_app(
        app, host="0.0.0.0", port=args.port, print=None,
        shutdown_timeout=args.drain_timeout + 10.0,
        backlog=args.listen_backlog,
    )


if __name__ == "__main__":
    main()
