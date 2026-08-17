# vLLM pool + affinity proxy

A pool of **independent vLLM servers** behind **one stable OpenAI-compatible endpoint**. The pipeline points at the endpoint and forgets it — add backends (Slurm or RunPod, preemptible or not) while the run is live; preempted/dead backends drop out automatically.

> This is the throughput infrastructure used for the paper's large runs (5000-prompt canonical
> pipelines, the 397B arms). It is NOT required for correctness: any single OpenAI-compatible
> vLLM server works (`chive/pipeline/slurm_scripts/submit_vllm_server*.sh`), just slower.
> See `docs/infra.md` for the overall execution model. The Slurm/RunPod scripts here are
> cluster-specific — adapt partitions, QoS names, and paths to your setup.

```
   pipeline ──► pool_proxy.py :PORT ──► vllm backend-A
   (stage*_llm/...        │ affinity by prompt hash    vllm backend-B
    @localhost:PORT)      │ + per-backend seq cap       vllm backend-C   ... (Slurm + RunPod)
                          │ + drain on SIGTERM          each backend has
                          │ + retry-on-another          its registry JSON
                          └ reads registry dir
```

For per-model flags, caps, and reliability lessons see `SERVER_TUNING.md`.

---

## Files

| File | What |
|---|---|
| `pool_proxy.py` | Front proxy: registry discovery, affinity routing, per-backend cap, retry-on-5xx, blacklist on sticky errors, `/pool` + `/pool_metrics`, live `/control`, SIGTERM drain. |
| `pool_monitor.py` | Continuous status CLI. |
| `smoke_test.py` | Offline + live correctness tests. |
| `submit_qwen3.6_35b.sh` | Qwen3.6-35B-A3B-FP8 single-GPU Slurm server. |
| `submit_qwen3.6_35b_dev.sh` | Same model, isolated `vllm_pool_dev/` registry. |
| `submit_qwen3.5_397b.sh` | Qwen3.5-397B-A17B-FP8 TP=4 Slurm server. |
| `submit_gemma3_27b.sh` | Gemma-3-27B-IT single-GPU Slurm server. |
| `submit_qwen3_8b.sh` | Qwen3-8B single-GPU Slurm server. |
| `submit_qwen3_32b.sh` | Qwen3-32B-FP8 single-GPU Slurm server. |
| `submit_llama3_1_8b.sh` | Llama-3.1-8B-Instruct single-GPU Slurm server. |
| `runpod/serve_qwen3.5_397b.sh` | Qwen3.5-397B vLLM launch on a RunPod pod (TP=4). |
| `runpod/serve_qwen3.6_35b.sh` | Qwen3.6-35B vLLM launch on a RunPod pod (TP=1). |
| `runpod/setup_runpod_vllm_env.sh` | One-time vLLM env install on a fresh RunPod pod. |
| `runpod/bench_chat_backends.py` | Pool throughput benchmark. |
| `SERVER_TUNING.md` | Per-model flags, caps, reliability lessons. |

---

## Quickstart

```bash
cd /workspace-vast/adamk/chive && source .env && source .venv/bin/activate
mkdir -p tools/vllm_pool/logs   # gitignored; proxy + sbatch logs land here
```

### Start a proxy

One proxy per model. Pick a port, registry dir, model name:

```bash
nohup .venv/bin/python tools/vllm_pool/pool_proxy.py \
    --port 8999 --proxy-name prod \
    --registry-dir /workspace-vast/$USER/vllm_pool \
    --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --max-seqs-per-upstream 160 \
    --adaptive-cap --adaptive-cap-max 320 \
    > tools/vllm_pool/logs/pool_proxy_prod_8999.out 2>&1 &
```

The proxy scans the registry dir every 15s. Backends appear/disappear as their JSON files come and go. See `SERVER_TUNING.md` for the right `--max-seqs-per-upstream` per model.

### Add Slurm backends

```bash
# Qwen3.6-35B-A3B-FP8 (single GPU each):
sbatch --qos=high --job-name=mu_pool_hi --array=0-7  tools/vllm_pool/submit_qwen3.6_35b.sh
sbatch --qos=low  --job-name=mu_pool_lo --array=0-11 tools/vllm_pool/submit_qwen3.6_35b.sh

# Qwen3.5-397B-A17B-FP8 (TP=4 per backend):
sbatch --qos=high --job-name=mu_pool_397b --array=0-1 tools/vllm_pool/submit_qwen3.5_397b.sh

# Gemma-3-27B-IT (single GPU each):
sbatch --qos=high --job-name=mu_pool_gemma --array=0-2 tools/vllm_pool/submit_gemma3_27b.sh
```

Each script self-registers on `/health` and writes `<jobid>.json` to its registry dir; cleanup trap removes the entry on EXIT/preempt.

### Add a RunPod backend

See **`Detailed RunPod onboarding`** below for the full step-by-step (includes the copy-paste-from-UI block).

### Point the pipeline at the proxy

```json
{ "concurrency": 1500,
  "stage1_llm": "vllm:Qwen/Qwen3.6-35B-A3B-FP8@http://localhost:8999",
  "judge_llm":  "vllm:Qwen/Qwen3.6-35B-A3B-FP8@http://localhost:8999",
  "stage3_llm": "vllm:Qwen/Qwen3.6-35B-A3B-FP8@http://localhost:8999",
  "target_llm": "vllm:Qwen/Qwen3.6-35B-A3B-FP8@http://localhost:8999",
  "stage5_llm": "vllm:Qwen/Qwen3.6-35B-A3B-FP8@http://localhost:8999" }
```

`concurrency` should be **roughly** `n_backends × cap × 2` for a Stage-3-style agentic workload: leaves plenty of in-flight requests to absorb preempt dips, while keeping the proxy's admit queue from starving. With ≤3 backends, 800 is fine; with ≥6, use 1500. Higher is wasted queue depth, not throughput.

---

## Routing

Requests with the same first-user-message hash are pinned to the same backend (SHA-256 over the first ≤512 chars of the first user message). All turns of an agentic investigation, and a request's repeat with different `temperature`/`n`, land on the same backend — so vLLM's prefix cache hits warm KV. If the preferred backend is unhealthy or at cap, request falls back to least-loaded.

Observed: prefix-cache hit rate ~20% in steady state at high concurrency; affinity is more about avoiding KV cache thrash than chasing 100% hit rate.

---

## Detailed RunPod onboarding

A second backend pool that lives on RunPod (cheaper than Slurm `low` QoS for sustained throughput). One pod typically gives 8× H200; for TP=4 models you run 2 vLLM processes per pod (ports 8000 + 8002, the latter through nginx on 8001).

### Step 1 — Generate a random API key

```bash
openssl rand -hex 32
```

Save this somewhere you can paste it into both the pod env vars AND the registry JSON. (vLLM only enforces it if you pass `--api-key`, but our proxy always sends it as `Authorization: Bearer ...` when the registry has `api_key`.)

### Step 2 — Create the RunPod pod (UI)

In the RunPod console, **New Pod**:

1. **Shape:** 8× H200 (or whatever you need for your TP).
2. **Container disk / Volume:** ephemeral is fine for short runs; mount a network volume only if you need model weights to persist across pods.
3. **SSH key:** add `~/.ssh/<your-runpod-key>.pub`.
4. **Expose TCP ports:** 22 (SSH), 8000 (first vLLM), and 8001 (second vLLM via nginx → 8002 internally). If you're only running one vLLM, just 22 + 8000.
5. Launch.

(Don't bother setting env vars in the RunPod UI — the operator-LLM writes `HF_TOKEN` / `VLLM_API_KEY` / `MODEL` directly into the pod's `.env` over SSH in Step 4. Setting them in the UI is redundant and easy to get out of sync with the registry JSON.)

### Step 3 — Copy these from the RunPod UI back to a notepad

After the pod is running, copy:

```
POD_ID=
SSH_HOST=
SSH_PORT=
RUNPOD_PUBLIC_IP=
RUNPOD_TCP_PORT_8000=   # the public TCP that maps to container 8000
RUNPOD_TCP_PORT_8001=   # only if you're running a second vLLM
```

(SSH_HOST and SSH_PORT come from "Connect → SSH over exposed TCP", looks like `ssh root@<ip> -p <port>`.)

### Step 4 — One-time pod setup (from the cluster)

```bash
# scp the repo subtree the pod needs (or git clone on the pod)
ssh -p $SSH_PORT -i ~/.ssh/<your-runpod-key> root@$SSH_HOST 'mkdir -p /root/chive'

scp -P $SSH_PORT -i ~/.ssh/<your-runpod-key> -r \
  tools/vllm_pool/runpod \
  root@$SSH_HOST:/root/chive/tools/vllm_pool/

# Write .env on the pod — this is the ONLY place HF_TOKEN / VLLM_API_KEY / MODEL
# get set (we don't set them in the RunPod UI). The serve script sources this file.
ssh -p $SSH_PORT -i ~/.ssh/<your-runpod-key> root@$SSH_HOST "cat > /root/chive/.env <<EOF
HF_TOKEN=$HF_TOKEN_VALUE
VLLM_API_KEY=$VLLM_API_KEY_VALUE
MODEL=$MODEL
EOF"

# Install vLLM (5-10 min)
ssh -p $SSH_PORT -i ~/.ssh/<your-runpod-key> root@$SSH_HOST \
  'cd /root/chive && bash tools/vllm_pool/runpod/setup_runpod_vllm_env.sh'
```

### Step 5 — Launch vLLM

For a **single-server pod** (one TP=N vLLM on port 8000):

```bash
ssh -p $SSH_PORT -i ~/.ssh/<your-runpod-key> root@$SSH_HOST \
  'cd /root/chive && mkdir -p /root/vllm_logs && \
   nohup env PORT=8000 CUDA_VISIBLE_DEVICES=0,1,2,3 \
     bash tools/vllm_pool/runpod/serve_qwen3.5_397b.sh \
     > /root/vllm_logs/vllm_8000.out 2>&1 < /dev/null &'
```

For a **dual-server pod** (TP=4 + TP=4 on 8 GPUs, exposing 8000 and 8002-via-nginx-8001), launch BOTH:

```bash
# Server 0 on GPUs 0-3, internal port 8000 (direct to container :8000)
ssh -p $SSH_PORT ... \
   'nohup env PORT=8000 CUDA_VISIBLE_DEVICES=0,1,2,3 \
     bash tools/vllm_pool/runpod/serve_qwen3.5_397b.sh \
     > /root/vllm_logs/vllm_8000.out 2>&1 < /dev/null &'

# Server 1 on GPUs 4-7, internal port 8002 (nginx on container :8001 will reverse-proxy)
ssh -p $SSH_PORT ... \
   'nohup env PORT=8002 CUDA_VISIBLE_DEVICES=4,5,6,7 \
     bash tools/vllm_pool/runpod/serve_qwen3.5_397b.sh \
     > /root/vllm_logs/vllm_8002.out 2>&1 < /dev/null &'
```

For dual-server pods you also need to **patch the container's nginx** to forward `:8001 → localhost:8002` and **NOT** convert upstream 502s into HTML-200 (the default RunPod nginx does both wrong things). See **RunPod nginx gotchas** below.

Wait for `Application startup complete` in the log (10-20 min for 397B; faster for 35B).

### Step 6 — Smoke test reasoning ALWAYS (Qwen models)

Qwen reasoning can silently fail (e.g. if `--reasoning-config '{}'` is missing the server still starts but `thinking_token_budget` requests get a 400). Before adding any Qwen backend to the registry:

```bash
API_KEY=<the openssl key>
MODEL=Qwen/Qwen3.5-397B-A17B-FP8
BASE=http://$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_8000/v1

# Verify JSON model list (not a RunPod-nginx HTML 200 fallback)
curl -sS -m 15 -H "Authorization: Bearer $API_KEY" "$BASE/models" | python -m json.tool

# Reasoning smoke: response MUST have non-empty `reasoning` field, and MUST NOT
# error with "thinking_token_budget is set but reasoning_config is not configured"
curl -sS -m 300 \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  "$BASE/chat/completions" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Think briefly, then answer: what is 2+2?\"}],\"max_tokens\":256,\"temperature\":0,\"thinking_token_budget\":64}" \
  | python -m json.tool
```

If `reasoning` is empty or you get the `reasoning_config` error, **do not add the backend** — fix the launcher flags first.

### Step 7 — Write registry JSON

One JSON per backend in the proxy's registry dir. For RunPod, include the `api_key`:

```bash
cat > /workspace-vast/$USER/runpod_397b_pool/runpod_<POD_ID>_8000.json <<EOF
{
  "url": "http://$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_8000/v1",
  "model": "Qwen/Qwen3.5-397B-A17B-FP8",
  "config_hash": "<from the launcher's startup log>",
  "source": "runpod",
  "pod_id": "<POD_ID>",
  "server_id": "runpod-<POD_ID>-0",
  "internal_port": 8000,
  "public_tcp_port": $RUNPOD_TCP_PORT_8000,
  "api_key": "$VLLM_API_KEY"
}
EOF
```

Proxy picks it up within 15s. Check:

```bash
curl -s http://localhost:8999/pool | python -m json.tool
```

### RunPod nginx gotchas

The default RunPod container nginx has two bad defaults that bite our proxy:

1. **`error_page 502 =200 @502;`** silently converts upstream 502s into a successful 200-with-HTML-readme response. The pool proxy's `/v1/models` health check parses the response as JSON, fails, marks the backend unhealthy on every transient overload.
2. **`worker_processes 1`** serializes all requests through a single worker; under sustained load, GET `/v1/models` waits behind long POST `/v1/chat/completions` calls and times out.

To patch (on each pod, only relevant for the dual-server `:8001 → :8002` config):

```bash
# Bump workers
sed -i 's/^worker_processes.*/worker_processes auto;/' /etc/nginx/nginx.conf || \
  sed -i '1i worker_processes auto;' /etc/nginx/nginx.conf

# Bump connections
sed -i 's/worker_connections 2048;/worker_connections 8192;/' /etc/nginx/nginx.conf

# Inside the listen-8001 server { location / { ... } } block, comment out:
#   proxy_intercept_errors on;
#   error_page 502 =200 @502;
# and add:
#   proxy_read_timeout 120s;
#   proxy_send_timeout 120s;
#   proxy_connect_timeout 10s;
#   proxy_http_version 1.1;
#   proxy_set_header Connection "";

nginx -t && nginx -s reload
```

---

## Operating

### Pool health snapshots

```bash
curl -s http://localhost:8999/pool         | python -m json.tool   # per-backend healthy/blacklisted/in_flight
curl -s http://localhost:8999/pool_metrics | python -m json.tool   # + KV usage, preempts, prefix-cache hit %
```

Key fields:
- `n_routable` — healthy AND not blacklisted (this is what's actually serving)
- `kv_cache_usage_avg`, `kv_cache_usage_max` — backend pressure
- `preemptions_per_sec` — non-zero is a tuning signal (lower cap)
- `last_error` per backend — surfaces transient errors without blacklisting

### Continuous monitor

```bash
.venv/bin/python tools/vllm_pool/pool_monitor.py \
  --proxy http://localhost:8999 --interval 30 \
  --progress-jsonl <run_dir>/investigations.jsonl   # optional, for inv/min
```

### Live cap tuning

```bash
curl -X POST http://localhost:8999/control \
  -H 'Content-Type: application/json' \
  -d '{"max_seqs_per_upstream": 110}'
```

Value persists to `proxy_<name>_<port>.control.json` so a proxy restart inherits it. See `SERVER_TUNING.md` for per-model defaults.

### Blacklist & how to clear it

The proxy blacklists a backend after:
- **1 request timeout** (`--request-timeout` default 900s — a timeout already cost a lot of wall time, strong wedge signal)
- **5 cumulative 5xx / network errors** (`--blacklist-after` default)

**Blacklist is sticky** — nothing in the proxy clears it. The proxy's docstring says only a fresh registry write re-admits a backend, which means:

```bash
# Un-blacklist a still-healthy backend (the proxy got spooked by a transient error):
cd /workspace-vast/$USER/runpod_397b_pool
mv runpod_<POD_ID>_8000.json runpod_<POD_ID>_8000.json.tmp_unblacklist
sleep 20   # >15s discovery cycle — proxy de-registers the upstream
mv runpod_<POD_ID>_8000.json.tmp_unblacklist runpod_<POD_ID>_8000.json
# Next 15s cycle, proxy re-discovers it as a fresh upstream with blacklisted=False
```

For Slurm backends, easier: `scancel <jobid>` + a fresh `sbatch` (the cleanup trap removes the registry file; new job writes a new `<jobid>.json`).

A simple watchdog script that auto-runs the `mv`-dance on any blacklisted upstream every couple minutes is the cleanest fix for long-running pools.

### Drain on SIGTERM

```bash
kill -TERM $(cat /workspace-vast/$USER/vllm_pool/proxy_prod_8999.pid)
```

New requests get 503 (`type: pool_proxy_draining`); in-flight requests are awaited up to `--drain-timeout` (default 180s) before the session closes. This is the critical fix for orphan requests — killing the proxy mid-flight without drain leaves vLLM holding sequences whose client is gone, competing with new traffic for KV cache for many minutes.

### Add/remove GPUs while running

```bash
# Add Slurm capacity:
sbatch --qos=low --job-name=mu_pool_lo --array=0-7 tools/vllm_pool/submit_qwen3.6_35b.sh

# Remove by name (safer than scancel <bare-jobid>):
scancel -u $USER --name=mu_pool_lo

# Remove a single backend from routing without killing the server:
mv /workspace-vast/$USER/vllm_pool/<jobid>.json{,.removed}
# (rename back to re-add)
```

---

## Endpoints

- `POST /v1/chat/completions`, `POST /v1/completions` — proxied to a backend
- `GET /v1/models` — union of healthy backends' served models
- `GET /pool` — full per-backend status (healthy/blacklisted/in_flight/route_count/error_count/config_hash) + totals (n_upstreams/n_healthy/n_blacklisted/n_routable)
- `GET /pool_metrics` — aggregated vLLM `/metrics` (vllm_running, vllm_waiting, preemptions, prefix-cache hit%/queries, KV usage) + proxy route counters
- `GET /control`, `POST /control` — live `max_seqs_per_upstream` (persisted to control file)
- `GET /health` — liveness

---

## Registry JSON schema

One JSON per backend in the proxy's registry dir. Required: `url`, `model`. Optional: `config_hash` (warned when divergent), `source` (provenance), `api_key` (proxy sends `Authorization: Bearer <key>`).

Slurm `<jobid>.json` (written by `submit_qwen*.sh`):
```json
{ "url": "http://node-12:58449/v1", "model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "config_hash": "cfff77c2a024541b", "node": "node-12", "port": 58449,
  "qos": "high", "job_id": "1650455" }
```

RunPod (manual):
```json
{ "url": "http://1.2.3.4:55555/v1", "model": "Qwen/Qwen3.5-397B-A17B-FP8",
  "config_hash": "3cb5a3806115a6d0", "source": "runpod",
  "pod_id": "abc123", "server_id": "runpod-abc123-0",
  "internal_port": 8000, "public_tcp_port": 55555,
  "api_key": "..." }
```

---

## Smoke tests

```bash
# Offline (always safe):
python tools/vllm_pool/smoke_test.py --tests consistency

# Live (need a running proxy):
python tools/vllm_pool/smoke_test.py \
  --tests route,affinity,cap_control,metrics_basic \
  --proxy-url http://localhost:8999

# Live + destructive (scancels a backend, resubmits replacement):
python tools/vllm_pool/smoke_test.py --tests churn --proxy-url http://localhost:8999
```

---

## Caveats

- **Co-location:** the proxy is addressed as `localhost:PORT`; the pipeline driver must run on the same host (so localhost resolves and the proxy can reach the backends).
- **Config integrity:** each backend hashes its semantic vLLM flags into `config_hash`. The proxy warns when a member's hash differs from the majority — investigate before adding to production data, because mismatched flags can silently blend behavior (e.g. a backend without `--reasoning-config '{}'` returning a 400 for `thinking_token_budget` requests while the rest serve normally).
- **Cutover risk:** killing the old proxy while a driver is mid-run leaves orphan sequences in vLLM. Drain logic prevents new orphans on clean shutdown, but old orphans (from before the new proxy) can saturate KV. If you must cut over: stop the driver first, wait for proxy `in_flight=0`, then SIGTERM the proxy.
- **Stage-4 long context:** verification prompts can be ~80-100k tokens; default `--max-model-len 81920` (~82k) is tight. Bump it (costs per-GPU concurrency) or run a separate large-context pool for stage 4.
