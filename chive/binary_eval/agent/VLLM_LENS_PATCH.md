# vllm-lens: `output_residual_stream` JSON-decode fix — NOW UPSTREAMED (no manual patch)

**As of 2026-06-09 there is nothing to apply by hand.** The fix below is included
in the maintained fork branch and installed declaratively, so `uv sync` reproduces
it and site-packages is never hand-edited.

## How it's installed
`pyproject.toml` pins vllm-lens to a commit on the fork branch:

```toml
[tool.uv.sources]
vllm-lens = { git = "https://github.com/adamkarvonen/vllm-lens.git", rev = "<commit on adam_dev>" }
```

`uv sync` (or `uv pip install --python <venv> "vllm-lens @ git+https://github.com/adamkarvonen/vllm-lens.git@<rev>"`)
installs it into the project venv. To take new fixes, bump
the `rev` and re-run `uv lock` + `uv sync`. **Restart any running vllm-lens server**
after changing the rev — the plugin is imported at process start.

## What the fix does (for reference)
In `vllm_lens/_activations_plugin.py`, just after `extra = effective_params.extra_args or {}`:

```python
if isinstance(extra.get("output_residual_stream"), str):
    extra["output_residual_stream"] = json.loads(extra["output_residual_stream"])
```

Over the OpenAI HTTP API, `vllm_xargs` values must be scalars, so a single-layer
request is sent as a JSON-encoded string (e.g. `"[31]"`). Without this decode the
worker (which filters layers only when the value `isinstance(..., list)`) captures
**all** layers — ~0.7 MB/token for gemma-3-27b, so large transcripts blow up the
HTTP transfer and `ao_tool` / NLA precompute assert on the wrong shape.

## History
Originally applied by hand (2026-06-05) to the editable fork, but a clean
`vllm-lens==1.1.0` PyPI pin in `pyproject.toml` meant `uv sync` reinstalled the
unpatched wheel into `.venv`, silently reverting it and breaking AO/NLA capture
(captured all 62 layers). Fixed for good by sourcing from the branch instead.
