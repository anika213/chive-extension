# scripts/

**To reproduce the paper you only need:**

| script | what it does |
|---|---|
| `download_data.py` | pulls the public HF dataset repo into `data/` — run this first |
| `gpu_preflight.sh` | sourced by every Slurm job script; fails fast if an assigned GPU is already occupied |

**Maintainer-side release plumbing** (kept for provenance; not needed to
reproduce anything, and some inputs are not part of the public release):

| script | what it does |
|---|---|
| `hf_layout.py` | single source of truth for how `data/` maps onto the HF dataset repo layout |
| `stage_hf_upload.py` | materializes the HF dataset layout under a staging dir for upload |
| `build_eval_blocklist.py` | built `data/eval_blocklist.json` (train/eval overlap audit); provenance only |
| `filter_eval_blocklist_data.py` | applied the blocklist to the shipped data files |
| `verify_eval_blocklist_clean.py` | independent check that shipped data contains no blocklisted rows |
| `fix_mechanism_concreteness_categories.py` | one-time repair of stringified `categories` lists in the shipped mechanism_concreteness.jsonl files (418/8118 rows) |
