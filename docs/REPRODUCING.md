# Reproduction and validation

Commands on this page run from the `clustering/` repository root. The separate sibling `chive/` repository is not required to browse reviews and was not modified.

## Browse saved results

Open [the catalogue](../reviews/index.html) directly. To serve it locally, run:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/reviews/index.html`. No inference, raw download or package installation is needed to read the saved reviews.

## Environment and inputs

The existing `.venv/` and `models/` remain local and ignored. `requirements.lock.txt` preserves the earlier environment snapshot; `requirements.current.txt` records the installed environment at organization time, including the later graph dependencies. Saved protocol/embedding metadata is authoritative for original run versions and model revisions. Python version is recorded in `environment.json`.

For a fresh environment, create a venv with the recorded Python version and install `requirements.current.txt`. Restore raw data using `scripts/download_qwen.py` and the pinned manifests. That downloader historically uses the separate `../chive-env/bin/python`; inspect its documented dependencies before replacing the existing environment. Models must be restored at the revisions recorded in their metadata; model downloads and embedding computation are separate from reading the saved results. `python_with_certificates.sh` supplies the local certifi bundle for HTTPS operations.

## Representative commands

```bash
# Validate the saved six-way comparison (writes its validation report).
.venv/bin/python scripts/validate_petri_checkpoint3.py
# Rebuild only checkpoint-3 presentation from saved results.
.venv/bin/python scripts/run_petri_checkpoint3.py report
# Existing preprocessing tests; free-function checks are also used by validators.
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests
```

Most producing scripts overwrite their named output folders. Commit/checkpoint results before intentionally rerunning them. `run_petri_checkpoint3.py` refuses a completed-run overwrite; its `report` argument regenerates presentation only. Historical commands containing `clustering/<script>.py` now mean `clustering/scripts/<script>.py` when run from the parent workspace. The human documentation has been updated; old commands embedded in preserved generated artifacts and absolute paths in provenance remain records of the original run.

## Reproduce individual analyses

Consult the [method guides](methods/README.md), [script index](../scripts/README.md), [checkpoint-2 protocol](CHECKPOINT2.md), [location protocol](LOCATION_SEED.md) and [research history](history.md). Each run's protocol and summary records describe actual inputs, selection, seeds and metrics. Do not compare the latest-only 269-record analysis with the all-checkpoint 801-record analysis as though only the algorithm changed.

The organization pass updates script root resolution and test imports only; it does not rerun embeddings or change assignments. Historical script hashes identify the original scripts, not the reorganized versions. `archive/artifact_inventory.json` supports independent preservation checks.
