#!/bin/bash
# Project-local HTTPS trust fix; no global Python or system configuration changes.
set -euo pipefail
chive_script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
chive_cert_bundle="$("$chive_script_dir/.venv/bin/python" -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE="$chive_cert_bundle"
export REQUESTS_CA_BUNDLE="$chive_cert_bundle"
exec "$chive_script_dir/.venv/bin/python" "$@"
