#!/bin/bash
# Behavior-similarity appendix: grade -> analyze -> spot check. CPU/API only.
set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
PY=python
DIR=experiments/model_behavior_similarity
mkdir -p "${CHIVE_DATA:-data}/experiments/model_behavior_similarity/results" "$DIR/plots" "$DIR/logs"

$PY "$DIR/grade_cross.py" 2>&1 | tee "$DIR/logs/grade_cross.log"
$PY "$DIR/analyze.py" 2>&1 | tee "$DIR/logs/analyze.log"
$PY "$DIR/make_spot_check.py" 2>&1 | tee "$DIR/logs/spot_check.log"
