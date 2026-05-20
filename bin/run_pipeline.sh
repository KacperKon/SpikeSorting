#!/bin/bash
# Run the full pipeline (sorting + curation) in a single detached screen session.
# Curation only runs if sorting completes successfully.
# Survives SSH disconnect; reattach anytime with: screen -r pipeline_<config>
#
# Usage (from project root): bash bin/run_pipeline.sh [config.yaml]

CONFIG=${1:-config.yaml}
SESSION="pipeline_$(basename $CONFIG .yaml)"
mkdir -p logs
LOG="logs/kk_pipeline_$(date +%y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -c "
    set -e
    PYTHONUNBUFFERED=1 micromamba run -n si_ks4 python \"$SCRIPT_DIR/pipeline_ks4.py\" \"$(pwd)/$CONFIG\" 2>&1 | tee >(grep -av $'\\r' >> \"$(pwd)/$LOG\")
    echo '--- Sorting complete, starting curation ---' | tee -a \"$(pwd)/$LOG\"
    PYTHONUNBUFFERED=1 micromamba run -n curation python \"$SCRIPT_DIR/pipeline_curation.py\" \"$(pwd)/$CONFIG\" 2>&1 | tee >(grep -av $'\\r' >> \"$(pwd)/$LOG\")
    echo '--- Pipeline complete ---' | tee -a \"$(pwd)/$LOG\"
"

echo "Pipeline started in screen session '$SESSION'."
echo "  Monitor live : screen -r $SESSION"
echo "  Follow log   : tail -f $LOG"
