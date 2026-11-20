#!/bin/bash
# Like run_pipeline.sh, but processes one recording at a time end-to-end
# (download -> sort -> curate -> report -> sync -> delete local copy) while the
# next recording is downloaded in the background. Keeps local disk use to ~2 runs.
# See bin/pipeline_streaming.py for details.
# Survives SSH disconnect; reattach anytime with: screen -r stream_<config>
#
# Usage (from project root): bash bin/run_pipeline_streaming.sh [config.yaml]

CONFIG=${1:-config.yaml}
SESSION="stream_$(basename $CONFIG .yaml)"
mkdir -p logs
LOG="logs/kk_pipeline_$(date +%y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -c "
    set -o pipefail
    PYTHONUNBUFFERED=1 micromamba run -n si_ks4 python \"$SCRIPT_DIR/pipeline_streaming.py\" \"$(pwd)/$CONFIG\" 2>&1 | tee >(grep -av $'\\r' >> \"$(pwd)/$LOG\")
    exit_code=\$?
    echo '--- Streaming pipeline complete ---' | tee -a \"$(pwd)/$LOG\"
    exit \$exit_code
"

echo "Streaming pipeline started in screen session '$SESSION'."
echo "  Monitor live : screen -r $SESSION"
echo "  Follow log   : tail -f $LOG"
