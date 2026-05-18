#!/bin/bash
# Run the curation pipeline in a detached screen session.
# Requires sorted data from run_sorting.sh to already exist.
# Survives SSH disconnect; reattach anytime with: screen -r curation_<config>
#
# Usage (from project root): bash bin/run_curation.sh [config.yaml]

CONFIG=${1:-config.yaml}
SESSION="curation_$(basename $CONFIG .yaml)"
mkdir -p logs
LOG="logs/kk_curation_$(date +%y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -c \
    "micromamba run -n curation python \"$SCRIPT_DIR/pipeline_curation.py\" \"$(pwd)/$CONFIG\" 2>&1 | tee \"$(pwd)/$LOG\""

echo "Curation started in screen session '$SESSION'."
echo "  Monitor live : screen -r $SESSION"
echo "  Follow log   : tail -f $LOG"