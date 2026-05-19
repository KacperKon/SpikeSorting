#!/bin/bash
# Run the spike sorting pipeline in a detached screen session.
# Survives SSH disconnect; reattach anytime with: screen -r sorting
#
# Usage: bash bin/run_sorting.sh [config.yaml]
#   config.yaml path is relative to where you run this script from (project root).

CONFIG=${1:-config.yaml}
SESSION="sorting_$(basename $CONFIG .yaml)"
mkdir -p logs
LOG="logs/kk_ks4_$(date +%y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

screen -dmS "$SESSION" bash -c \
    "micromamba run -n si_ks4 python \"$SCRIPT_DIR/pipeline_ks4.py\" \"$(pwd)/$CONFIG\" 2>&1 | tee -a \"$(pwd)/$LOG\""

echo "Pipeline started in screen session '$SESSION'."
echo "  Monitor live : screen -r $SESSION"
echo "  Follow log   : tail -f $LOG"