#!/bin/bash
# Generate a PDF report for each unit in a sorted recording.
# Requires sorted + curated data from run_pipeline.sh or run_sorting.sh + run_curation.sh.
#
# Usage (from project root): bash bin/run_report.sh [config.yaml]

CONFIG=${1:-config.yaml}
SESSION="report_$(basename $CONFIG .yaml)"
mkdir -p logs
LOG="logs/kk_report_$(date +%y%m%d).log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" bash -c \
    "PYTHONUNBUFFERED=1 micromamba run -n si_ks4 python \"$SCRIPT_DIR/pipeline_report.py\" \"$(pwd)/$CONFIG\" 2>&1 | tee >(grep -av $'\\r' >> \"$(pwd)/$LOG\")"

echo "Report started in screen session '$SESSION'."
echo "  Monitor live : screen -r $SESSION"
echo "  Follow log   : tail -f $LOG"
