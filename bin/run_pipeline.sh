#!/bin/bash
# Run the full pipeline (sorting + curation) in a single detached screen session.
# Curation and report always run, even if some recordings failed during sorting
# (they skip recordings where output is missing). The session exits with a
# non-zero code if any stage had failures.
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
    set -o pipefail
    exit_code=0
    PYTHONUNBUFFERED=1 micromamba run -n si_ks4 python \"$SCRIPT_DIR/pipeline_ks4.py\" \"$(pwd)/$CONFIG\" 2>&1 | tee >(grep -av $'\\r' >> \"$(pwd)/$LOG\") || exit_code=\$?
    if [ \$exit_code -ne 0 ]; then
        echo '--- Sorting had failures, proceeding to curation for completed recordings ---' | tee -a \"$(pwd)/$LOG\"
    else
        echo '--- Sorting complete, starting curation ---' | tee -a \"$(pwd)/$LOG\"
    fi
    PYTHONUNBUFFERED=1 micromamba run -n curation python \"$SCRIPT_DIR/pipeline_curation.py\" \"$(pwd)/$CONFIG\" 2>&1 | tee >(grep -av $'\\r' >> \"$(pwd)/$LOG\") || exit_code=\$?
    echo '--- Curation complete, generating report ---' | tee -a \"$(pwd)/$LOG\"
    PYTHONUNBUFFERED=1 micromamba run -n si_ks4 python \"$SCRIPT_DIR/pipeline_report.py\" \"$(pwd)/$CONFIG\" 2>&1 | tee >(grep -av $'\\r' >> \"$(pwd)/$LOG\") || exit_code=\$?
    echo '--- Pipeline complete ---' | tee -a \"$(pwd)/$LOG\"
    exit \$exit_code
"

echo "Pipeline started in screen session '$SESSION'."
echo "  Monitor live : screen -r $SESSION"
echo "  Follow log   : tail -f $LOG"
