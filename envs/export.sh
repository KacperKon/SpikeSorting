#!/bin/bash
# Run this on the Linux sorting machine to regenerate the environment lock files.
# Commit the resulting .yml files so other users can replicate the exact environment.
#
# Usage: bash envs/export.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Exporting si_ks4 environment..."
micromamba env export -n si_ks4 > "$SCRIPT_DIR/si_ks4.yml"
echo "  -> envs/si_ks4.yml"

echo "Exporting curation environment..."
micromamba env export -n curation > "$SCRIPT_DIR/curation.yml"
echo "  -> envs/curation.yml"

echo "Done. Commit both .yml files to git."
