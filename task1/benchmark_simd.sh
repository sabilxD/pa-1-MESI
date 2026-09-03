#!/usr/bin/env bash
set -euo pipefail

ENV_ACTIVATE="$HOME/general_env/bin/activate"

if [[ -f "$ENV_ACTIVATE" ]]; then
    # Activate user's python environment
    source "$ENV_ACTIVATE"
fi

python3 evaluate_simd.py "$@"
