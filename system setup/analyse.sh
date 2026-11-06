#!/usr/bin/env bash
# analyse.sh · analyse PDFs from the terminal, offline.   ./analyse.sh judgment.pdf   |   ./analyse.sh ~/cases/ --json out.json
set -euo pipefail
cd "$(dirname "$0")"; source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python analyse.py "$@"
