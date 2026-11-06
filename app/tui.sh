#!/usr/bin/env bash
# tui.sh · terminal UI.  Attaches to a running ./run_app.sh server if there is one, otherwise loads the models itself.
#   ./tui.sh                         attach (or load) · ./tui.sh --local   always load in-process
#   ./tui.sh --model gemma3:4b       chat model when loading in-process · ./tui.sh --verifier none   Hyena only
#   ./tui.sh --embed "murder by stabbing"    search the offline library (CPU only) and exit
set -euo pipefail
cd "$(dirname "$0")"; source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python tui.py "$@"
