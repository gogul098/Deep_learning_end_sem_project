#!/usr/bin/env bash
# run_app.sh · start the offline server → http://127.0.0.1:7860   (web app + /architecture explainer + API for the TUI)
#   ./run_app.sh                              built-in chat model
#   ./run_app.sh --model gemma3:4b            any name from `ollama list`
#   ./run_app.sh --verifier none              Hyena encoder only (no Llama in memory)
#   ./run_app.sh --lan                        serve the office network (open port 7860 in your firewall)
set -euo pipefail
cd "$(dirname "$0")"; source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python -u app.py "$@"
