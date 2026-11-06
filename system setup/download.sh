#!/usr/bin/env bash
# download.sh · one-time download of the trained v7 model, Llama-3.2-3B and InLegalBERT into ./models (needs internet once).
#   HF_TOKEN=hf_xxx ./download.sh                    (repo defaults to <you>/hyena-llama-legal-v7)
#   HF_TOKEN=hf_xxx REPO=user/hyena-llama-legal-v7 ./download.sh
set -euo pipefail
cd "$(dirname "$0")"; source .venv/bin/activate
HF_TOKEN="${HF_TOKEN:-<your-hf-token>}"
export HF_TOKEN
REPO="${REPO:-$(python -c "from huggingface_hub import whoami; import os; print(whoami(token=os.environ['HF_TOKEN'])['name'])")/hyena-llama-legal-v7}"
python download_model.py --repo "$REPO" --token "$HF_TOKEN"
