#!/usr/bin/env bash
# download.sh · one-time setup of the models (needs internet once). The trained model comes from the repo's ../model_files;
# this downloads the Llama-3.2-3B base and InLegalBERT into ./models and links everything in models/paths.json.
#   ./download.sh                                         (no token needed)
#   REPO=<hf-user>/hyena-llama-legal-v7 ./download.sh     (optional: trained model from a Hugging Face repo; private repos read HF_TOKEN from your shell)
set -euo pipefail
cd "$(dirname "$0")"; source .venv/bin/activate
if [ -n "${REPO:-}" ]; then
  exec python download_model.py --repo "$REPO"
fi
exec python download_model.py
