#!/usr/bin/env bash
# setup_arch.sh · one-time setup of the IPC Section Assistant (v7) on Arch Linux.
#   ./setup_arch.sh                      → installs everything (asks for sudo only for pacman)
#   HF_TOKEN=hf_xxx ./setup_arch.sh      → also downloads the trained model (otherwise run ./download.sh later)
set -euo pipefail
cd "$(dirname "$0")"
say() { printf '\n\033[1;35m▶ %s\033[0m\n' "$*"; }

say "1/5 · system packages (pacman)"
need=()
command -v uv >/dev/null      || need+=(uv)
command -v git >/dev/null     || need+=(git)
if ((${#need[@]})); then sudo pacman -S --needed --noconfirm "${need[@]}"; else echo "uv and git already installed"; fi

say "2/5 · NVIDIA GPU check"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
else
    echo "No working NVIDIA driver found → CPU-only install (slow: minutes per judgment, ~14 GB RAM)."
    echo "For a GPU: sudo pacman -S nvidia-open nvidia-utils (or 'nvidia' for older cards), reboot, run this script again."
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

say "3/5 · Python 3.12 virtual environment (.venv, managed by uv; Arch's system Python is left untouched)"
uv python install 3.12
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

say "4/5 · PyTorch + libraries (PyTorch wheels bring their own CUDA runtime: no system CUDA needed)"
uv pip install torch --index-url "$TORCH_INDEX"
uv pip install -r requirements.txt
python - <<'PY'
import torch
print("torch", torch.__version__, "· CUDA available:", torch.cuda.is_available(),
      "·", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

say "5/5 · trained model"
if [[ -f models/paths.json ]]; then
    echo "model already downloaded (models/)"
elif [[ -n "${HF_TOKEN:-}" ]]; then
    ./download.sh
else
    echo "Next: HF_TOKEN=hf_xxx ./download.sh   (a READ token for your private repo; ~7.5 GB, once)"
fi
echo; echo "Done. Start the web page with ./run_app.sh  ·  analyse a PDF with ./analyse.sh judgment.pdf"
