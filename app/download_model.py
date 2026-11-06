"""
download_model.py · run ONCE with internet. Prepares everything the app needs in ./models:
  trained model          the repo's own ../model_files folder (Hyena encoders + Llama LoRA verifier + legal_hybrid.py)
                         is linked as the app's "final" model, so no private download is needed.
                         (--repo <hf-user>/hyena-llama-legal-v7 downloads models/final/ from a Hugging Face repo instead)
  models/llama/          Llama-3.2-3B-Instruct (base model, the ungated unsloth mirror named in the checkpoint)
  models/inlegalbert/    InLegalBERT (sentence reader)
  models/paths.json      where the app finds the three of them
After this, the app runs fully OFFLINE.

  python download_model.py                                   # uses ../model_files (default)
  python download_model.py --repo <hf-user>/hyena-llama-legal-v7   # optional: trained model from the Hub
  (a token is only needed for a private --repo: set the HF_TOKEN environment variable, or run `hf auth login` once)
"""
import argparse, json, os
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=None, help="optional: Hugging Face repo with final/ (e.g. <user>/hyena-llama-legal-v7)")
ap.add_argument("--final", default=os.path.join(HERE, "..", "model_files"),
                help="folder with the trained model (default: the repo's model_files/)")
ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
ap.add_argument("--out", default=os.path.join(HERE, "models"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

need = ["hybrid_model.pt", "legal_hybrid.py", "llama_lora"]
if a.repo:
    print("1/3 trained model (final/) from", a.repo, "…")
    snapshot_download(a.repo, allow_patterns=["final/*", "final/**"], local_dir=a.out, token=a.token)
    final = os.path.join(a.out, "final")
else:
    final = os.path.abspath(a.final)
    print("1/3 trained model: using", final)
missing = [f for f in need if not os.path.exists(os.path.join(final, f))]
if missing:
    raise SystemExit(f"{final} is incomplete ({missing}). Did `git lfs pull` fetch the model files?")
with open(os.path.join(final, "hybrid_model.pt"), "rb") as f:
    if f.read(7) == b"version":
        raise SystemExit("hybrid_model.pt is a Git LFS pointer, not the model. Run:  git lfs install && git lfs pull")

import torch
ck = torch.load(os.path.join(final, "hybrid_model.pt"), map_location="cpu", weights_only=False)
print(f"   ok · version {ck.get('version')} · base {ck['base_model']} · encoder {ck['legal_encoder']}")

print("2/3 Llama base model (~6.5 GB) …")
snapshot_download(ck["base_model"], local_dir=os.path.join(a.out, "llama"),
                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*"], token=a.token)
print("3/3 InLegalBERT (~0.5 GB) …")
snapshot_download(ck["legal_encoder"], local_dir=os.path.join(a.out, "inlegalbert"),
                  allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "tokenizer*", "vocab*"], token=a.token)

# paths are stored relative to models/ so the whole repo can be moved or cloned elsewhere
json.dump({"final": os.path.relpath(final, a.out).replace(os.sep, "/"), "base_model": "llama", "legal_encoder": "inlegalbert"},
          open(os.path.join(a.out, "paths.json"), "w"), indent=1)
print(f"\ndone · models in {os.path.abspath(a.out)} · trained model in {final}"
      f"\nnext:  python analyse.py <judgment.pdf>   or   python app.py")
