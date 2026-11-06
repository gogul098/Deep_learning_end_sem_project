"""
download_model.py · run ONCE with internet. Downloads everything the app needs into ./models:
  models/final/          your trained v6 model (private HF repo: Hyena encoder + Hyena layers + LoRA + code)
  models/llama/          Llama-3.2-3B-Instruct (base model)
  models/inlegalbert/    InLegalBERT (sentence reader)
After this, the app runs fully OFFLINE.

  python download_model.py --repo <your-hf-username>/hyena-llama-legal-v6 --token hf_xxx
  (a READ token is enough; or run `hf auth login` once and leave out --token)
"""
import argparse, json, os
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="e.g. gogulakrishnan256/hyena-llama-legal-v6")
ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

print("1/3 trained model (final/) …")
snapshot_download(a.repo, allow_patterns=["final/*", "final/**"], local_dir=a.out, token=a.token)
final = os.path.join(a.out, "final")
need = ["hybrid_model.pt", "legal_hybrid.py", "llama_lora"]
missing = [f for f in need if not os.path.exists(os.path.join(final, f))]
if missing:
    raise SystemExit(f"final/ is incomplete ({missing}). Has the Kaggle run finished and pushed 'final'?")

import torch
ck = torch.load(os.path.join(final, "hybrid_model.pt"), map_location="cpu", weights_only=False)
print(f"   ok · version {ck.get('version')} · base {ck['base_model']} · encoder {ck['legal_encoder']}")

print("2/3 Llama base model (~6.5 GB) …")
llama = snapshot_download(ck["base_model"], local_dir=os.path.join(a.out, "llama"),
                          allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*"], token=a.token)
print("3/3 InLegalBERT (~0.5 GB) …")
bert = snapshot_download(ck["legal_encoder"], local_dir=os.path.join(a.out, "inlegalbert"),
                         allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "tokenizer*", "vocab*"], token=a.token)

json.dump({"final": "final", "base_model": "llama", "legal_encoder": "inlegalbert"},
          open(os.path.join(a.out, "paths.json"), "w"), indent=1)
print(f"\ndone · everything is in {os.path.abspath(a.out)}\nnext:  python analyse.py <judgment.pdf>   or   python app.py")
