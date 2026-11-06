"""download_reranker.py · one-time download of the cross-encoder reranker (~90 MB) into ./models/reranker (no token needed).

  python download_reranker.py
After this the router measures RerankerScore with the same model as the prob_reas study, and passages are re-ranked.
"""
import os
from huggingface_hub import snapshot_download

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "reranker")
snapshot_download("cross-encoder/ms-marco-MiniLM-L-6-v2", local_dir=out,
                  allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer*", "vocab*"])
if not any(f.endswith(".safetensors") for f in os.listdir(out)):          # older repo layout: PyTorch weights
    snapshot_download("cross-encoder/ms-marco-MiniLM-L-6-v2", local_dir=out, allow_patterns=["pytorch_model.bin"])
print("reranker ->", out)
