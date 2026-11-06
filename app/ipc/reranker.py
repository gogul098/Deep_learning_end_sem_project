"""reranker.py · cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2, ~90 MB, CPU), as in prob_reas.

Scores (question, passage) pairs jointly, which separates the passage that ANSWERS a question from passages that merely
share its words far better than vector or keyword similarity. Used twice:
  * to re-order the retrieved candidates (the best 5 of ~12 go to the model),
  * as the router's RerankerScore / ChunkDensity signals - the most informative signals in the prob_reas study
    (information gain 0.24 bits), measured with the SAME model, so the learned cut-offs carry over.
Score = sigmoid(logit), the sentence-transformers CrossEncoder default for this model (0..1).
Missing model folder -> available() is False and the retriever keeps its lexical relevance score.
Download once: python download_reranker.py   (-> models/reranker)
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(os.path.dirname(HERE), "models", "reranker")
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_dir: str = DEFAULT_DIR, threads: int = 4, max_length: int = 512):
        self.model_dir, self.threads, self.max_length = model_dir, threads, max_length
        self._tok = self._model = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return os.path.isfile(os.path.join(self.model_dir, "config.json"))

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(self.threads)
            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            for _ in range(5):                       # same guard as the query embedder (empty 'meta' weights)
                m = AutoModelForSequenceClassification.from_pretrained(self.model_dir, low_cpu_mem_usage=False).eval()
                if not any(p.is_meta for p in m.parameters()):
                    self._model = m
                    break
            if self._model is None:
                raise RuntimeError("reranker loaded with empty weights")

    def score(self, query: str, passages: List[str], batch: int = 16) -> List[float]:
        if not passages:
            return []
        import torch
        with self._lock:
            self._load()
            out = []
            for i in range(0, len(passages), batch):
                x = self._tok([query] * len(passages[i:i + batch]), [p[:2000] for p in passages[i:i + batch]],
                              padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                with torch.no_grad():
                    logits = self._model(**x).logits.squeeze(-1)
                out += torch.sigmoid(logits).reshape(-1).tolist()
            return out


_SHARED: Optional[CrossEncoderReranker] = None


def get_reranker() -> Optional[CrossEncoderReranker]:
    """The shared reranker, or None when the model is not downloaded (or IPC_RERANKER=0)."""
    global _SHARED
    if os.environ.get("IPC_RERANKER", "1") == "0":
        return None
    if _SHARED is None:
        _SHARED = CrossEncoderReranker(os.environ.get("IPC_RERANKER_DIR", DEFAULT_DIR))
    return _SHARED if _SHARED.available() else None
