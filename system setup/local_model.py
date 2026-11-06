"""local_model.py · loads the trained v7 model from ./models, fully offline. Used by analyse.py, app.py, and tui.py."""
import os, sys, json, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("LEGAL_MODELS_DIR", os.path.join(HERE, "models"))


def read_document(path: str) -> str:
    if path.lower().endswith(".pdf"):
        import pymupdf
        with pymupdf.open(path) as d:
            return "\n".join(p.get_text() for p in d)
    return open(path, encoding="utf-8", errors="ignore").read()


def load(models_dir: str = MODELS, device: str = None, embedder_on_cpu: bool = None, per_section: bool = False):
    """Returns (predict, info). predict(text, hide_cited=True) -> dict (sections, scores, candidates, evidence, ...)."""
    paths = json.load(open(os.path.join(models_dir, "paths.json")))
    p = {k: v if os.path.isabs(v) else os.path.join(models_dir, v) for k, v in paths.items()}
    sys.path.insert(0, p["final"])
    import legal_hybrid as LH                                   # the exact code the model was trained with
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu = device.startswith("cuda")
    if not gpu:
        print("⚠ no NVIDIA GPU found: running on CPU (needs ~14 GB RAM; a judgment takes a few minutes)")
    if embedder_on_cpu is None:                                  # small GPUs: InLegalBERT reads on the CPU
        embedder_on_cpu = gpu and torch.cuda.get_device_properties(0).total_memory < 7 * 1024**3
    t0 = time.time()
    tok, llm, encoders, ck = LH.load_trained(p["final"], device, base_model=p["base_model"], legal_encoder=p["legal_encoder"],
                                             load_4bit=gpu, embedder_device="cpu" if embedder_on_cpu else None)
    if not per_section:                                          # one threshold generalised best on the test set
        ck["inference"]["class_thresholds"] = None
    pred = LH.LegalPredictor(tok, llm, encoders, ck)
    info = {
        "device": torch.cuda.get_device_name(0) if gpu else "CPU",
        "load_s": round(time.time() - t0, 1),
        "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2) if gpu else None,
        "embedder": "CPU" if (embedder_on_cpu or not gpu) else "GPU",
        "inference": ck["inference"],
        "desc": ck["statute_desc"],
        "tok": tok,
        "llm": llm,
        "ck": ck,
        "pred": pred,
    }

    def predict(text: str, hide_cited: bool = True):
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        r = pred(text, hide_cited=hide_cited)
        r["seconds"] = round(time.time() - t, 1)
        r["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2) if gpu else None
        return r

    return predict, info
