"""tests/test_pipeline.py · score parity + timing of the pipeline.   python -m tests.test_pipeline judgment.pdf"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from ipc.pipeline import Analyzer
from local_model_text import read_document

text = read_document(sys.argv[1])
t = time.time(); a = Analyzer(shared_prefix=False).load(); print(f"load {time.time() - t:.1f}s", a.load_seconds, "vram", a.vram())
ev = []
a.analyse(text, use_cache=False)                       # warm-up (CUDA kernels)
t = time.time(); r0 = a.analyse(text, use_cache=False, emit=ev.append); print(f"plain verifier : {time.time() - t:.1f}s")
a.shared_prefix = True
t = time.time(); r1 = a.analyse(text, use_cache=False, emit=ev.append); print(f"shared prefix  : {time.time() - t:.1f}s  (prefix {r1['shared_prefix_tokens']} tokens)")
print("stages:", {e['name']: e['ms'] for e in ev[-20:] if e['type'] == 'stage' and e['status'] == 'end'})
worst = max(abs(r0["candidates"][c]["llm"] - r1["candidates"][c]["llm"]) for c in r0["candidates"])
print("max |llm score diff| plain vs shared prefix:", round(worst, 4))
print("sections", r1["sections"], r1["scores"], "peak vram", r1["peak_vram_gb"])
t = time.time(); r2 = a.analyse(text); print(f"cached re-run  : {time.time() - t:.2f}s cached={r2['cached']}")
assert worst < 0.02 and r0["sections"] == r1["sections"]
print("pipeline parity OK")
