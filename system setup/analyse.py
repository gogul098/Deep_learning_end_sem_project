"""
analyse.py · analyse judgments from the command line (offline).
  python analyse.py judgment.pdf [more.pdf ...]
  python analyse.py judgment.pdf --show-cited      # keep section numbers already written in the text
  python analyse.py judgment.pdf --verifier none   # Hyena encoder only: no Llama, ~0.5 s, 0.4 GB VRAM
  python analyse.py judgment.pdf --explain         # also write the 4-part explanation with the chat model (--model NAME)
  python analyse.py folder/ --json results.json    # every PDF in a folder, results saved as JSON
"""
import argparse, glob, json, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from ipc.engine import Engine
from ipc.statutes import one_line

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--show-cited", action="store_true", help="do not hide 'Section 302' etc. already in the text")
ap.add_argument("--cpu", action="store_true", help="force CPU")
ap.add_argument("--json", default=None, help="save all results to this file")
ap.add_argument("--verifier", default=None, choices=["llama", "none"], help="none = Hyena encoder only")
ap.add_argument("--role", default="accused", choices=["accused", "complainant", "neutral"], help="lawyer perspective: accused (defence) | complainant (prosecution) | neutral")
ap.add_argument("--explain", action="store_true", help="also stream the strategic explanation/briefing (needs a chat model)")
ap.add_argument("--model", default=None, help="chat model for --explain: builtin | an `ollama list` name | none")
ap.add_argument("--no-cache", action="store_true", help="ignore the on-disk result cache")
a = ap.parse_args()

files = []
for p in a.paths:
    files += sorted(glob.glob(os.path.join(p, "*.pdf")) + glob.glob(os.path.join(p, "*.txt"))) if os.path.isdir(p) else [p]
eng = Engine(model=a.model, verifier=a.verifier, device="cpu" if a.cpu else None, role=a.role).load()
an = eng.an
print(f"models loaded in {sum(an.load_seconds.values()):.1f} s on {an.device}" + (f" · VRAM {an.vram()} GB" if an.vram() else "") + f" · verifier {an.verifier}\n")
out = {}
for f in files:
    res = None
    for ev in eng.analyse(path=f, hide_cited=not a.show_cited, use_cache=not a.no_cache):
        if ev["type"] == "result":
            res = ev["result"]
        elif ev["type"] == "error":
            print("ERROR:", ev["message"]); break
    if res is None:
        continue
    print("=" * 78 + f"\n{os.path.basename(f)}\n{res['doc_tokens']:,} tokens · {res['sentences']} sentences · Llama read {res['llm_tokens']:,} tokens · "
          f"{res['seconds']} s" + (f" · peak VRAM {res['peak_vram_gb']} GB" if res["peak_vram_gb"] else "") + (" · cached" if res["cached"] else ""))
    print("\nAPPLICABLE IPC SECTIONS")
    for s in res["sections"]:
        print(f"  • {s} [{res['scores'][s]:.2f}]: {res['statutes'].get(s, '')}")
        for e in res["evidence"].get(s, [])[:1]:
            print(f"      evidence [{e['id']}]: {e['text'][:200]}")
    if not res["sections"]:
        print("  (none above the threshold)")
    print("\nCANDIDATES CHECKED (encoder · Llama · final)")
    for c, v in res["candidates"].items():
        llm = "  –" if v["llm"] is None else f"{v['llm']:.2f}"
        print(f"  {c:14s} {v['encoder']:.2f} · {llm} · {v['final']:.2f}" + ("  ✓" if c in res["sections"] else ""))
    if a.explain:
        print(f"\nSTRATEGIC BRIEFING ({a.role.upper()} PERSPECTIVE)\n")
        for ev in eng.explain(role=a.role):
            if ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "error":
                print("\nERROR:", ev["message"])
        print()
    print()
    out[os.path.basename(f)] = {k: res[k] for k in ("sections", "scores", "candidates", "doc_tokens", "seconds")} | \
        {"evidence": {c: [e["text"] for e in v] for c, v in res["evidence"].items()}}
print("Decision support only: a qualified lawyer must verify every section.")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1); print("saved", a.json)
