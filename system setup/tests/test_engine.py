"""tests/test_engine.py · end-to-end through Engine.   python -m tests.test_engine judgment.pdf [ollama-model]"""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from ipc.engine import Engine

path = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else "builtin"
e = Engine(model=model); t = time.time(); e.load(); print(f"engine loaded {time.time()-t:.1f}s", e.an.load_seconds)
print("models:", [m["name"] for m in e.models()])

def run(gen, show_tokens=True):
    buf, t0, first = [], time.time(), None
    for ev in gen:
        if ev["type"] == "stage" and ev["status"] == "end": print(f"   stage {ev['name']:10s} {ev.get('ms','')}ms {ev.get('info','') if ev['name'] in ('embed','verify','hyena') else ''}")
        elif ev["type"] == "route": print(f"   ROUTE {ev['route']} p_fail={ev['p_fail']} high={ev['high_signals']} best_rel={ev['best_relevance']} passages={[p['id'] for p in ev['passages']]}")
        elif ev["type"] == "token":
            first = first or time.time() - t0; buf.append(ev["text"])
        elif ev["type"] == "result": print("   RESULT", ev["result"]["sections"], ev["result"]["scores"])
        elif ev["type"] == "error": print("   ERROR", ev["message"])
        elif ev["type"] == "done": print("   done", ev["stats"])
    txt = "".join(buf)
    if txt: print(f"   [first token {first:.1f}s, total {time.time()-t0:.1f}s]\n" + "\n".join("   | " + l for l in txt.splitlines()))
    return txt

print("\n== analyse"); run(e.analyse(path=path))
print("\n== explain"); run(e.explain())
for q in ["Was the applicant granted bail?", "What does section 376 punish?", "When was he arrested?", "What is the capital of France?", "what about it"]:
    print(f"\n== chat: {q}"); run(e.chat(q))
