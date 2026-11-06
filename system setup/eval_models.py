"""eval_models.py · run the same chat questions on several chat models and save the answers side by side.

Needs the server running (python -u app.py) and the models pulled in Ollama.
  python eval_models.py judgment.pdf eval_questions.json --models builtin qwen2.5:7b llama3.1:8b

eval_questions.json: [{"role": "neutral|accused|complainant", "q": "question text"}, ...]
Each question starts from a fresh chat (the PDF is re-opened from cache), so earlier answers do not leak into later ones.
Output: eval_results.md (one section per question, one answer per model) + eval_results.json.
"""
import argparse, json, time
from ipc.client import HttpEngine

ap = argparse.ArgumentParser()
ap.add_argument("pdf")
ap.add_argument("questions")
ap.add_argument("--models", nargs="+", default=["builtin", "qwen2.5:7b", "llama3.1:8b"])
ap.add_argument("--url", default="http://127.0.0.1:7860")
ap.add_argument("--out", default="eval_results")
a = ap.parse_args()

if not HttpEngine.alive(a.url):
    raise SystemExit(f"No server at {a.url}. Start it first:  python -u app.py")
eng = HttpEngine(a.url)
qs = json.load(open(a.questions, encoding="utf-8"))
results = {m: [] for m in a.models}

for model in a.models:
    print(f"\n=== {model}")
    eng.set_model(model)
    for i, item in enumerate(qs, 1):
        for ev in eng.analyse(path=a.pdf):              # cached after the first time: resets the chat history
            if ev.get("type") == "error":
                raise SystemExit(ev["message"])
        eng.set_role(item["role"])
        text, route, stats, t0 = [], None, {}, time.time()
        for ev in eng.chat(item["q"], role=item["role"]):
            t = ev.get("type")
            if t == "token":
                text.append(ev["text"])
            elif t == "route":
                route = ev.get("route")
            elif t == "done":
                stats = ev.get("stats", {})
            elif t == "error":
                text.append(f"\n[ERROR] {ev['message']}")
        ans = "".join(text).strip()
        results[model].append({"q": item["q"], "role": item["role"], "route": route, "answer": ans,
                               "seconds": round(time.time() - t0, 1), "tokens_per_s": stats.get("tokens_per_s")})
        print(f"  Q{i:<2} {route or '-':<8} {time.time() - t0:5.1f}s  {ans[:70]!r}")

json.dump(results, open(a.out + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
with open(a.out + ".md", "w", encoding="utf-8") as f:
    f.write(f"# Model comparison · {a.pdf}\n\n")
    for i, item in enumerate(qs):
        f.write(f"## Q{i + 1} · {item['role']}\n\n> {item['q']}\n\n")
        for m in a.models:
            r = results[m][i]
            f.write(f"### {m}  ·  route {r['route']} · {r['seconds']} s · {r['tokens_per_s']} tok/s\n\n{r['answer']}\n\n")
print(f"\nsaved {a.out}.md and {a.out}.json")
