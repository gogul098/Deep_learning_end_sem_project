"""bench_models.py · stress test of the chat pipeline on several judgments with one chat model (default Qwen 2.5 7B).

  python bench_models.py                                   # every PDF in test-case-files/, questions from stress_cases.json
  python bench_models.py --folder test-case-files --model qwen2.5:7b --repeat 2 --no-explain

For each judgment: analyse (sections, low-confidence flag) -> /explain once -> every question in a fresh chat ->
a follow-up chain in ONE conversation. Every answer is checked automatically (include / exclude regexes from
stress_cases.json), plus: cut-off answers, citations [S#] that do not exist, route, latency, speed, VRAM, CPU/GPU split.
Output: bench_out/stress_report.md (summary first, then every answer) + bench_out/stress_results.json
"""
import argparse, hashlib, json, os, re, sys, time, urllib.request

os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ipc.engine import Engine, gpu_stats
from ipc import prompts
from ipc.llm import OLLAMA

CITE = re.compile(r"\[?\bS(\d{1,4})\b")


def ollama_split(model: str) -> str:
    """'100% GPU' or e.g. '15%/85% CPU/GPU', from Ollama's /api/ps (what `ollama ps` shows)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=2) as r:
            for m in json.load(r).get("models", []):
                if m.get("name", "").split(":")[0] == model.split(":")[0]:
                    size, vram = m.get("size", 0), m.get("size_vram", 0)
                    if not size:
                        return "?"
                    g = round(vram / size * 100)
                    return "100% GPU" if g >= 99 else f"{100 - g}%/{g}% CPU/GPU"
    except Exception:
        pass
    return "-"


def check(answer: str, inc, exc):
    fails = [f"missing /{p}/" for p in (inc or []) if not re.search(p, answer, re.I)]
    fails += [f"contains /{p}/" for p in (exc or []) if re.search(p, answer, re.I)]
    return fails


def run_stream(events):
    text, route, p_fail, stats, err, t0 = [], None, None, {}, None, time.time()
    for ev in events:
        k = ev.get("type")
        if k == "token":
            text.append(ev["text"])
        elif k == "route":
            route, p_fail = ev.get("route"), ev.get("p_fail")
        elif k == "done":
            stats = ev.get("stats", {}) or stats
        elif k == "error":
            err = ev.get("message")
        elif k == "revise":                       # grounding check rejected the first answer: keep only the revision
            first, text = "".join(text), []
            issues = ev.get("issues", [])
        elif k == "verify":
            left = ev.get("issues", [])
    out = {"answer": "".join(text).strip(), "route": route, "p_fail": p_fail, "stats": stats, "error": err,
           "seconds": round(time.time() - t0, 1)}
    if "first" in locals():
        out.update({"revised": True, "first_answer": first.strip(), "grounding_issues": issues, "unverified": locals().get("left", [])})
    return out


def router_log(eng) -> dict:
    """What the Bayesian router saw for the last question: raw signals, High/Low evidence, bottlenecks (calibration data)."""
    d = eng.last_route or {}
    return {"raw": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else float(v)) for k, v in (d.get("raw") or {}).items()},
            "evidence": d.get("evidence"), "bottlenecks": d.get("bottlenecks"), "p_fail": d.get("p_hallucination_high"),
            "route": d.get("route"), "route_note": d.get("route_note"), "route_action": d.get("route_action"),
            "calibrated": d.get("calibrated", False),
            # what the model was given: passage ids in prompt order (the first max_passages are used) + their start
            "max_passages": d.get("max_passages"),
            "passages": [f"{p.get('id')}: {' '.join(str(p.get('text', '')).split())[:90]}" for p in (d.get("passages") or [])]}


def annotate(r: dict, n_sent: int, limit: int, inc=None, exc=None, expect_clarify: bool = False) -> dict:
    a = r["answer"]
    cites = [int(x) for x in CITE.findall(a)]
    r["citations"] = len(cites)
    r["bad_citations"] = sorted({c for c in cites if c >= n_sent})
    toks = r["stats"].get("tokens") or 0
    r["cut_off"] = bool(a) and (toks >= limit - 2 or not re.search(r"[.!?\])\"'”*:]\s*$", a))
    a = re.sub(r"\n*\[unverified:.*\]\s*$", "", a, flags=re.S)       # the warning itself is not part of the answer
    r["fails"] = check(a, inc, exc) if (inc or exc) else []
    if expect_clarify and r["route"] != "Clarify":
        r["fails"].append(f"vague question answered without clarifying (route {r['route']})")
    if r["error"]:
        r["fails"].append("ERROR " + r["error"])
    if not a and not r["error"]:
        r["fails"].append("empty answer")
    r["ok"] = not r["fails"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=os.path.join(HERE, "test-case-files"))
    ap.add_argument("--cases", default=os.path.join(HERE, "stress_cases.json"))
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--repeat", type=int, default=1, help="ask every question N times (consistency check)")
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "bench_out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    chat_limit = int(os.environ.get("IPC_CHAT_TOKENS", "1000"))
    exp_limit = int(os.environ.get("IPC_EXPLAIN_TOKENS", "1200"))
    cases = json.load(open(a.cases, encoding="utf-8"))["cases"]

    # PDFs, de-duplicated by content
    pdfs, seen = [], {}
    for f in sorted(os.listdir(a.folder)):
        if not f.lower().endswith((".pdf", ".txt", ".md")):
            continue
        p = os.path.join(a.folder, f)
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        if h in seen:
            print(f"skip {f[:12]}…  (identical to {seen[h][:12]}…)")
            continue
        seen[h] = f
        case = next((c for c in cases if f.startswith(c["match"])), None)
        if case is None:
            print(f"skip {f[:12]}…  (no questions for it in {os.path.basename(a.cases)})")
            continue
        pdfs.append((p, case))

    eng = Engine(model=a.model)
    eng.load(emit=lambda e: print("  ", e["message"]) if e.get("message") else None)
    eng.set_model(a.model)
    print(f"model {a.model} · {len(pdfs)} judgments · repeat {a.repeat}\n")
    T0, docs = time.time(), []

    for path, case in pdfs:
        name = os.path.basename(path)
        print(f"== {case['title']}")
        doc = {"file": name, "title": case["title"], "questions": [], "chain": [], "explain": None}
        t = time.time()
        for ev in eng.analyse(path=path, use_cache=True):
            if ev.get("type") == "error":
                raise SystemExit(ev["message"])
        pub = eng.public(eng.res)
        n_sent = len(eng.res["sentences_list"])
        doc["analysis"] = {"seconds": round(time.time() - t, 1), "sections": pub["sections"],
                           "scores": {s: round(v, 3) for s, v in pub["scores"].items()},
                           "weak": pub.get("weak_prediction"), "sentences": n_sent, "doc_tokens": pub["doc_tokens"]}
        print(f"   sections {pub['sections']} weak={pub.get('weak_prediction')} · {n_sent} sentences")

        if not a.no_explain:
            r = run_stream(eng.explain(role="neutral"))
            doc["explain"] = annotate(r, n_sent, exp_limit)
            doc["explain"]["complete"] = prompts.is_complete(r["answer"])
            print(f"   explain {r['seconds']}s · {len(r['answer'].split())} words · complete={doc['explain']['complete']}")

        for q in case["questions"]:
            for k in range(a.repeat):
                eng.clear_chat()
                r = annotate(run_stream(eng.chat(q["q"], role=q["role"])), n_sent, chat_limit, q.get("include"), q.get("exclude"),
                             q.get("expect_clarify", False))
                r["router"] = router_log(eng)
                r.update({"id": q["id"], "role": q["role"], "tests": q.get("tests", ""), "q": q["q"], "run": k + 1,
                          "vram_gb": gpu_stats().get("used_gb"), "split": ollama_split(a.model)})
                doc["questions"].append(r)
                tps = r["stats"].get("tokens_per_s")
                print(f"   {q['id']:<4} {'PASS' if r['ok'] else 'FAIL'} {r['route'] or '-':7} {r['seconds']:>5}s "
                      f"{tps or '-':>5} tok/s  {'CUT ' if r['cut_off'] else ''}{'; '.join(r['fails'])[:90]}")

        eng.clear_chat()                                       # follow-up chain: ONE conversation
        for i, c in enumerate(case.get("chain", []), 1):
            r = annotate(run_stream(eng.chat(c["q"], role="neutral")), n_sent, chat_limit, c.get("include"), c.get("exclude"))
            r.update({"id": f"{case['questions'][0]['id'][0]}-chain{i}", "q": c["q"], "router": router_log(eng)})
            doc["chain"].append(r)
            print(f"   chain{i} {'PASS' if r['ok'] else 'FAIL'} {r['seconds']:>5}s  {'; '.join(r['fails'])[:90]}")
        docs.append(doc)
        json.dump({"model": a.model, "embedder": "inlegalbert", "calibration": getattr(eng.retriever, "calibration_file", None),
                   "docs": docs}, open(os.path.join(a.out, "stress_results.json"), "w"), indent=1, default=str)

    write_report(a, docs, round(time.time() - T0))
    print(f"\nreport -> {os.path.join(a.out, 'stress_report.md')}")


def router_table(items):
    """Is P(fail) informative? Mean P(fail) for answers that passed vs failed, and what each route led to."""
    rows = [q for q in items if q.get("router") and q["router"].get("p_fail") is not None]
    if not rows:
        return "(no router data)"
    ok = [q["router"]["p_fail"] for q in rows if q["ok"]]
    bad = [q["router"]["p_fail"] for q in rows if not q["ok"]]
    mean = lambda v: round(sum(v) / len(v), 3) if v else "-"
    pairs = [(b, g) for b in bad for g in ok]                       # AUC: P(a failed answer got a higher P(fail) than a passed one)
    auc = round(sum(1.0 if b > g else 0.5 if b == g else 0.0 for b, g in pairs) / len(pairs), 3) if pairs else "-"
    brier = round(sum((q["router"]["p_fail"] - (0 if q["ok"] else 1)) ** 2 for q in rows) / len(rows), 3)
    L = [f"Calibrated: **{'yes' if rows[0]['router'].get('calibrated') else 'no (hand-set network)'}** · "
         f"mean P(fail) on passed answers {mean(ok)}, on failed answers {mean(bad)} · AUC {auc} (0.5 = no better than chance) · Brier {brier}", "",
         "| Route | Answers | Passed |", "|---|---|---|"]
    for route in ("Serve", "Search", "Clarify"):
        g = [q for q in rows if q["router"].get("route") == route]
        if g:
            L.append(f"| {route} | {len(g)} | {sum(q['ok'] for q in g)} |")
    return "\n".join(L)


def explain_cell(ex):
    if not ex:
        return "-"
    return ("complete" if ex.get("complete") else "INCOMPLETE") + f", {ex['seconds']} s"


def write_report(a, docs, total_s):
    allq = [q for d in docs for q in d["questions"]]
    chains = [c for d in docs for c in d["chain"]]
    tps = [q["stats"].get("tokens_per_s") for q in allq if q["stats"].get("tokens_per_s")]
    ft = [q["stats"].get("first_token_s") for q in allq if q["stats"].get("first_token_s")]
    pct = lambda n, d: f"{n}/{d} ({round(100 * n / d) if d else 0}%)"
    L = [f"# Stress test · {a.model}", "",
         f"{len(docs)} judgments · {len(allq)} answers · {len(chains)} follow-ups · total {total_s // 60} min {total_s % 60} s", "",
         "## Summary", "", "| | |", "|---|---|",
         f"| Answers passing all checks | **{pct(sum(q['ok'] for q in allq), len(allq))}** |",
         f"| Follow-up chain passing | {pct(sum(c['ok'] for c in chains), len(chains))} |",
         f"| Cut-off answers | {sum(q['cut_off'] for q in allq)} |",
         f"| Answers with no [S#] citation | {sum(q['citations'] == 0 for q in allq)} |",
         f"| Citations to sentences that do not exist | {sum(len(q['bad_citations']) for q in allq)} |",
         f"| Errors | {sum(bool(q['error']) for q in allq)} |",
         f"| Revised by the grounding check | {sum(bool(q.get('revised')) for q in allq)} (still unverified after revision: {sum(bool(q.get('unverified')) for q in allq)}) |",
         f"| Speed (median) | {sorted(tps)[len(tps) // 2] if tps else '-'} tok/s |",
         f"| First token (median / max) | {sorted(ft)[len(ft) // 2] if ft else '-'} s / {max(ft) if ft else '-'} s |",
         f"| Peak VRAM seen | {max((q['vram_gb'] or 0) for q in allq) if allq else '-'} GB |",
         f"| Ollama placement | {', '.join(sorted({q['split'] for q in allq}))} |", "",
         "## Router", "", router_table(allq + chains), "",
         "## Per judgment", "", "| Judgment | Sections (weak?) | Explain | Pass | Traps passed | Chain |", "|---|---|---|---|---|---|"]
    for d in docs:
        qs = d["questions"]
        traps = [q for q in qs if "TRAP" in q["tests"]]
        ex = d["explain"]
        L.append(f"| {d['title']} | {', '.join(d['analysis']['sections']) or '-'} ({'weak' if d['analysis']['weak'] else 'ok'}) | "
                 f"{explain_cell(ex)} | "
                 f"{pct(sum(q['ok'] for q in qs), len(qs))} | {pct(sum(q['ok'] for q in traps), len(traps))} | "
                 f"{pct(sum(c['ok'] for c in d['chain']), len(d['chain']))} |")
    L += ["", "## Failed checks", "", "| Q | Tests | Route | Why |", "|---|---|---|---|"]
    for q in allq:
        if not q["ok"] or q["cut_off"] or q["bad_citations"]:
            why = q["fails"] + (["cut off"] if q["cut_off"] else []) + ([f"bad cites {q['bad_citations']}"] if q["bad_citations"] else [])
            L.append(f"| {q['id']}{'' if a.repeat == 1 else ' #' + str(q['run'])} | {q['tests']} | {q['route']} | {'; '.join(why)} |")
    for d in docs:
        L += ["", f"## {d['title']}", ""]
        if d["explain"]:
            L += ["### Explanation (neutral)", "", d["explain"]["answer"] or f"*(error: {d['explain']['error']})*", ""]
        for q in d["questions"]:
            s = q["stats"]
            L += [f"### {q['id']} · {q['role']} · {q['tests']} · {'PASS' if q['ok'] else 'FAIL'}",
                  f"> {q['q']}", "",
                  f"*route {q['route']} · P(fail) {q['p_fail']} · {q['seconds']} s · first token {s.get('first_token_s')} s · "
                  f"{s.get('tokens_per_s')} tok/s · prompt {s.get('prompt_tokens')} tok · {q['split']}*", "",
                  q["answer"] or f"*(no answer: {q['error']})*", ""]
            if q["fails"]:
                L += [f"**Failed:** {'; '.join(q['fails'])}", ""]
            if q.get("revised"):
                L += [f"**Grounding check revised this answer.** Problems in the first version: {' '.join(q['grounding_issues'])}", "",
                      "<details><summary>first version</summary>", "", q["first_answer"], "", "</details>", ""]
        if d["chain"]:
            L += ["### Follow-up chain (one conversation)", ""]
            for c in d["chain"]:
                L += [f"**{c['q']}** {'PASS' if c['ok'] else 'FAIL'}", "", c["answer"], ""]
    open(os.path.join(a.out, "stress_report.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
