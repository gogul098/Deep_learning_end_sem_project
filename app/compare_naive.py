"""compare_naive.py · our architecture vs. a naive offline LLM (the same Qwen2.5-7B, given the judgment text directly).

Both systems answer the SAME generated questions (ipc_qa/questions.jsonl: parties, judge, date, outcome, charges, FIR facts,
trap questions whose answer is not in the judgment, vague questions, 3-step follow-up chains) on the SAME judgments, graded by
the SAME automatic grader as the router labels (label_ipc.grade).

  naive   Qwen2.5-7B through Ollama. Prompt = the judgment text + the question (chains: the earlier turns too).
          --ctx sets the context window. A judgment longer than the window is cut in the middle (start and end kept:
          parties, judge and the final order live there), which is kinder than Ollama's default of dropping the start.
  ours    the full app: Hyena + Llama verifier (sections), InLegalBERT retrieval + reranker, Bayesian router (Serve /
          Search / Clarify), the same Qwen2.5-7B as the writer, grounding check. Exactly what a lawyer gets in the app.

  python compare_naive.py ipc_qa/questions.jsonl --system naive --ctx 32768 --limit 60
  python compare_naive.py ipc_qa/questions.jsonl --system naive --ctx 8192  --limit 60
  python compare_naive.py ipc_qa/questions.jsonl --system ours              --limit 60
  python compare_naive.py --stress --system naive --ctx 8192      # the 57 hand-written lawyer questions (test-case-files/)
  python compare_naive.py --stress --system ours
  python compare_naive.py ipc_qa/questions.jsonl --memory --system naive --ctx 8192 --limit 10   # long conversations
  python compare_naive.py ipc_qa/questions.jsonl --memory --system ours --limit 10
  python compare_naive.py --report compare/                       # tables + chart -> compare/REPORT.md, compare/compare.png

Resumable: finished question ids are skipped. --limit N takes N judgments spread evenly over the set (same N = same judgments).
"""
from __future__ import annotations

import argparse, glob, hashlib, json, os, re, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

NAIVE_SYSTEM = ("You are a legal assistant. Answer the user's question about the Indian court judgment given below, using only that "
                "judgment and anything the user tells you in this conversation. If the judgment does not state the answer, say \"not stated in the document\". Be concise.")
CLARIFY_BACK = re.compile(r"\b(?:could you|can you|please)\s+(?:clarify|specify|be more specific)|which (?:\w+\s){0,4}do you mean|"
                          r"what do you mean|more specific", re.I)


def pick_docs(qs, limit):
    docs = sorted({q["doc"] for q in qs})
    if not limit or limit >= len(docs):
        return docs
    step = len(docs) / limit
    return [docs[int(i * step)] for i in range(limit)]


NAIVE_TAG = "naive"
TRIM = "middle"            # middle: keep the start and end (kind to the baseline) · tail: keep only the end, as Ollama does by default


def TRIM_SET(v):
    globals()["TRIM"] = v


def fit(text: str, ctx: int, reserve: int = 900) -> tuple:
    """Cut a judgment that does not fit the window (about 3.5 characters per token for legal English)."""
    budget = max(2000, int((ctx - reserve) * 3.5))
    if len(text) <= budget:
        return text, False
    if TRIM == "tail":
        return text[-budget:], True
    head = int(budget * 0.6)
    return text[:head] + "\n\n[... middle of the judgment omitted: it does not fit the context window ...]\n\n" + text[-(budget - head):], True


def fix_judge_gold(by_doc, docs, pdfs):
    """'Which judge authored this?': the metadata lists the bench, so take the author from the judgment itself (as label_ipc --regrade)."""
    from label_ipc import author_from_pdf
    for d in docs:
        au = None
        for q in by_doc[d]:
            if q["type"] == "meta_judge" or (q.get("chain") and "authored" in q.get("question", "")):
                au = au or author_from_pdf(os.path.join(pdfs, d + ".pdf"))
                if au:
                    q["answers"] = [au] + ([au.split()[-1]] if len(au.split()[-1]) >= 5 else [])
                    q["author_from_pdf"] = True


def grade_row(reply: str, q: dict, asked_back: bool) -> dict:
    from label_ipc import grade
    g = grade(reply, q)
    if q["grade"] == "vague":               # nothing can be answered: the right move is to ask the user what they mean
        g["correct"] = bool(asked_back)
    return g


# ───────────────────────────── naive: the model alone ─────────────────────────────
def run_naive(qs_by_doc, docs, pdfs, model, ctx, out):
    import requests
    from local_model_text import read_document
    url = __import__("ipc.llm", fromlist=["ollama_url"]).ollama_url()
    if not url.startswith("http"):
        url = "http://" + url
    done = _done(out)
    f = open(out, "a", encoding="utf-8")
    for di, doc in enumerate(docs, 1):
        pending = [q for q in qs_by_doc[doc] if q["id"] not in done]
        path = os.path.join(pdfs, f"{doc}.pdf")
        if not pending or not os.path.exists(path):
            continue
        text, cut = fit(read_document(path), ctx)
        judgment = f"JUDGMENT:\n{text}"

        def ask(history, q):
            msgs = [{"role": "system", "content": NAIVE_SYSTEM}, {"role": "user", "content": judgment + "\n\nQUESTION: " + q["question"]}]
            if history:                       # a follow-up chain: the judgment once, then the conversation
                msgs = [{"role": "system", "content": NAIVE_SYSTEM}, {"role": "user", "content": judgment + "\n\nQUESTION: " + history[0][0]},
                        {"role": "assistant", "content": history[0][1]}]
                for hq, ha in history[1:]:
                    msgs += [{"role": "user", "content": hq}, {"role": "assistant", "content": ha}]
                msgs.append({"role": "user", "content": q["question"]})
            t0 = time.time()
            r = requests.post(f"{url}/api/chat", json={"model": model, "messages": msgs, "stream": False,
                                                        "options": {"num_ctx": ctx, "temperature": 0, "num_predict": 400}}, timeout=900)
            r.raise_for_status()
            j = r.json()
            reply = j.get("message", {}).get("content", "").strip()
            return reply, round(time.time() - t0, 1), j.get("prompt_eval_count")

        rows = []
        singles = [q for q in pending if not q.get("chain")]
        chains = defaultdict(list)
        for q in pending:
            if q.get("chain"):
                chains[q["chain"]].append(q)
        for q in singles:
            reply, secs, ptok = ask([], q)
            rows.append(_row(q, reply, secs, NAIVE_TAG, asked_back=bool(CLARIFY_BACK.search(reply)),
                             extra={"truncated": cut, "prompt_tokens": ptok}))
        for cid, cq in chains.items():
            hist = []
            for q in sorted(cq, key=lambda x: x["step"]):
                reply, secs, ptok = ask(hist, q)
                hist.append((q["question"], reply))
                rows.append(_row(q, reply, secs, NAIVE_TAG, asked_back=bool(CLARIFY_BACK.search(reply)),
                                 extra={"truncated": cut, "prompt_tokens": ptok}))
        _write(f, rows, di, len(docs), doc)
    f.close()


# ───────────────────────────── ours: the full app ─────────────────────────────
def run_ours(qs_by_doc, docs, pdfs, model, out):
    os.environ["IPC_FORCE_SERVE"] = "0"                       # the real router decides (label_ipc forces serving)
    os.environ.setdefault("IPC_CASES_DIR", os.path.join(os.path.dirname(os.path.abspath(out)), "cases"))
    import label_ipc
    os.environ["IPC_FORCE_SERVE"] = "0"
    from ipc.engine import Engine
    eng = Engine(model=model)
    eng.load(emit=lambda e: print("  ", e["message"]) if e.get("message") else None)
    eng.set_model(model)
    done = _done(out)
    f = open(out, "a", encoding="utf-8")
    for di, doc in enumerate(docs, 1):
        pending = [q for q in qs_by_doc[doc] if q["id"] not in done]
        path = os.path.join(pdfs, f"{doc}.pdf")
        if not pending or not os.path.exists(path):
            continue
        t_an = time.time()
        err = next((ev["message"] for ev in eng.analyse(path=path, use_cache=True) if ev.get("type") == "error"), None)
        if err:
            print(f"  ! {doc}: {err}")
            continue
        t_an = round(time.time() - t_an, 1)

        def one(q):
            r = label_ipc.ask(eng, q, model)
            route = r.get("route_router") or ""
            asked_back = route == "Clarify"
            g = grade_row(r["reply"], q, asked_back)
            return {**{k: r[k] for k in ("id", "doc", "type", "question", "answers", "grade", "chain", "step", "reply", "timing_s")},
                    **g, "system": "ours", "route": route, "asked_back": asked_back,
                    "cited": bool(re.search(r"\[S\d+", r["reply"])), "grounding_revised": r.get("grounding_revised"),
                    "gold_retrieved": r.get("gold_retrieved"), "analyse_s": t_an}

        rows = []
        for q in [q for q in pending if not q.get("chain")]:
            eng.clear_chat()
            rows.append(one(q))
        chains = defaultdict(list)
        for q in pending:
            if q.get("chain"):
                chains[q["chain"]].append(q)
        for cid, cq in chains.items():
            eng.clear_chat()
            for q in sorted(cq, key=lambda x: x["step"]):
                rows.append(one(q))
        _write(f, rows, di, len(docs), doc)
    f.close()


def _row(q, reply, secs, system, asked_back=False, extra=None):
    g = grade_row(reply, q, asked_back)
    return {**{k: q[k] for k in ("id", "doc", "type", "question", "answers", "grade", "chain", "step")}, "reply": reply,
            "timing_s": secs, **g, "system": system, "asked_back": asked_back, "cited": bool(re.search(r"\[S\d+", reply)),
            **(extra or {})}


def _done(out):
    if not os.path.exists(out):
        return set()
    return {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}


def _write(f, rows, di, n, doc):
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.flush()
    if rows:
        print(f"[{di}/{n}] {doc}: {sum(r['correct'] for r in rows)}/{len(rows)} correct · {rows[0]['system']}", flush=True)


# ───────────────────────────── stress set: hand-written lawyer questions ─────────────────────────────
ROLE_NOTE = {"accused": "You are assisting the DEFENCE counsel (the accused / applicant is the client). ",
             "complainant": "You are assisting the PROSECUTION / complainant's counsel. ", "neutral": ""}


def stress_items(cases_file, folder):
    cases = json.load(open(cases_file, encoding="utf-8"))["cases"]
    out = []
    for c in cases:
        f = next((os.path.join(folder, x) for x in sorted(os.listdir(folder)) if x.startswith(c["match"])), None)
        if f:
            out.append((f, c))
    return out


def stress_check(answer, inc, exc):
    fails = [f"missing /{p}/" for p in (inc or []) if not re.search(p, answer, re.I)]
    return fails + [f"contains /{p}/" for p in (exc or []) if re.search(p, answer, re.I)]


def run_stress(system, model, ctx, cases_file, folder, out):
    items = stress_items(cases_file, folder)
    rows = []
    if system == "naive":
        import requests
        from local_model_text import read_document
        url = __import__("ipc.llm", fromlist=["ollama_url"]).ollama_url()
        url = url if url.startswith("http") else "http://" + url

        def chat(msgs):
            t0 = time.time()
            r = requests.post(f"{url}/api/chat", json={"model": model, "messages": msgs, "stream": False,
                                                        "options": {"num_ctx": ctx, "temperature": 0, "num_predict": 700}}, timeout=900)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "").strip(), round(time.time() - t0, 1)

        for path, case in items:
            text, cut = fit(read_document(path), ctx)
            judg = f"JUDGMENT:\n{text}"
            for q in case["questions"]:
                sysp = ROLE_NOTE.get(q["role"], "") + NAIVE_SYSTEM
                ans, secs = chat([{"role": "system", "content": sysp}, {"role": "user", "content": judg + "\n\nQUESTION: " + q["q"]}])
                fails = stress_check(ans, q.get("include"), q.get("exclude"))
                if q.get("expect_clarify") and not CLARIFY_BACK.search(ans):
                    fails.append("vague question answered without asking what the user means")
                rows.append({"id": q["id"], "case": case["title"], "tests": q.get("tests", ""), "role": q["role"], "q": q["q"],
                             "answer": ans, "fails": fails, "ok": not fails, "seconds": secs, "truncated": cut})
            hist = [{"role": "system", "content": NAIVE_SYSTEM}]
            for i, c in enumerate(case.get("chain", []), 1):
                hist.append({"role": "user", "content": (judg + "\n\nQUESTION: " if i == 1 else "") + c["q"]})
                ans, secs = chat(hist)
                hist.append({"role": "assistant", "content": ans})
                fails = stress_check(ans, c.get("include"), c.get("exclude"))
                rows.append({"id": f"{case['questions'][0]['id'][0]}-chain{i}", "case": case["title"], "tests": "follow-up chain",
                             "role": "neutral", "q": c["q"], "answer": ans, "fails": fails, "ok": not fails, "seconds": secs, "truncated": cut})
            print(f"{case['title'][:60]}: {sum(r['ok'] for r in rows if r['case'] == case['title'])} passed", flush=True)
    else:
        os.environ["IPC_FORCE_SERVE"] = "0"
        os.environ.setdefault("IPC_CASES_DIR", os.path.join(os.path.dirname(os.path.abspath(out)), "cases"))
        from ipc.engine import Engine
        eng = Engine(model=model)
        eng.load(emit=lambda e: print("  ", e["message"]) if e.get("message") else None)
        eng.set_model(model)

        def ask(qtext, role):
            t0, text, route = time.time(), [], None
            for ev in eng.chat(qtext, role=role):
                k = ev.get("type")
                if k == "token":
                    text.append(ev["text"])
                elif k == "revise":
                    text = []
                elif k == "route":
                    route = ev.get("route")
            return "".join(text).strip(), route, round(time.time() - t0, 1)

        for path, case in items:
            for ev in eng.analyse(path=path, use_cache=True):
                if ev.get("type") == "error":
                    raise SystemExit(ev["message"])
            for q in case["questions"]:
                eng.clear_chat()
                ans, route, secs = ask(q["q"], q["role"])
                fails = stress_check(ans, q.get("include"), q.get("exclude"))
                if q.get("expect_clarify") and route != "Clarify":
                    fails.append("vague question answered without asking what the user means")
                rows.append({"id": q["id"], "case": case["title"], "tests": q.get("tests", ""), "role": q["role"], "q": q["q"],
                             "answer": ans, "route": route, "fails": fails, "ok": not fails, "seconds": secs,
                             "cited": bool(re.search(r"\[S\d+", ans))})
            eng.clear_chat()
            for i, c in enumerate(case.get("chain", []), 1):
                ans, route, secs = ask(c["q"], "neutral")
                fails = stress_check(ans, c.get("include"), c.get("exclude"))
                rows.append({"id": f"{case['questions'][0]['id'][0]}-chain{i}", "case": case["title"], "tests": "follow-up chain",
                             "role": "neutral", "q": c["q"], "answer": ans, "route": route, "fails": fails, "ok": not fails,
                             "seconds": secs, "cited": bool(re.search(r"\[S\d+", ans))})
            print(f"{case['title'][:60]}: {sum(r['ok'] for r in rows if r['case'] == case['title'])} passed", flush=True)
    for r in rows:
        r["system"] = (f"naive-{ctx // 1024}k" + ("-tail" if TRIM == "tail" else "")) if system == "naive" else "ours"
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{sum(r['ok'] for r in rows)}/{len(rows)} passed -> {out}")


def stress_report(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.jsonl")))
    data = {}
    for p in files:
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        if rows:
            data[rows[0]["system"]] = rows
    if not data:
        return []
    order = sorted([s for s in data if s != "ours"]) + (["ours"] if "ours" in data else [])
    nm = lambda s: "Our architecture" if s == "ours" else f"Naive Qwen2.5-7B ({s.split('-')[1]})"
    L = ["", "## Hand-written lawyer questions (stress set: long answers, lists, precedents, both perspectives)", "",
         "| Question kind | n | " + " | ".join(nm(s) for s in order) + " |", "|---|---|" + "---|" * len(order)]
    kinds = defaultdict(lambda: defaultdict(list))
    for s in order:
        for r in data[s]:
            k = "follow-up chain" if r["tests"] == "follow-up chain" else (r["role"] if r["role"] != "neutral" else "neutral")
            kinds[k][s].append(r["ok"])
    label = {"neutral": "Neutral reading questions", "accused": "Defence-perspective questions",
             "complainant": "Prosecution-perspective questions", "follow-up chain": "Follow-up chains"}
    for k in ("neutral", "accused", "complainant", "follow-up chain"):
        if k in kinds:
            n = len(kinds[k][order[0]])
            L.append(f"| {label[k]} | {n} | " + " | ".join(f"{sum(kinds[k][s])}/{len(kinds[k][s])}" for s in order) + " |")
    L.append(f"| **All** | **{len(data[order[0]])}** | " + " | ".join(
        f"**{sum(r['ok'] for r in data[s])}/{len(data[s])}**" for s in order) + " |")
    return L


# ───────────────────────────── memory: long conversations (LongMemEval-style) ─────────────────────────────
GENERIC = ["Summarise the facts of this case in a few lines.", "What were the main arguments of the appellant or petitioner?",
           "What were the main arguments of the respondent?", "Which earlier judgments did the court rely on?",
           "What reasons did the court give for its decision?", "What evidence or material did the court discuss?",
           "Were any witnesses or documents discussed? Give details.", "What legal provisions were discussed in this judgment?"]


def memory_script(doc, qs, length):
    """One conversation: two notes from the user, filler turns (questions about the judgment), a correction, then probes."""
    h = int(hashlib.md5(doc.encode()).hexdigest()[:6], 16)
    ref, d1, d2 = f"LX-{1000 + h % 9000}", f"{3 + h % 20} March 2027", f"{5 + h % 20} April 2027"
    doc_qs = [q for q in qs if not q.get("chain") and q["grade"] in ("match", "sections")]
    probes_doc = [q for q in doc_qs if q["type"] in ("meta_judge", "meta_outcome", "meta_petitioner")][:2]
    fill_doc = [q for q in doc_qs if q not in probes_doc]
    fillers = ([{"kind": "filler", "q": q["question"], "gold": q} for q in fill_doc] + [{"kind": "chatter", "q": g} for g in GENERIC])
    fillers = fillers[:2] if length == "short" else fillers[:16]
    half = len(fillers) // 2 if length == "long" else len(fillers)
    turns = [{"kind": "note", "q": f"Note for this conversation: my file reference number for this case is {ref}."},
             {"kind": "note", "q": f"Note: the next hearing in this matter is listed on {d1}."}]
    turns += fillers[:half]
    turns += [{"kind": "note", "q": f"Correction: the next hearing has been moved to {d2}."}]
    turns += fillers[half:]
    turns += [{"kind": "recall", "q": "What is my file reference number for this case?", "inc": [re.escape(ref)]},
              {"kind": "update", "q": "When is the next hearing in this matter?", "inc": [re.escape(d2.split()[0]) + r"\D{0,6}April"],
               "exc": [re.escape(d1.split()[0]) + r"\D{0,6}March"]}]
    turns += [{"kind": "doc_late", "q": q["question"], "gold": q} for q in probes_doc]
    return turns


def run_memory(system, model, ctx, qs_by_doc, docs, pdfs, out):
    rows = []
    if system == "naive":
        import requests
        from local_model_text import read_document
        url = __import__("ipc.llm", fromlist=["ollama_url"]).ollama_url()
        url = url if url.startswith("http") else "http://" + url
        HIST = 2500                                   # tokens left for the conversation; the judgment gets the rest
    else:
        os.environ["IPC_FORCE_SERVE"] = "0"
        os.environ.setdefault("IPC_CASES_DIR", os.path.join(os.path.dirname(os.path.abspath(out)), "cases"))
        from ipc.engine import Engine
        eng = Engine(model=model)
        eng.load(emit=lambda e: print("  ", e["message"]) if e.get("message") else None)
        eng.set_model(model)
    for di, doc in enumerate(docs, 1):
        path = os.path.join(pdfs, doc + ".pdf")
        if system == "naive":
            text, cut = fit(read_document(path), ctx, reserve=900 + HIST)
            sysmsg = NAIVE_SYSTEM + "\n\nJUDGMENT:\n" + text
            hist_tokens = max(HIST, ctx - 900 - int(len(text) / 3.5))     # a short judgment leaves more room for the chat
        else:
            if next((ev for ev in eng.analyse(path=path, use_cache=True) if ev.get("type") == "error"), None):
                continue
        for length in ("short", "long"):
            hist, dropped = [], 0
            if system == "ours":
                eng.clear_chat()
            for k, t in enumerate(memory_script(doc, qs_by_doc[doc], length)):
                t0 = time.time()
                if system == "naive":
                    budget, keep = int(hist_tokens * 3.5), []
                    for m in reversed(hist):                        # newest turns first, as many as fit
                        budget -= len(m["content"])
                        if budget < 0:
                            break
                        keep.insert(0, m)
                    dropped = len(hist) - len(keep)
                    r = requests.post(f"{url}/api/chat", json={"model": model, "stream": False,
                                      "messages": [{"role": "system", "content": sysmsg}] + keep + [{"role": "user", "content": t["q"]}],
                                      "options": {"num_ctx": ctx, "temperature": 0, "num_predict": 300}}, timeout=900)
                    r.raise_for_status()
                    ans = r.json().get("message", {}).get("content", "").strip()
                    hist += [{"role": "user", "content": t["q"]}, {"role": "assistant", "content": ans}]
                else:
                    parts = []
                    for ev in eng.chat(t["q"], role="neutral"):
                        if ev.get("type") == "token":
                            parts.append(ev["text"])
                        elif ev.get("type") == "revise":
                            parts = []
                    ans = "".join(parts).strip()
                if t["kind"] in ("note", "chatter"):
                    continue
                if "gold" in t:
                    ok = grade_row(ans, t["gold"], False)["correct"]
                else:
                    ok = not stress_check(ans, t.get("inc"), t.get("exc"))
                rows.append({"doc": doc, "length": length, "turn": k + 1, "kind": t["kind"], "q": t["q"], "answer": ans, "ok": ok,
                             "seconds": round(time.time() - t0, 1), "history_dropped": dropped if system == "naive" else 0})
        print(f"[{di}/{len(docs)}] {doc}: " + " · ".join(
            f"{L}: {sum(r['ok'] for r in rows if r['doc'] == doc and r['length'] == L and r['kind'] in ('recall', 'update'))}/2 memory"
            for L in ("short", "long")), flush=True)
    tag = (f"naive-{ctx // 1024}k") if system == "naive" else "ours"
    for r in rows:
        r["system"] = tag
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"-> {out}")


def memory_report(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.jsonl")))
    data = {}
    for p in files:
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        if rows:
            data[rows[0]["system"]] = rows
    if not data:
        return []
    order = sorted([s for s in data if s != "ours"]) + (["ours"] if "ours" in data else [])
    nm = lambda s: "Our architecture" if s == "ours" else f"Naive Qwen2.5-7B ({s.split('-')[1]})"
    kinds = [("recall", "Recall a note the user gave at the start"), ("update", "Use the user's later correction (not the old date)"),
             ("doc_late", "Answer about the judgment at the end of the conversation"), ("filler", "Judgment questions during the conversation")]
    L = ["", "## Memory over a conversation (LongMemEval-style; same model, same 8k window)", "",
         "Each conversation: the user gives two notes, asks questions about the judgment, corrects one note, then is probed. "
         "Short = 2 judgment questions before the probes, long = up to 16. "
         "The naive model keeps the judgment (cut to fit) plus as many recent messages as still fit in its window.", "",
         "| Test | Conversation | " + " | ".join(nm(s) for s in order) + " |", "|---|---|" + "---|" * len(order)]
    for k, label in kinds:
        for length in ("short", "long"):
            cells = []
            for s in order:
                rs = [r for r in data[s] if r["kind"] == k and r["length"] == length]
                cells.append(f"{sum(r['ok'] for r in rs)}/{len(rs)}" if rs else "–")
            if any(c != "–" for c in cells):
                L.append(f"| {label} | {length} | " + " | ".join(cells) + " |")
    nv = [s for s in order if s != "ours"]
    if nv:
        lr = [r for r in data[nv[0]] if r["length"] == "long"]
        if lr:
            L.append(f"\nIn long conversations the naive model had to drop on average "
                     f"{sum(r['history_dropped'] for r in lr) / len(lr):.0f} earlier messages to keep the judgment in its window.")
    return L


# ───────────────────────────── report ─────────────────────────────
GROUPS = [("Case details (parties, judge, date)", lambda t: t in ("meta_petitioner", "meta_respondent", "meta_judge", "meta_date")),
          ("Outcome of the case", lambda t: t == "meta_outcome"),
          ("Charges / sections", lambda t: t == "fact_charges"),
          ("Facts (FIR date, police station, sentence)", lambda t: t.startswith("fact_") and t != "fact_charges"),
          ("Follow-up chains (3-turn conversations)", lambda t: t == "chain"),
          ("Trap: answer is NOT in the judgment", lambda t: t == "trap"),
          ("Vague: should ask what the user means", lambda t: t == "vague")]


def report(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.jsonl")))
    data = {}
    for p in files:
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        rows = [r for r in rows if not r.get("excluded")]
        if rows:
            data[rows[0]["system"]] = {r["id"]: r for r in rows}
    if not data:
        raise SystemExit(f"no result files in {folder}")
    common = set.intersection(*(set(d) for d in data.values()))          # only questions every system answered
    systems = sorted(data, key=lambda s: (s != "ours", s))
    order = sorted([s for s in systems if s.startswith("naive")], key=lambda s: int(re.sub(r"\D", "", s) or 0)) + \
            [s for s in systems if s == "ours"]
    name = {s: ("Our architecture" if s == "ours" else f"Naive Qwen2.5-7B ({s.split('-')[1]} context"
                + (", Ollama-default cut" if s.endswith("-tail") else "") + ")") for s in order}

    def pct(rows, key="correct"):
        return 100.0 * sum(bool(r[key]) for r in rows) / len(rows) if rows else float("nan")

    def ci(p, n):
        import math
        if not n:
            return 0.0
        p /= 100
        return 196 * math.sqrt(max(p * (1 - p), 1e-9) / n)

    L = ["# Our architecture vs. a naive offline LLM", "",
         f"Same model (Qwen2.5-7B via Ollama) as the writer in both systems, same {len(common)} questions on "
         f"{len({data[order[0]][i]['doc'] for i in common})} judgments, same automatic grader.", ""]
    hdr = "| Question group | n | " + " | ".join(name[s] for s in order) + " |"
    L += [hdr, "|" + "---|" * (2 + len(order))]
    for gname, fn in GROUPS:
        ids = [i for i in common if fn(data[order[0]][i]["type"])]
        if ids:
            L.append(f"| {gname} | {len(ids)} | " + " | ".join(f"{pct([data[s][i] for i in ids]):.0f}%" for s in order) + " |")
    allr = {s: [data[s][i] for i in common] for s in order}
    L.append(f"| **All questions** | **{len(common)}** | " + " | ".join(
        f"**{pct(allr[s]):.0f}%** ±{ci(pct(allr[s]), len(common)):.0f}" for s in order) + " |")
    L += ["", "## Reliability", "", "| | " + " | ".join(name[s] for s in order) + " |", "|---|" + "---|" * len(order)]
    answerable = {s: [r for r in allr[s] if r["grade"] not in ("abstain", "vague")] for s in order}
    L.append("| Wrong answers stated as fact (hallucinated), answerable questions | " +
             " | ".join(f"{pct(answerable[s], 'hallucinated'):.0f}%" for s in order) + " |")
    trap = {s: [r for r in allr[s] if r["grade"] == "abstain"] for s in order}
    L.append("| Invented an answer to a trap question | " + " | ".join(f"{100 - pct(trap[s]):.0f}%" for s in order) + " |")
    L.append("| Answers that cite the judgment's sentences ([S#], checkable) | " + " | ".join(f"{pct(allr[s], 'cited'):.0f}%" for s in order) + " |")
    trunc = {s: [r for r in allr[s] if "truncated" in r] for s in order}
    L.append("| Judgment did not fit the model's context (middle cut) | " + " | ".join(
        (f"{pct(trunc[s], 'truncated'):.0f}% of questions" if trunc[s] else "never (retrieval)") for s in order) + " |")
    import statistics as st
    L.append("| Median seconds per answer | " + " | ".join(f"{st.median([r['timing_s'] for r in allr[s]]):.1f}" for s in order) + " |")
    if "ours" in data:
        o = allr["ours"]
        L += ["", f"Our router: {dict(Counter(r.get('route') for r in o))} · grounding check revised "
                  f"{sum(bool(r.get('grounding_revised')) for r in o)} answers · gold sentence retrieved for "
                  f"{pct([r for r in o if r.get('gold_retrieved') is not None], 'gold_retrieved'):.0f}% of fact questions."]
    # long vs short judgments: where the naive model loses the text
    if any(trunc.values()):
        nv = [s for s in order if s.startswith("naive")][-1:]            # the largest window: the fairest naive baseline
        L += ["", f"## Judgments that fit the naive model's largest window ({nv[0].split('-')[1]}) vs. ones that did not", "",
              "| | " + " | ".join(name[s] for s in order) + " |", "|---|" + "---|" * len(order)]
        for label, want in (("Judgment fitted", False), ("Judgment cut", True)):
            ids = [i for i in common if data[nv[0]][i].get("truncated") == want]
            if ids:
                L.append(f"| {label} (n={len(ids)}) | " + " | ".join(f"{pct([data[s][i] for i in ids]):.0f}%" for s in order) + " |")
    L += stress_report(os.path.join(folder, "stress"))
    L += memory_report(os.path.join(folder, "memory"))
    md = os.path.join(folder, "REPORT.md")
    open(md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = [(g, fn) for g, fn in GROUPS if any(fn(data[order[0]][i]["type"]) for i in common)]
        fig, ax = plt.subplots(figsize=(11, 5))
        w = 0.8 / len(order)
        greys = ["#b8bec8", "#9aa4b2", "#7d8794", "#d3d7de"]
        cols = [("#2f7de1" if x == "ours" else greys[k % len(greys)]) for k, x in enumerate(order)]
        for k, s in enumerate(order):
            vals = [pct([data[s][i] for i in common if fn(data[s][i]["type"])]) for _, fn in groups]
            ax.bar([x + k * w for x in range(len(groups))], vals, w, label=name[s], color=(cols[k] if cols else None))
        ax.set_xticks([x + w * (len(order) - 1) / 2 for x in range(len(groups))])
        ax.set_xticklabels([g.split(" (")[0].split(":")[0] for g, _ in groups], rotation=20, ha="right")
        ax.set_ylabel("% correct"); ax.set_ylim(0, 100); ax.legend(frameon=False)
        ax.set_title("Correct answers by question type"); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(folder, "compare.png"), dpi=160)
        print("chart ->", os.path.join(folder, "compare.png"))
    except Exception as e:                                                 # noqa: BLE001
        print("(no chart:", e, ")")


def main():
    if "--report" in sys.argv:
        k = sys.argv.index("--report")
        return report(sys.argv[k + 1] if len(sys.argv) > k + 1 else os.path.join(HERE, "compare"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="?", help="ipc_qa/questions.jsonl (not needed with --stress)")
    ap.add_argument("--memory", action="store_true", help="long-conversation memory test (notes, corrections, late recall)")
    ap.add_argument("--stress", action="store_true", help="run the hand-written stress set (stress_cases.json, test-case-files/)")
    ap.add_argument("--system", required=True, choices=["naive", "ours"])
    ap.add_argument("--pdfs", default=None, help="default: <questions folder>/pdfs")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--ctx", type=int, default=8192, help="naive only: context window in tokens (8192 = the app's own window)")
    ap.add_argument("--trim", default="middle", choices=["middle", "tail"],
                    help="naive only: how a too-long judgment is cut (middle = keep start+end; tail = keep the end, Ollama's default)")
    ap.add_argument("--limit", type=int, default=60, help="number of judgments (spread evenly); 0 = all")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "compare"))
    a = ap.parse_args()
    if a.stress:
        TRIM_SET(a.trim)
        d = os.path.join(a.out_dir, "stress")
        os.makedirs(d, exist_ok=True)
        tag = (f"naive-{a.ctx // 1024}k" + ("-tail" if a.trim == "tail" else "")) if a.system == "naive" else "ours"
        return run_stress(a.system, a.model, a.ctx, os.path.join(HERE, "stress_cases.json"), os.path.join(HERE, "test-case-files"),
                          os.path.join(d, f"{tag}.jsonl"))
    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    by_doc = defaultdict(list)
    for q in qs:
        by_doc[q["doc"]].append(q)
    pdfs = a.pdfs or os.path.join(os.path.dirname(os.path.abspath(a.questions)), "pdfs")
    if a.memory:
        have = [q for q in qs if os.path.exists(os.path.join(pdfs, q["doc"] + ".pdf"))]
        mdocs = pick_docs(have, a.limit)
        fix_judge_gold(by_doc, mdocs, pdfs)
        d = os.path.join(a.out_dir, "memory")
        os.makedirs(d, exist_ok=True)
        tag = f"naive-{a.ctx // 1024}k" if a.system == "naive" else "ours"
        return run_memory(a.system, a.model, a.ctx, by_doc, mdocs, pdfs, os.path.join(d, f"{tag}.jsonl"))
    have = [q for q in qs if os.path.exists(os.path.join(pdfs, q["doc"] + ".pdf"))]     # only judgments whose PDF is here
    if not have:
        raise SystemExit(f"no PDFs for these questions in {pdfs}")
    docs = pick_docs(have, a.limit)
    fix_judge_gold(by_doc, docs, pdfs)
    os.makedirs(a.out_dir, exist_ok=True)
    TRIM_SET(a.trim)
    tag = (f"naive-{a.ctx // 1024}k" + ("-tail" if a.trim == "tail" else "")) if a.system == "naive" else "ours"
    out = os.path.join(a.out_dir, f"{tag}.jsonl")
    print(f"{tag}: {len(docs)} judgments, {sum(len(by_doc[d]) for d in docs)} questions -> {out}")
    if a.system == "naive":
        globals()["NAIVE_TAG"] = tag
        run_naive(by_doc, docs, pdfs, a.model, a.ctx, out)
    else:
        run_ours(by_doc, docs, pdfs, a.model, out)


if __name__ == "__main__":
    main()
