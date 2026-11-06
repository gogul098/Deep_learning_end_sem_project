"""label_ipc.py · run every generated question through the IPC assistant and grade the answer (prob_reas labelling).

  python label_ipc.py ipc_qa/questions.jsonl --pdfs ipc_qa/pdfs --model qwen2.5:7b
  python label_ipc.py ipc_qa/questions.jsonl --pdfs ipc_qa/pdfs --shard 0/2 --checkpoint     # half the judgments, backed up

For each judgment: analyse it once (same pipeline as the app), then ask its questions in fresh chats (a 'chain' is
asked as ONE conversation). The router measures its signals and records the route it WOULD take, but every question
is answered (IPC_FORCE_SERVE=1): the label is "did serving now fail?", the outcome the network learns.
Grading (prob_reas grade, adapted):
  match      an accepted answer appears in the reply (whole words, or a close form); "not stated" / "I don't know" = fail
  sections   at least half of the charged sections appear in the reply
  abstain    trap question: correct only if the reply says the judgment does not state it
  vague      no answer can be right: serving it counts as a failure (the router should have asked)
Also recorded: was the gold sentence among the passages given to the model (retrieval recall), grounding revisions,
timings. Output: one JSON line per question in --out, resumable (finished ids are skipped).
Then on your machine:  python learn_router.py labels/ipc_qwen2.5-7b.jsonl
"""
import argparse, json, os, re, string, sys, time
from collections import Counter, defaultdict

os.environ["IPC_FORCE_SERVE"] = "1"
if "--checkpoint" not in sys.argv:                      # the checkpoint talks to the Hub; models load from ./models either way
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NOT_STATED = re.compile(r"not (?:stated|mentioned|recorded|discussed|available|specified|provided|given|described)|"
                        r"does not (?:mention|state|contain|record|discuss|refer|specify|provide|say)|"
                        r"no (?:mention|reference|information|details?|record|such)|records no|there is no|"
                        r"i (?:do not|don't) know|unknown|cannot be determined", re.I)
NEGATED = re.compile(r"\b(?:not|never|no|neither|nor)\s+(?:been\s+|be\s+|was\s+|were\s+)?$", re.I)


# ----------------------------------------------------------------------------------------------- grading (prob_reas)
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


STOP = {"and", "of", "in", "on", "for", "to", "by", "with", "or", "jr", "de"}


def _close_match(r: str, a: str) -> bool:
    from difflib import SequenceMatcher
    rt = [t for t in r.split() if t not in STOP]
    at = [t for t in a.split() if t not in STOP]
    if rt and at and set(rt) <= set(at) and len(set(rt)) / len(set(at)) >= 0.5 and max(map(len, rt)) >= 3:
        return True
    return len(r) >= 6 and len(a) >= 6 and SequenceMatcher(None, r, a).ratio() >= 0.9


def _squash(s: str) -> str:                     # no spaces, articles kept: "A. S. Gadkari" -> "asgadkari"
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _found(reply_norm: str, gold_norm: str, reply: str = "", gold: str = "") -> bool:
    i = f" {reply_norm} ".find(f" {gold_norm} ")
    if i < 0:
        g = _squash(gold or gold_norm)
        return len(g) >= 6 and g in _squash(reply or reply_norm)
    return not NEGATED.search(f" {reply_norm} "[:i + 1][-30:])           # "was not allowed" is not "allowed"

MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                         "september", "october", "november", "december"])}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())}); MONTHS["sept"] = 9
_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
_DATE_PATS = [(re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?(?:\s+day\s+of)?\s+" + _MON + r",?\s+(\d{4})\b", re.I), ("d", "m", "y")),
              (re.compile(r"\b" + _MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I), ("m", "d", "y")),
              (re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b"), ("d", "mn", "y"))]


def dates_in(text: str) -> set:
    out = set()
    for pat, order in _DATE_PATS:
        for m in pat.finditer(text or ""):
            v = dict(zip(order, m.groups()))
            mon = int(v["mn"]) if "mn" in v else MONTHS.get(v["m"].lower().rstrip("."))
            try:
                out.add((int(v["d"]), int(mon), int(v["y"])))
            except (TypeError, ValueError):
                pass
    return out


OUTCOME_STEM = {"dismissed": r"dismiss", "rejected": r"reject", "fails": r"\bfail", "allowed": r"\ballow",
                "set aside": r"set(?:s|ting)? aside|quash", "succeeds": r"succeed", "disposed": r"dispos",
                "remanded": r"remand", "remitted": r"remit", "partly allowed": r"part(?:ly|ially)? allow|allowed in part"}
_NEG_BEFORE = re.compile(r"\b(?:not|never|no)\s+(?:been\s+|be\s+)?$", re.I)
STATION_STOP = {"fir", "the", "said", "concerned", "local", "same", "that", "this", "nearest", "jurisdictional", "a", "an"}


def clean_station(g: str):
    """'Kotbhai for' -> 'Kotbhai'; 'is hardly' / 'within whose' / 'FIR before' -> None (generator picked junk)."""
    toks = []
    for t in g.split():
        if not t[:1].isupper() or t.lower() in STATION_STOP or t.isupper() and len(t) <= 3:
            break
        toks.append(t)
    return " ".join(toks) if toks and len(" ".join(toks)) >= 3 else None


def gold_problem(q: dict):
    """Why this question's gold answer can't be trusted (the row is then left out of learning), else None."""
    t, ans = q.get("type", ""), q.get("answers") or []
    if t == "fact_police_station" and not any(clean_station(a) for a in ans):
        return "junk police-station gold"
    if t in ("meta_petitioner", "meta_respondent") and any(re.match(r"(?i)\s*in\s+re\b|suo\s+motu", a) for a in ans):
        return "suo motu / in re case: no real party"
    if t == "meta_judge" and not q.get("author_from_pdf"):
        return "judge gold from metadata (bench list, not the author): re-grade with --pdfs"
    return None


def grade(reply: str, q: dict) -> dict:
    r = normalize(reply)
    empty = r == ""
    abstained = (not empty) and bool(NOT_STATED.search(reply))
    mode = q["grade"]
    raw_golds = [a for a in q.get("answers", []) if normalize(a)]
    if q.get("type") in ("meta_petitioner", "meta_respondent") and mode == "match":
        for a in list(raw_golds):                         # "COMMISSIONER OF INCOME TAX-1" / "M/S X PVT. LTD. & ORS."
            core = re.sub(r"(?i)[\s,&-]*(?:\d+|ors?\.?|others|anr\.?|another|etc\.?)\s*$", "", a).strip(" .,-")
            if core and core != a:
                raw_golds.append(core)
    if q.get("type") == "fact_police_station":
        raw_golds = [c for c in (clean_station(a) for a in raw_golds) if c]
    if q.get("type") == "meta_judge":                    # "Justice D.Y. Chandrachud" answers "Dhananjaya Y Chandrachud"
        raw_golds += [a.split()[-1] for a in raw_golds if len(a.split()[-1]) >= 5]
    golds = [normalize(a) for a in raw_golds]
    if mode == "abstain":
        correct = abstained
    elif mode == "vague":
        correct = False
    elif mode == "sections":
        hits = sum(1 for g, a in zip(golds, raw_golds) if _found(r, g, reply, a))
        correct = (not empty) and golds and hits >= max(1, (len(golds) + 1) // 2)
    elif golds and dates_in(" ".join(raw_golds)):                       # a date: compare the dates, not the spelling
        correct = (not empty) and bool(dates_in(reply) & dates_in(" ".join(raw_golds)))
    elif golds and all(a.lower() in OUTCOME_STEM for a in raw_golds):  # outcome: "dismissing the appeal" = dismissed
        low = reply.lower()
        correct = (not empty) and any(not _NEG_BEFORE.search(low[:m.start()][-25:])
                                      for a in raw_golds for m in re.finditer(OUTCOME_STEM[a.lower()], low))
    else:
        correct = (not empty) and (any(_found(r, g, reply, a) for g, a in zip(golds, raw_golds)) or any(_close_match(r, g) for g in golds))
        if abstained and not correct:
            pass
        elif abstained and correct and len(r.split()) < 12:               # "not stated" + a stray matching word
            correct = False
    return {"correct": bool(correct), "abstained": bool(abstained), "empty": empty,
            "hallucinated": not (correct or abstained or empty), "excluded": gold_problem(q)}


def evidence_retrieved(evidence: str, passages: list) -> bool:
    """Was the sentence holding the gold answer given to the model? (a substring of 60 characters is enough)"""
    if not evidence:
        return None
    ev = normalize(evidence)[:120]
    keys = [ev[i:i + 60] for i in range(0, max(1, len(ev) - 59), 30)]
    text = " ".join(normalize(p.get("text", "")) for p in passages)
    return any(k and k in text for k in keys)


# ----------------------------------------------------------------------------------------------- one question
def ask(eng, q, model, role="neutral"):
    t0 = time.time()
    text, stats, route_ev, issues, err = [], {}, {}, [], None
    for ev in eng.chat(q["question"], role=role):
        k = ev.get("type")
        if k == "token":
            text.append(ev["text"])
        elif k == "route":
            route_ev = ev
        elif k == "revise":
            text, issues = [], ev.get("issues", [])
        elif k == "done":
            stats = ev.get("stats", {}) or stats
        elif k == "error":
            err = ev.get("message")
    reply = re.sub(r"\n*\[unverified:.*\]\s*$", "", "".join(text).strip(), flags=re.S)
    d = eng.last_route or {}
    raw = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else float(v)) for k, v in (d.get("raw") or {}).items()}
    g = grade(reply, q)
    return {**{k: q[k] for k in ("id", "doc", "type", "question", "answers", "grade", "chain", "step")},
            "raw": raw, "evidence": d.get("evidence"), "route_router": d.get("route_router", d.get("route")),
            "p_fail": d.get("p_hallucination_high"), "blame": d.get("blame"),
            "gold_retrieved": evidence_retrieved(q.get("evidence", ""), d.get("passages") or route_ev.get("passages") or []),
            "reply": reply, **g, "grounding_revised": bool(issues), "grounding_issues": issues, "error": err,
            "model": model, "dataset": "ipc-" + q["type"].split("_")[0], "timing_s": round(time.time() - t0, 1),
            "tokens_per_s": stats.get("tokens_per_s")}


AUTHOR = re.compile(r"(?m)^[ \t]*(?:Dr\.?[ \t]+)?((?:[A-Z]\.[ \t]?){0,3}[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z.]*){0,3})[ \t]*,[ \t]*(?:C\.?[ \t]?)?J\.?I?\.?[ \t]*[:.]?[ \t]*$|^[ \t]*((?:[A-Z]\.[ \t]?){0,3}[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z.]*){0,3})[ \t]*,[ \t]*(?:C\.?[ \t]?)?J\.[ \t]*:")


def author_from_pdf(path: str):
    """The judge who wrote the judgment, as the PDF names him: 'J.B. PARDIWALA, J. : 1. Leave granted.'"""
    try:
        import pymupdf
        with pymupdf.open(path) as d:
            text = "\n".join(p.get_text() for p in d)[:20000]
    except Exception:
        return None
    i = max(text.upper().find("JUDGMENT"), 0)
    m = AUTHOR.search(text, i) or AUTHOR.search(text)
    return (m.group(1) or m.group(2)).strip() if m else None


def regrade(files, pdfs=None):
    """Re-grade saved answers with the current grade() (no models needed):
         python label_ipc.py --regrade labels/*.jsonl [--pdfs ipc_qa/pdfs]
    With --pdfs, the gold author of 'which judge authored' questions is taken from the judgment text (the metadata
    'judge' column lists the bench, presiding judge first, not the author)."""
    authors = {}
    for f in files:
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        changed = 0
        for r in rows:
            if pdfs and (r["type"] == "meta_judge" or (r.get("chain") and "authored" in r.get("question", ""))):
                if r["doc"] not in authors:
                    authors[r["doc"]] = author_from_pdf(os.path.join(pdfs, r["doc"] + ".pdf"))
                au = authors[r["doc"]]
                if au:
                    r["answers"] = [au] + ([au.split()[-1]] if len(au.split()[-1]) >= 5 else [])
                    r["author_from_pdf"] = True
            g = grade(r.get("reply", ""), r)
            changed += g["correct"] != r.get("correct")
            r.update(g)
        with open(f, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        by = defaultdict(list)
        for r in rows:
            if not r.get("excluded"):
                by[r["type"]].append(r["correct"])
        ex = Counter(r["excluded"] for r in rows if r.get("excluded"))
        print(f"{f}: {len(rows)} answers, {changed} grades changed, {sum(ex.values())} left out of learning: {dict(ex)}")
        for t, c in sorted(by.items()):
            print(f"  {t:<22}{len(c):>5}  correct {sum(c) / len(c):.0%}")


def main():
    if "--regrade" in sys.argv:
        args = [x for x in sys.argv[1:] if x != "--regrade"]
        pdfs = None
        if "--pdfs" in args:
            k = args.index("--pdfs"); pdfs = args[k + 1]; del args[k:k + 2]
        return regrade(args, pdfs)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions")
    ap.add_argument("--pdfs", default=None, help="folder with <doc>.pdf (default: <questions folder>/pdfs)")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--out", default=None, help="default: labels/ipc_<model>.jsonl (shards: ..._shard<i>.jsonl)")
    ap.add_argument("--shard", default="0/1", help="i/n: label every n-th judgment starting at i (run notebooks in parallel)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many judgments (a quick trial)")
    ap.add_argument("--checkpoint", action="store_true", help="back up to a private HF dataset (needs a WRITE HF_TOKEN)")
    a = ap.parse_args()
    i_sh, n_sh = (int(x) for x in a.shard.split("/"))
    pdfs = a.pdfs or os.path.join(os.path.dirname(os.path.abspath(a.questions)), "pdfs")
    slug = re.sub(r"[^a-z0-9.]+", "-", a.model.lower()).strip("-")
    out = a.out or os.path.join(HERE, "labels", f"ipc_{slug}" + (f"_shard{i_sh}" if n_sh > 1 else "") + ".jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ckpt = None
    if a.checkpoint:
        from label_checkpoint import HFCheckpoint
        ckpt = HFCheckpoint()
        ckpt.pull(out)
        ckpt.watch(out)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}

    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    by_doc = defaultdict(list)
    for q in qs:
        by_doc[q["doc"]].append(q)
    docs = [d for k, d in enumerate(sorted(by_doc)) if k % n_sh == i_sh]
    if a.limit:
        docs = docs[:a.limit]
    todo = sum(1 for d in docs for q in by_doc[d] if q["id"] not in done)
    print(f"{len(docs)} judgments in shard {a.shard} · {todo} questions to label · {len(done)} already done -> {out}")

    from ipc.engine import Engine
    eng = Engine(model=a.model)
    eng.load(emit=lambda e: print("  ", e["message"]) if e.get("message") else None)
    eng.set_model(a.model)
    n_new, t_start = 0, time.time()
    f = open(out, "a", encoding="utf-8")
    for di, doc in enumerate(docs, 1):
        pending = [q for q in by_doc[doc] if q["id"] not in done]
        if not pending:
            continue
        path = os.path.join(pdfs, f"{doc}.pdf")
        if not os.path.exists(path):
            print(f"  ! missing {path}")
            continue
        err = None
        for ev in eng.analyse(path=path, use_cache=True):
            if ev.get("type") == "error":
                err = ev["message"]
        if err:
            print(f"  ! {doc}: analysis failed: {err}")
            continue
        singles = [q for q in pending if not q.get("chain")]
        chains = defaultdict(list)
        for q in pending:
            if q.get("chain"):
                chains[q["chain"]].append(q)
        rows = []
        for q in singles:
            eng.clear_chat()
            rows.append(ask(eng, q, a.model))
        for cid, cq in chains.items():
            eng.clear_chat()                                      # one conversation
            for q in sorted(cq, key=lambda x: x["step"]):
                rows.append(ask(eng, q, a.model))
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_new += 1
            if ckpt:
                f.flush()
                ckpt.maybe_push(n_new)
        f.flush()
        ok = sum(r["correct"] for r in rows)
        rate = (time.time() - t_start) / max(1, n_new)
        print(f"[{di}/{len(docs)}] {doc}: {ok}/{len(rows)} correct · {rate:.1f} s/question · "
              f"~{rate * (todo - n_new) / 3600:.1f} h left")
    f.close()
    if ckpt:
        ckpt.push()
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    print(f"\n{len(rows)} labelled answers in {out}")
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    for t, rs in sorted(by.items()):
        print(f"  {t:<22}{len(rs):>5}  correct {sum(r['correct'] for r in rs) / len(rs):.0%}")
    rec = [r["gold_retrieved"] for r in rows if r.get("gold_retrieved") is not None]
    if rec:
        print(f"  gold sentence retrieved for {sum(rec) / len(rec):.0%} of {len(rec)} fact questions (reranker recall)")
    print("  router would have:", dict(Counter(r["route_router"] for r in rows)))


if __name__ == "__main__":
    main()
