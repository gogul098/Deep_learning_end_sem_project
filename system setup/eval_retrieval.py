"""eval_retrieval.py · which retrieval setting puts the sentence that holds the answer in front of the model? (no LLM)

  python eval_retrieval.py                 # ipc_qa/questions.jsonl + ipc_qa/pdfs, ~10 min on GPU
  python eval_retrieval.py --limit 60      # quick try

For every generated FACT question (gold answer sentence known) it compares, over the SAME judgments and questions:
  pool     12 / 30 / 60 / 100 first-stage candidates handed to the reranker
  fusion   linear  = cosine + 0.25 * BM25 (today)      rrf = 1/(60+rank_dense) + 1/(60+rank_bm25)
  unit     sent    = each sentence embedded alone (today)  win = sentence i scored by the window S[i-1] S[i] S[i+1]
  clean    raw     = text as split (today)          clean = page footers / file names / footnote lines removed
           before embedding and BM25 (citations still use the original sentence numbers)
Metric: gold sentence among the 5 sentences the model gets (@5), and with each one's +-1 neighbours (@5+nb, what the
app actually sends). The current cross-encoder (models/reranker) re-ranks every setting.
"""
import argparse, json, os, re, sys, time
from collections import defaultdict
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_rerankers import Embedder, Reranker, gold_indices      # noqa: E402

PAGE_JUNK = re.compile(r"\b\d{1,3}\s+of\s+\d{1,3}\s+\S+\.(?:odt|docx?|pdf)\b|\bPage\s+\d{1,3}\s+of\s+\d{1,3}\b|"
                       r"\b\d{1,3}\s+of\s+\d{1,3}\b(?!\s*\d)|\S+\.(?:odt|docx?|pdf)\b", re.I)
FOOTNOTE = re.compile(r"^\s*\d{1,2}\s+(?:\(?\d{4}\)?|AIR|SCC|Cri|Bail\s+Application|Writ|Criminal|Civil|SLP|MANU|\[\d{4}\])", re.I)


def clean(s: str) -> str:
    """For the index only: page footers, file names, and footnote lines ('17 2025 SCC Online Del 21.')."""
    if FOOTNOTE.search(s) and len(s.split()) <= 14:
        return ""
    return re.sub(r"\s+", " ", PAGE_JUNK.sub(" ", s)).strip()


def ranks(scores):
    r = np.empty(len(scores), dtype=float)
    r[np.argsort(-scores, kind="mergesort")] = np.arange(1, len(scores) + 1)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=os.path.join(HERE, "ipc_qa", "questions.jsonl"))
    ap.add_argument("--pdfs", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "bench_out", "retrieval_eval.json"))
    a = ap.parse_args()
    pdfs = a.pdfs or os.path.join(os.path.dirname(os.path.abspath(a.questions)), "pdfs")

    import torch, pymupdf
    from ipc.pipeline import _paths, _lh
    from ipc.retriever import Corpus, tokens
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    LH = _lh(_paths(os.path.join(HERE, "models")).get("final", os.path.join(HERE, "models", "final")))
    emb = Embedder(os.path.join(HERE, "models", "inlegalbert"), dev)
    rr = Reranker("current", dev)

    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    qs = [q for q in qs if q["type"].startswith("fact_") and q.get("evidence")]
    if a.limit:
        qs = qs[:a.limit]
    by_doc = defaultdict(list)
    for q in qs:
        by_doc[q["doc"]].append(q)
    print(f"{len(qs)} fact questions in {len(by_doc)} judgments · device {dev}")

    POOLS = [12, 30, 60, 100]
    SETTINGS = list(product(["linear", "rrf"], ["sent", "win"], ["raw", "clean"]))
    hits = {(f, u, c, p): [] for (f, u, c) in SETTINGS for p in POOLS}
    hits_nb = {k: [] for k in hits}
    ceiling = {k: [] for k in hits}
    t0, n_q = time.time(), 0
    for n, (doc, dq) in enumerate(sorted(by_doc.items()), 1):
        with pymupdf.open(os.path.join(pdfs, doc + ".pdf")) as d:
            S = LH.split_sentences("\n".join(p.get_text() for p in d))
        golds = [(q, gold_indices(q["evidence"], S)) for q in dq]
        golds = [(q, g) for q, g in golds if g]
        if not golds:
            continue
        views = {}
        for c in ("raw", "clean"):
            base = S if c == "raw" else [clean(s) for s in S]
            sent_txt = [t if t else " " for t in base]
            win_txt = [" ".join(x for x in base[max(0, i - 1):i + 2] if x) or " " for i in range(len(S))]
            for u, txt in (("sent", sent_txt), ("win", win_txt)):
                views[(u, c)] = (txt, Corpus(txt, emb(txt)))
        for q, gold in golds:
            n_q += 1
            qv = emb([q["question"]])[0]
            qt = tokens(q["question"])
            for (f, u, c) in SETTINGS:
                txt, corp = views[(u, c)]
                cos = corp.vecs @ qv
                bm = corp.bm25(qt)
                if f == "linear":
                    first = cos + 0.25 * (bm / bm.max() if bm.max() > 0 else bm)
                else:
                    first = 1.0 / (60 + ranks(cos)) + 1.0 / (60 + ranks(bm))
                order = list(np.argsort(-first, kind="mergesort")[:max(POOLS)])
                # the reranker reads what the user would see: the sentence (or its window, for 'win')
                sc = rr.score(q["question"], [txt[i] for i in order]) if order else np.array([])
                rscore = dict(zip(order, sc))
                for p in POOLS:
                    cand = order[:p]
                    top5 = sorted(cand, key=lambda i: -rscore[i])[:5]
                    nb = {j for i in top5 for j in (i - 1, i, i + 1)}
                    hits[(f, u, c, p)].append(bool(set(top5) & gold))
                    hits_nb[(f, u, c, p)].append(bool(nb & gold))
                    ceiling[(f, u, c, p)].append(bool(set(cand) & gold))
        if n % 25 == 0:
            print(f"  {n}/{len(by_doc)} judgments ({time.time() - t0:.0f}s)")

    print(f"\n{n_q} questions whose gold sentence was found\n")
    print(f"{'fusion':8}{'unit':6}{'clean':7}{'pool':>5}{'@5':>7}{'@5+nb':>8}{'in pool':>9}")
    rows = []
    for (f, u, c) in SETTINGS:
        for p in POOLS:
            k = (f, u, c, p)
            row = {"fusion": f, "unit": u, "clean": c, "pool": p, "at5": float(np.mean(hits[k])),
                   "at5_nb": float(np.mean(hits_nb[k])), "in_pool": float(np.mean(ceiling[k]))}
            rows.append(row)
            tag = "   <- today" if (f, u, c, p) == ("linear", "sent", "raw", 30) else ""
            print(f"{f:8}{u:6}{c:7}{p:>5}{row['at5']:>7.0%}{row['at5_nb']:>8.0%}{row['in_pool']:>9.0%}{tag}")
        print()
    best = max(rows, key=lambda r: (r["at5_nb"], r["at5"], -r["pool"]))
    print("best (by @5+nb, then @5, then smaller pool):", best)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"n": n_q, "rows": rows, "best": best}, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
