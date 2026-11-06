"""compare_rerankers.py · which reranker puts the sentence that holds the answer into the passages the model sees?

  python compare_rerankers.py                       # ipc_qa/questions.jsonl + ipc_qa/pdfs, the default candidates
  python compare_rerankers.py --rerankers current BAAI/bge-reranker-base --limit 50

For every generated FACT question (charges, FIR date, police station, sentence, compensation) we know the sentence of the
judgment that holds the answer. This script does exactly what the app's retriever does (same sentence split, same
InLegalBERT vectors + BM25 search, top RERANK_POOL candidates) and then lets each reranker order them. Reported:
  pool@12   gold sentence among the 12 candidates at all (no reranker can do better than this)
  @1 / @5   gold sentence ranked first / among the RETRIEVE_K=5 the model gets
  pool@30   same with a 30-candidate pool, to see whether a bigger pool would help
No LLM, no Hyena; GPU used for the embedder and rerankers when available. Candidate rerankers are downloaded once into
models/rerankers/<name> (internet needed the first time).
"""
import argparse, json, os, re, sys, time
os.environ["HF_HUB_OFFLINE"] = "0"; os.environ["TRANSFORMERS_OFFLINE"] = "0"   # the app runs offline; this script downloads
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEFAULT = ["current", "cross-encoder/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3"]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def gold_indices(evidence, sentences):
    """Sentences that contain (a 60-character window of) the saved evidence sentence."""
    ev = norm(evidence)[:160]
    keys = [ev[i:i + 60] for i in range(0, max(1, len(ev) - 59), 20)] or [ev]
    ns = [norm(s) for s in sentences]
    hit = {i for i, s in enumerate(ns) for k in keys if k and k in s}
    if not hit:                                    # evidence spans two sentences in this split: match either half
        for i in range(len(ns) - 1):
            if any(k in ns[i] + " " + ns[i + 1] for k in keys):
                hit |= {i, i + 1}
    return hit


class Embedder:
    def __init__(self, path, device):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModel.from_pretrained(path).to(device).eval()

    def __call__(self, texts):
        t = self.torch
        out = np.zeros((len(texts), self.m.config.hidden_size), dtype="float32")
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for b in range(0, len(order), 64):
            ids = order[b:b + 64]
            x = self.tok([texts[i] for i in ids], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with t.no_grad():
                h = self.m(**x).last_hidden_state.float()
            mk = x["attention_mask"][..., None].float()
            out[ids] = t.nn.functional.normalize((h * mk).sum(1) / mk.sum(1), dim=-1).cpu().numpy()
        return out


class Reranker:
    def __init__(self, name, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device, self.name = torch, device, name
        if name == "current":
            path = os.path.join(HERE, "models", "reranker")
        else:
            path = os.path.join(HERE, "models", "rerankers", name.replace("/", "__"))
            if not os.path.isfile(os.path.join(path, "config.json")):
                from huggingface_hub import snapshot_download
                print(f"  downloading {name} ...")
                snapshot_download(name, local_dir=path, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "tokenizer*", "vocab*", "sentencepiece*"])
        self.tok = AutoTokenizer.from_pretrained(path)
        dt = torch.float16 if device == "cuda" else torch.float32
        self.m = AutoModelForSequenceClassification.from_pretrained(path, dtype=dt).to(device).eval()
        self.params = sum(p.numel() for p in self.m.parameters()) / 1e6

    def score(self, q, texts):
        t = self.torch
        x = self.tok([q] * len(texts), texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with t.no_grad():
            lg = self.m(**x).logits.float()
        lg = lg[:, 0] if lg.shape[-1] == 1 else lg[:, -1]
        return t.sigmoid(lg).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=os.path.join(HERE, "ipc_qa", "questions.jsonl"))
    ap.add_argument("--pdfs", default=None)
    ap.add_argument("--rerankers", nargs="+", default=DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="first N fact questions only (a quick try)")
    ap.add_argument("--out", default=os.path.join(HERE, "bench_out", "reranker_compare.json"))
    a = ap.parse_args()
    pdfs = a.pdfs or os.path.join(os.path.dirname(os.path.abspath(a.questions)), "pdfs")

    import torch, pymupdf
    from ipc.pipeline import _paths, _lh
    from ipc.retriever import DocIndex, RERANK_POOL, RETRIEVE_K
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths = _paths(os.path.join(HERE, "models"))
    LH = _lh(paths.get("final", os.path.join(HERE, "models", "final")))
    emb = Embedder(os.path.join(HERE, "models", "inlegalbert"), dev)

    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    qs = [q for q in qs if q["type"].startswith("fact_") and q.get("evidence")]
    if a.limit:
        qs = qs[:a.limit]
    by_doc = defaultdict(list)
    for q in qs:
        by_doc[q["doc"]].append(q)
    print(f"{len(qs)} fact questions in {len(by_doc)} judgments · device {dev}")

    # 1. the app's candidates for every question (pool of 12 and of 30)
    items, t0 = [], time.time()
    for n, (doc, dq) in enumerate(sorted(by_doc.items()), 1):
        with pymupdf.open(os.path.join(pdfs, doc + ".pdf")) as d:
            text = "\n".join(p.get_text() for p in d)
        sents = LH.split_sentences(text)
        idx = DocIndex.build(sents, emb(sents))
        for q in dq:
            gold = gold_indices(q["evidence"], sents)
            if not gold:
                continue
            qv = emb([q["question"]])[0]
            pool30 = [h["i"] for h in idx.corpus.search(q["question"], qv, 30)]
            items.append({"id": q["id"], "type": q["type"], "q": q["question"], "gold": sorted(gold),
                          "pool": pool30[:RERANK_POOL], "pool30": pool30, "texts": {i: sents[i] for i in pool30}})
        if n % 25 == 0:
            print(f"  {n}/{len(by_doc)} judgments indexed ({time.time() - t0:.0f}s)")
    print(f"{len(items)} questions whose gold sentence was found in the split\n")
    if not items:
        raise SystemExit("no usable questions")
    in_pool = np.mean([bool(set(it["pool"]) & set(it["gold"])) for it in items])
    in_pool30 = np.mean([bool(set(it["pool30"]) & set(it["gold"])) for it in items])
    nonre = np.mean([bool(set(it["pool"][:RETRIEVE_K]) & set(it["gold"])) for it in items])

    # 2. each reranker orders the same candidates
    rows = [("no reranker (vector+BM25 order)", None, nonre, np.mean([it["pool"][0] in it["gold"] for it in items]), None, None)]
    for name in a.rerankers:
        try:
            rr = Reranker(name, dev)
        except Exception as e:
            print(f"  ! {name}: {type(e).__name__}: {e}")
            continue
        t1, at1, at5, at5_30 = time.time(), [], [], []
        for it in items:
            for key, store in (("pool", None), ("pool30", None)):
                cand = it[key]
                s = rr.score(it["q"], [it["texts"][i] for i in cand])
                order = [cand[j] for j in np.argsort(-s)]
                hit5 = bool(set(order[:RETRIEVE_K]) & set(it["gold"]))
                if key == "pool":
                    at1.append(order[0] in it["gold"]); at5.append(hit5)
                else:
                    at5_30.append(hit5)
        ms = (time.time() - t1) / len(items) / 2 * 1000
        rows.append((f"{name} ({rr.params:.0f}M)", ms, float(np.mean(at5)), float(np.mean(at1)), float(np.mean(at5_30)), name))
        del rr
        if dev == "cuda":
            torch.cuda.empty_cache()

    print(f"gold sentence among the 12 candidates: {in_pool:.0%}   (among 30: {in_pool30:.0%})  - the ceiling for any reranker\n")
    print(f"{'':48}{'@5 (pool 12)':>13}{'@1':>7}{'@5 (pool 30)':>14}{'ms/question':>13}")
    for label, ms, r5, r1, r30, _ in rows:
        print(f"{label:48}{r5:>13.0%}{r1:>7.0%}{(f'{r30:.0%}' if r30 is not None else '-'):>14}{(f'{ms:.0f}' if ms else '-'):>13}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"n": len(items), "in_pool12": in_pool, "in_pool30": in_pool30,
               "rows": [{"reranker": r[0], "ms": r[1], "at5": r[2], "at1": r[3], "at5_pool30": r[4]} for r in rows]},
              open(a.out, "w"), indent=1, default=float)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
