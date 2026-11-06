"""f1boost.py · can the trained v7 model score higher WITHOUT retraining the encoder or the verifier?

Two changes, both decided on DEV judgments (2021-22) and checked ONCE on the same 150 TEST judgments as the paper:
  1. wider shortlist: the verifier also checks encoder ranks 11-20 (30% of correct sections never reached it at k=10)
  2. a learned combiner instead of 0.6*encoder + 0.4*verifier >= 0.35: logistic regression / gradient boosting over
     encoder prob, verifier prob, rank, section frequency, document statistics, co-occurrence

  python f1boost.py fetch --eval models/final/eval_predictions.json --dev-years 2021 2022 --n-dev 400 --out data
  python f1boost.py dump  --data data --views full facts --out dump.jsonl         (GPU; models from ./models)
  python f1boost.py fit   --dump dump.jsonl --out report                           (CPU, minutes)
"""
from __future__ import annotations

import argparse, io, json, os, random, re, sys, tarfile, time, urllib.request
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BUCKET = "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com"


# ═══════════════════════════════ 1. fetch judgments (AWS Open Data) ═══════════════════════════════
def tar_urls(year):
    base = f"{BUCKET}/data/tar/year={year}/english"
    try:
        with urllib.request.urlopen(f"{base}/english.index.json", timeout=60) as r:
            names = sorted(set(re.findall(r"[\w.-]+\.tar", r.read().decode("utf-8", "ignore"))))
        if names:
            return [f"{base}/{n}" for n in names]
    except Exception:
        pass
    return [f"{base}/english.tar"]


def pdf_text(data: bytes) -> str:
    import pymupdf
    with pymupdf.open(stream=data, filetype="pdf") as d:
        return "\n".join(p.get_text() for p in d)


def cmd_fetch(a):
    from silver import silver_labels
    import torch
    ev = json.load(open(a.eval, encoding="utf-8"))
    test = {}
    for r in ev:
        if r["view"] == "full":
            test[r["key"].split("|")[0]] = r["gold"]
    labels = torch.load(os.path.join(a.models, "final", "hybrid_model.pt"), map_location="cpu", weights_only=False)["label_names"]
    os.makedirs(os.path.join(a.out, "pdfs"), exist_ok=True)
    rows, rnd = [], random.Random(13)
    need_test = dict(test)
    years = sorted(set(a.dev_years) | {int(k[:4]) for k in test})
    n_dev = 0
    for year in years:
        for url in tar_urls(year):
            if year not in a.dev_years and not any(k.startswith(str(year)) for k in need_test):
                break
            print("streaming", url, flush=True)
            with urllib.request.urlopen(url, timeout=900) as r, tarfile.open(fileobj=r, mode="r|*") as tar:
                for m in tar:
                    if not m.isfile() or not m.name.lower().endswith(".pdf"):
                        continue
                    base = os.path.basename(m.name)
                    is_test = base in need_test
                    want_dev = year in a.dev_years and n_dev < a.n_dev and rnd.random() < a.dev_rate
                    if not (is_test or want_dev):
                        continue
                    data = tar.extractfile(m).read()
                    try:
                        text = pdf_text(data)
                    except Exception:
                        continue
                    if is_test:
                        gold = need_test.pop(base)
                        split = "test"
                    else:
                        if not 800 <= len(text.split()) <= 25000:
                            continue
                        gold = sorted(silver_labels(text, labels))
                        if not gold:                                   # like training: judgments that cite IPC sections
                            continue
                        split, n_dev = "dev", n_dev + 1
                    open(os.path.join(a.out, "pdfs", base), "wb").write(data)
                    rows.append({"doc": base, "split": split, "gold": gold})
                    if len(rows) % 50 == 0:
                        print(f"  {len(rows)} judgments ({n_dev} dev, {len(test) - len(need_test)} test)", flush=True)
                    if n_dev >= a.n_dev and not any(k.startswith(str(year)) for k in need_test):
                        break
    json.dump(rows, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    print(f"done: {sum(r['split'] == 'dev' for r in rows)} dev, {sum(r['split'] == 'test' for r in rows)} test "
          f"({len(need_test)} test judgments not found: {list(need_test)[:5]})")


# ═══════════════════════════════ 2. dump model scores (GPU) ═══════════════════════════════
def cmd_dump(a):
    import torch
    paths = json.load(open(os.path.join(a.models, "paths.json")))
    P = {k: v if os.path.isabs(v) else os.path.join(a.models, v) for k, v in paths.items()}
    sys.path.insert(0, P["final"])
    import legal_hybrid as LH
    tok, llm, encoders, ck = LH.load_trained(P["final"], base_model=P["base_model"], legal_encoder=P["legal_encoder"])
    cfg, labels = ck["inference"], ck["label_names"]
    key = lambda s: (int(re.findall(r"\d+", s)[0]), s)
    split = json.load(open(os.path.join(a.data, "split.json")))
    done = set()
    if os.path.exists(a.out):
        done = {(json.loads(l)["doc"], json.loads(l)["view"]) for l in open(a.out, encoding="utf-8") if l.strip()}
    f = open(a.out, "a", encoding="utf-8")
    dev = next(encoders[0].parameters()).device
    t0, n = time.time(), 0

    def verify(passages, cands, evidence):
        acc = {c: 0.0 for c in cands}
        n_ord = max(1, cfg.get("verify_orders", 1))
        for k in range(n_ord):
            order = cands if k % 2 == 0 else cands[::-1]
            user = LH.verify_user(passages, order, ck["statute_desc"], evidence)
            for c, p in zip(order, LH.score_candidates(llm, tok, ck["verify_system"], user, order)):
                acc[c] += p / n_ord
        return acc

    i_sh, n_sh = (int(x) for x in a.shard.split("/"))
    for k_doc, r in enumerate(split):
        if k_doc % n_sh != i_sh:
            continue
        text = pdf_text(open(os.path.join(a.data, "pdfs", r["doc"]), "rb").read())
        masked = LH.hide_cited_sections(text)
        for view in a.views:
            if (r["doc"], view) in done:
                continue
            sents, flags = LH.view_sentences(LH.split_sentences(masked), view)
            doc = LH.prepare_doc(sents, tok, a.max_tokens, flags)
            with torch.no_grad(), torch.autocast("cuda", dtype=ck["dtype"], enabled=dev.type == "cuda"):
                p_enc, out = LH.ensemble_probs(encoders, ck["embedder"](doc.sentences).to(dev))
                p_enc = p_enc.float().cpu()
                order = torch.argsort(p_enc, descending=True).tolist()
                A = LH.shortlist(p_enc, cfg["k_min"], cfg["k_max"], cfg["min_p"])
                B = [j for j in order[:a.k_wide] if j not in A]
                passages = LH.key_passages(doc, out["scores"], cfg["passage_tokens"], cfg["head_tokens"])
                ev = {labels[j]: LH.evidence_for(doc, out["attn"], j) for j in A + B}
                ca = sorted([labels[j] for j in A], key=key)
                cb = sorted([labels[j] for j in B], key=key)
                pa = verify(passages, ca, ev)                       # exactly the deployed verifier call
                pb = verify(passages, cb, ev) if cb else {}         # the extra ranks, in a second call of the same form
            row = {"doc": r["doc"], "split": r["split"], "view": view, "gold": r["gold"], "tokens": len(doc.ids),
                   "p_enc": [round(float(x), 5) for x in p_enc], "rank": order[:a.k_wide],
                   "short": [labels[j] for j in A], "wide": [labels[j] for j in B],
                   "p_llm": {**{c: round(v, 5) for c, v in pa.items()}, **{c: round(v, 5) for c, v in pb.items()}}}
            f.write(json.dumps(row) + "\n"); f.flush(); n += 1
            if n % 20 == 0:
                print(f"  {n} done · {(time.time() - t0) / n:.1f} s each", flush=True)
    f.close()
    json.dump({"labels": labels, "inference": {k: v for k, v in cfg.items() if k != "class_thresholds"}},
              open(os.path.splitext(a.out)[0] + "_meta.json", "w"))
    print("dump complete")


# ═══════════════════════════════ 3. fit and evaluate (CPU) ═══════════════════════════════
def prf(gold, pred):
    tp = sum(len(g & p) for g, p in zip(gold, pred)); fp = sum(len(p - g) for g, p in zip(gold, pred))
    fn = sum(len(g - p) for g, p in zip(gold, pred))
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def macro(gold, pred, labels):
    f = []
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if c in g and c in p); fp = sum(1 for g, p in zip(gold, pred) if c not in g and c in p)
        fn = sum(1 for g, p in zip(gold, pred) if c in g and c not in p)
        if tp + fp + fn:
            f.append(2 * tp / (2 * tp + fp + fn))
    return sum(f) / len(f) if f else 0.0


def baseline_pred(r, labels, inf):
    """The deployed rule: fuse inside the 10-candidate shortlist, encoder alone outside, one threshold."""
    sc = {}
    for j, c in enumerate(labels):
        pe = r["p_enc"][j]
        sc[c] = inf["alpha"] * pe + (1 - inf["alpha"]) * r["p_llm"][c] if c in r["short"] else pe
    pred = {c for c, v in sc.items() if v >= inf["threshold"]}
    if not pred and inf.get("at_least_one"):
        pred = {max(sc, key=sc.get)}
    return pred, sc


def features(r, labels, prior, pmi):
    """One row per candidate in the wide list (encoder ranks 1-20)."""
    idx = {c: j for j, c in enumerate(labels)}
    cands = r["short"] + r["wide"]
    pe = r["p_enc"]
    top = sorted(pe, reverse=True)
    rows, names = [], []
    rank_of = {labels[j]: k for k, j in enumerate(r["rank"])}
    strong = [c for c in cands if pe[idx[c]] >= 0.3]
    for c in cands:
        e, l = pe[idx[c]], r["p_llm"].get(c, 0.5)
        co = max([pmi.get((c, o), 0.0) * pe[idx[o]] for o in strong if o != c] or [0.0])
        rows.append([e, l, e * l, 0.6 * e + 0.4 * l, rank_of.get(c, 20), float(c in r["short"]), prior.get(c, 0.0),
                     top[0], sum(1 for x in pe if x >= 0.3), len(r["short"]), min(1.0, r["tokens"] / 20000), co])
        names.append(c)
    return rows, names


def cmd_fit(a):
    import math
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    meta = json.load(open(os.path.splitext(a.dump[0])[0] + "_meta.json"))
    labels, inf = meta["labels"], meta["inference"]
    rows = [json.loads(l) for p in a.dump for l in open(p, encoding="utf-8") if l.strip()]
    os.makedirs(a.out, exist_ok=True)
    L = ["# Raising F1 without retraining: wider shortlist + learned combiner", "",
         "Everything below was chosen on DEV judgments (2021-22, silver labels from citations) and applied ONCE to the same "
         "150 TEST judgments as the paper (official labels).", ""]
    results = {}
    for view in sorted({r["view"] for r in rows}):
        dev = [r for r in rows if r["view"] == view and r["split"] == "dev"]
        test = [r for r in rows if r["view"] == view and r["split"] == "test"]
        if not dev or not test:
            continue
        g = lambda rs: [set(r["gold"]) for r in rs]
        # section prior + co-occurrence (PMI) from DEV labels only
        cnt, pair = Counter(), Counter()
        for r in dev:
            gs = sorted(set(r["gold"]))
            cnt.update(gs)
            pair.update((x, y) for x in gs for y in gs if x != y)
        N = len(dev)
        prior = {c: cnt[c] / N for c in labels}
        pmi = {k: max(0.0, math.log((v / N) / (prior[k[0]] * prior[k[1]] + 1e-9) + 1e-9)) / 5 for k, v in pair.items()}

        # 0. deployed rule (reproduction)
        base = [baseline_pred(r, labels, inf)[0] for r in test]
        res = {"Deployed v7 (k=10, 0.6/0.4 fusion, threshold 0.35)": base}
        # oracle ceilings (how much the shortlist limits us)
        cap10 = sum(len(set(r["gold"]) & set(r["short"])) for r in test) / max(1, sum(len(r["gold"]) for r in test))
        cap20 = sum(len(set(r["gold"]) & set(r["short"] + r["wide"])) for r in test) / max(1, sum(len(r["gold"]) for r in test))

        # 1. wider shortlist, same formula; alpha/threshold tuned on dev
        def wide_pred(r, al, th):
            sc = {}
            for j, c in enumerate(labels):
                pe = r["p_enc"][j]
                sc[c] = al * pe + (1 - al) * r["p_llm"][c] if c in r["p_llm"] else pe
            p = {c for c, v in sc.items() if v >= th}
            return p or {max(sc, key=sc.get)}
        best = max(((al, th) for al in [0.3, 0.4, 0.5, 0.6, 0.7] for th in [x / 100 for x in range(20, 61, 2)]),
                   key=lambda z: prf(g(dev), [wide_pred(r, *z) for r in dev])[2])
        res[f"Wider shortlist (k=20), fusion tuned on dev (alpha {best[0]}, threshold {best[1]})"] = [wide_pred(r, *best) for r in test]

        # 2. learned combiners over the wide candidate list (fit on dev, threshold by dev cross-validation)
        Xd, yd, gid = [], [], []
        for i, r in enumerate(dev):
            X, names = features(r, labels, prior, pmi)
            Xd += X; yd += [int(c in set(r["gold"])) for c in names]; gid += [i] * len(names)
        Xd, yd, gid = np.array(Xd), np.array(yd), np.array(gid)

        def fit_model(kind, X, y):
            m = (LogisticRegression(C=1.0, max_iter=2000, class_weight=None) if kind == "lr" else
                 HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0))
            return m.fit(X, y)

        def predict_docs(m, rs, th):
            out = []
            for r in rs:
                X, names = features(r, labels, prior, pmi)
                p = m.predict_proba(np.array(X))[:, 1]
                s = {c for c, v in zip(names, p) if v >= th}
                out.append(s or {names[int(np.argmax(p))]})
            return out

        for kind, label in (("lr", "logistic regression"), ("gb", "gradient boosting")):
            # 5-fold cross-validation on dev (folds by judgment) to choose the threshold honestly
            folds = np.arange(len(dev)) % 5
            oof = np.zeros(len(yd))
            for k in range(5):
                tr = ~np.isin(gid, np.where(folds == k)[0])
                m = fit_model(kind, Xd[tr], yd[tr])
                oof[~tr] = m.predict_proba(Xd[~tr])[:, 1]

            def dev_f1(th):
                pred = [set() for _ in dev]
                for i, r in enumerate(dev):
                    names = features(r, labels, prior, pmi)[1]
                    ps = oof[gid == i]
                    pred[i] = {c for c, v in zip(names, ps) if v >= th} or {names[int(np.argmax(ps))]}
                return prf(g(dev), pred)[2]
            th = max([x / 100 for x in range(15, 71, 2)], key=dev_f1)
            m = fit_model(kind, Xd, yd)
            res[f"Wider shortlist + learned combiner ({label}, threshold {th} from dev CV)"] = predict_docs(m, test, th)

        # report + bootstrap CI of the gain over the deployed rule
        L += [f"## View: {view} (dev {len(dev)} judgments, test {len(test)})", "",
              f"Correct sections reachable: {cap10:.1%} with the 10-candidate shortlist, {cap20:.1%} with 20.", "",
              "| System | Precision | Recall | Micro-F1 | Macro-F1 | Gain vs deployed [95% CI] |", "|---|---|---|---|---|---|"]
        rnd = random.Random(7)
        boot = [[rnd.randrange(len(test)) for _ in test] for _ in range(2000)]
        for name, pred in res.items():
            P, R, F = prf(g(test), pred)
            M = macro(g(test), pred, labels)
            if pred is base:
                ci = "–"
            else:
                d = sorted(prf([g(test)[i] for i in b], [pred[i] for i in b])[2] - prf([g(test)[i] for i in b], [base[i] for i in b])[2]
                           for b in boot)
                ci = f"{F - prf(g(test), base)[2]:+.3f} [{d[49]:+.3f}, {d[1949]:+.3f}]"
            L.append(f"| {name} | {P:.3f} | {R:.3f} | **{F:.3f}** | {M:.3f} | {ci} |")
            results[(view, name)] = F
        L.append("")
    open(os.path.join(a.out, "F1BOOST.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch"); f.add_argument("--eval", required=True); f.add_argument("--dev-years", nargs="+", type=int, default=[2021, 2022])
    f.add_argument("--n-dev", type=int, default=400); f.add_argument("--dev-rate", type=float, default=0.6)
    f.add_argument("--models", default=os.path.join(HERE, "models")); f.add_argument("--out", default="data")
    d = sp.add_parser("dump"); d.add_argument("--data", default="data"); d.add_argument("--views", nargs="+", default=["full"])
    d.add_argument("--models", default=os.path.join(HERE, "models")); d.add_argument("--k-wide", type=int, default=20)
    d.add_argument("--max-tokens", type=int, default=65536); d.add_argument("--out", default="dump.jsonl")
    d.add_argument("--shard", default="0/1", help="i/n: every n-th judgment (one process per GPU)")
    t = sp.add_parser("fit"); t.add_argument("--dump", nargs="+", default=["dump.jsonl"]); t.add_argument("--out", default="report")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "dump": cmd_dump, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
