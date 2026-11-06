"""learn_router.py · learn the IPC router (prob_reas learn_bn method) for one embedder + chat model.

  python learn_router.py --transfer-only bench_out/stress_results.json
        start from the prob_reas 6,000-question profile (same LLM: Qwen2.5-7B); vector-distance and domain-similarity
        cut-offs re-mapped to InLegalBERT on these results; nothing learned from IPC labels yet
  python learn_router.py bench_out/stress_results.json --labels labels_run2.json
        MAP-EM on IPC labelled answers with that profile as the prior; tau chosen so served answers fail <= 20%

Writes router_calibration/inlegalbert__<llm>.json (prob_reas profile format). The router loads it whenever that chat
model is active (or set IPC_ROUTER_PROFILE=<file>). Delete the file to go back to the hand-set network.
Labels: the benchmark's automatic checks, overridden by --labels {id: true/false} (true = answer was right).
"""
import argparse, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ipc import learned_bn as L
from ipc.retriever import CALIB_DIR, EMBEDDER, model_slug

PRIOR_GLOB = [os.path.join(HERE, "..", "bayesian_epistemic_routing_final", "profiles", "*__combined.json"),
              os.path.join(HERE, "router_calibration", "prob_reas_*__combined.json")]


def load_rows(files, labels):
    rows, model, rer = [], None, set()
    for f in files:
        if f.endswith(".jsonl"):                     # label_ipc.py output: one graded answer per line
            for line in open(f, encoding="utf-8"):
                q = json.loads(line) if line.strip() else None
                if not q or not q.get("raw") or q.get("error") or q.get("empty") or q.get("excluded"):
                    continue
                model = model or q.get("model")
                rer.add(q["raw"].get("reranker", "lexical"))
                rows.append({"id": q["id"], "raw": q["raw"], "correct": bool(labels.get(q["id"], q["correct"]))})
            continue
        d = json.load(open(f, encoding="utf-8"))
        model = model or d.get("model")
        for doc in d["docs"]:
            for q in doc["questions"] + doc.get("chain", []):
                raw = (q.get("router") or {}).get("raw")
                if not raw or q.get("error"):
                    continue
                rer.add(raw.get("reranker", "lexical"))
                rows.append({"id": q["id"], "raw": raw, "correct": bool(labels.get(q["id"], q["ok"]))})
    return rows, model, rer


def auc(p, y):
    if y.min() == y.max():
        return float("nan")
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(profile, rows, y):
    R = L.Router(profile)
    p = np.array([R.route(r["raw"])["p_hallucination_high"] for r in rows])
    routes = [R.route(r["raw"])["route"] for r in rows]
    served = np.array([x == "Serve" for x in routes])
    return {"auroc": round(auc(p, y), 3), "served": int(served.sum()),
            "served_fail": round(float(y[served].mean()), 3) if served.any() else None,
            "routes": {k: routes.count(k) for k in ("Serve", "Search", "Clarify")}}


def learn(prior_profile, rows, y, args):
    prior_cuts = prior_profile["cutoffs"]
    ignore = list(prior_profile["meta"].get("ignore", []))
    cuts = L.learn_cutoffs(rows, y, L.LEVELS, prior_cuts=prior_cuts, min_rows=args.min_rows)
    X = L.evidence_matrix(rows, cuts, ignore)
    params = L.em(X, y, L.LEVELS, prior=L.params_from_profile(prior_profile), prior_strength=args.prior_strength)
    p = L.FastBN(params, L.LEVELS).p_fail(X)
    tau = L.choose_threshold(p, y, args.target_fail) or prior_profile["meta"].get("serve_below", 0.2)
    return params, cuts, float(tau), ignore


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--labels")
    ap.add_argument("--prior", help="prob_reas profile (default: ../bayesian_epistemic_routing_final/profiles/*__combined.json)")
    ap.add_argument("--transfer-only", action="store_true")
    ap.add_argument("--prior-strength", type=float, default=L.PRIOR_STRENGTH)
    ap.add_argument("--target-fail", type=float, default=0.07, help="served answers may fail at most this often (IPC base rate ~12%%; 0.1 already serves ~97%% and leaves the wider Search route unused, which the stress test showed costs answers)")
    ap.add_argument("--min-rows", type=int, default=300, help="fewer measured values than this: keep the prior cut-offs")
    ap.add_argument("--model")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true", help="write the profile even if the checks fail")
    a = ap.parse_args()

    prior_path = a.prior or next((p for g in PRIOR_GLOB for p in sorted(glob.glob(g))), None)
    if not prior_path:
        raise SystemExit("No prob_reas profile found. Pass --prior <path to ..._combined.json>.")
    prior_raw = json.load(open(prior_path, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8")) if a.labels else {}
    rows, model, rer = load_rows(a.results, labels)
    model = a.model or model or "qwen2.5:7b"
    if not rows:
        raise SystemExit("No router data in these results. Run bench_models.py first.")
    y = np.array([int(not r["correct"]) for r in rows])
    print(f"prior: {os.path.basename(prior_path)} ({prior_raw['meta'].get('n')} questions, LLM {prior_raw['meta'].get('llm_profile', {}).get('llm')})")
    print(f"IPC data: {len(rows)} answers, {y.sum()} failed, reranker in these results: {', '.join(sorted(rer))}")

    transfer = L.transfer_profile(prior_raw, rows, EMBEDDER)
    if "cross-encoder" not in rer:
        # reranker scores in these results are the old lexical relevance, not the cross-encoder the prior was learned with
        transfer["meta"]["ignore"] = sorted(set(transfer["meta"]["ignore"]) | {"RerankerScore", "ChunkDensity"})
        print("  ! these results have no cross-encoder scores: RerankerScore and ChunkDensity left unobserved. "
              "Install the reranker (download_reranker.py) and re-run the benchmark.")
    for n in L.EMBEDDER_DEPENDENT:
        c = transfer["cutoffs"][n]
        print(f"  {n:<20} cut-offs {'re-mapped to InLegalBERT' if c.get('quantile_mapped') else 'kept'}: {[round(x, 3) for x in c['cuts']]}")

    rep = {"hand-set": None, "transferred": evaluate(transfer, rows, y)}
    if a.transfer_only:
        profile = transfer
        profile["meta"].update({"llm": model, "n_ipc": 0})
    else:
        params, cuts, tau, ignore = learn(transfer, rows, y, a)
        meta = {**transfer["meta"], "llm": model, "n_ipc": len(rows), "fail_rate_ipc": round(float(y.mean()), 3),
                "serve_below": round(tau, 4), "target_fail": a.target_fail, "prior_strength": a.prior_strength,
                "prior_file": os.path.basename(prior_path), "ignore": ignore}
        profile = L.to_profile(params, cuts, L.LEVELS, meta)
        rep["learned (fit)"] = evaluate(profile, rows, y)
        # 5-fold cross-validation: learn on 4/5, route the held-out fifth
        # folds by JUDGMENT (id "doc:..."), so questions about one judgment are never in both halves
        docs = sorted({r["id"].split(":")[0] for r in rows})
        fold_of = {d: k % 5 for k, d in enumerate(np.random.default_rng(0).permutation(docs))}
        fold = np.array([fold_of[r["id"].split(":")[0]] for r in rows])
        pcv = np.zeros(len(rows)); rcv = [""] * len(rows)
        for f in range(5):
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
            if not len(te) or not len(tr):
                continue
            pr, cu, ta, ig = learn(transfer, [rows[i] for i in tr], y[tr], a)
            Rf = L.Router(L.to_profile(pr, cu, L.LEVELS, {**meta, "serve_below": ta, "ignore": ig}))
            for i in te:
                out = Rf.route(rows[i]["raw"]); pcv[i] = out["p_hallucination_high"]; rcv[i] = out["route"]
        s = np.array([x == "Serve" for x in rcv])
        rep["learned (5-fold CV)"] = {"auroc": round(auc(pcv, y), 3), "served": int(s.sum()),
                                      "served_fail": round(float(y[s].mean()), 3) if s.any() else None,
                                      "routes": {k: rcv.count(k) for k in ("Serve", "Search", "Clarify")}}
    from ipc import bayes as B                                      # the hand-set network, for comparison
    from ipc.retriever import CUTS
    ph = np.array([B.route_query(B.extract_evidence(r["raw"], CUTS))["p_hallucination_high"] for r in rows])
    rh = [B.threshold_route(x) for x in ph]
    sh = np.array([x == "Serve" for x in rh])
    rep["hand-set"] = {"auroc": round(auc(ph, y), 3), "served": int(sh.sum()), "served_fail": round(float(y[sh].mean()), 3) if sh.any() else None,
                       "routes": {k: rh.count(k) for k in ("Serve", "Search", "Clarify")}}

    print(f"\n{'':22}{'AUROC':>7}{'served':>8}{'fail among served':>19}   routes")
    for k, v in rep.items():
        print(f"{k:22}{v['auroc']:>7}{v['served']:>8}{str(v['served_fail']):>19}   {v['routes']}")
    print("  (AUROC: does P(fail) rank wrong answers above right ones; 0.5 = chance. Served answers should fail <= "
          f"{a.target_fail:.0%}.)")
    if a.dry:
        return
    # Only activate a profile that the data supports: one learned with the same reranker as the prior, that serves
    # questions, and whose served answers (cross-validated) fail close to the target rate.
    check = rep["transferred"] if a.transfer_only else rep["learned (5-fold CV)"]
    problems = []
    if "cross-encoder" not in rer:
        problems.append("the results were measured without the cross-encoder reranker")
    if not check["served"]:
        problems.append("it would serve no question at all")
    elif check["served_fail"] is not None and check["served_fail"] > a.target_fail + 0.1:
        problems.append(f"served answers fail {check['served_fail']:.0%} on held-out data (target {a.target_fail:.0%})")
    if problems and not a.force:
        print("\nNOT activated: " + "; ".join(problems) + ".\nThe router keeps the hand-set network. More labelled answers "
              "fix this; --force writes it anyway.")
        return
    os.makedirs(CALIB_DIR, exist_ok=True)
    out = os.path.join(CALIB_DIR, f"{EMBEDDER}__{model_slug(model)}.json")
    json.dump(profile, open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nwritten {out}  (serve threshold tau = {profile['meta'].get('serve_below')})")


if __name__ == "__main__":
    main()
