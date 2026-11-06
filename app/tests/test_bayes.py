"""tests/test_bayes.py · run: python -m tests.test_bayes   (add --pgmpy <python-with-pgmpy> to cross-check against pgmpy)"""
import json, os, random, subprocess, sys
from ipc import bayes as B


def check_scenarios():
    for label, ev, expected in B.SCENARIOS:
        r = B.route_query(ev)
        assert r["route"] == B.threshold_route(r["p_hallucination_high"]), label
        assert r["route"] == expected, f"{label}: got {r['route']} expected {expected}"
        print(f"OK  {label:<42} P(H)={r['p_hallucination_high']:.3f} -> {r['route']}")


def check_raw():
    raw = {"word_count": 9, "question_type": "factoid", "named_entity_count": 2, "sub_question_count": 1,
           "top1_faiss_distance": 0.35, "top1_rerank_score": 0.18, "relevant_chunk_count": 1,
           "mean_source_authority": 0.7, "is_ambiguous_followup": False, "unresolved_reference_count": 0,
           "entity_monthly_views": 250_000, "is_time_sensitive": True, "kb_domain_similarity": 0.6}
    ev = B.extract_evidence(raw)
    assert "QueryReformulation" not in ev and ev["RerankerScore"] == "High" and ev["TemporalSensitivity"] == "High"
    print("OK  raw measurements ->", B.route_query(ev)["route"])


def random_cases(n=40, seed=1):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        out.append({k: rnd.choice(["Low", "High"]) for k in B.INPUT_NODES if rnd.random() < 0.6})
    return out


def dump_for_pgmpy(path):
    cases = random_cases()
    json.dump([{"evidence": c, "ours": B.route_query(c)} for c in cases], open(path, "w"))


def check_pgmpy(py):
    here = os.path.dirname(os.path.abspath(__file__))
    ref = os.path.join(here, "..", "..", "..") if False else None
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "bayes_cases.json")
    dump_for_pgmpy(tmp)
    code = r'''
import json, sys
sys.argv = ["x"]
from itertools import product
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import importlib.util
spec = importlib.util.spec_from_file_location("b", sys.stdin.readline().strip()); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
ST = ["Low", "High"]
def root(n, p): return TabularCPD(n, 2, [[1 - p], [p]], state_names={n: ST})
def w(n, parents, ws, base):
    col = [min(max(base + sum(x for bit, x in zip(c, ws) if bit), 0.02), 0.98) for c in product([0, 1], repeat=len(parents))]
    return TabularCPD(n, 2, [[1 - h for h in col], col], evidence=parents, evidence_card=[2] * len(parents), state_names={n: ST, **{p: ST for p in parents}})
m = DiscreteBayesianNetwork(); cpds = []
for bn, inp in b.MODULES.items():
    names = list(inp); m.add_edges_from((x, bn) for x in names)
    cpds += [root(x, v[0]) for x, v in inp.items()]; cpds.append(w(bn, names, [v[1] for v in inp.values()], b.BOTTLENECK_BASE))
bs = list(b.CONVERGENCE_WEIGHTS); m.add_edges_from((x, "HallucinationRisk") for x in bs)
cpds.append(w("HallucinationRisk", bs, list(b.CONVERGENCE_WEIGHTS.values()), b.RISK_BASE))
m.add_edge("HallucinationRisk", "OptimalRoute")
cpds.append(TabularCPD("OptimalRoute", 3, [[0.5, 0.0], [0.4, 0.4], [0.1, 0.6]], evidence=["HallucinationRisk"], evidence_card=[2],
                       state_names={"OptimalRoute": b.ROUTES, "HallucinationRisk": ST}))
m.add_cpds(*cpds); assert m.check_model()
inf = VariableElimination(m); worst = 0.0
for c in json.load(open(sys.stdin.readline().strip())):
    r = inf.query(["HallucinationRisk"], evidence=c["evidence"], show_progress=False)
    p = float(r.values[r.state_names["HallucinationRisk"].index("High")])
    rb = inf.query(["OptimalRoute"], evidence=c["evidence"], show_progress=False)
    d = {k: float(rb.values[rb.state_names["OptimalRoute"].index(k)]) for k in b.ROUTES}
    worst = max(worst, abs(p - c["ours"]["p_hallucination_high"]), *(abs(d[k] - c["ours"]["route_distribution"][k]) for k in d))
    for bn in bs:      # blame = P(bottleneck=High | evidence, risk=High)
        q = inf.query([bn], evidence={**c["evidence"], "HallucinationRisk": "High"}, show_progress=False)
        worst = max(worst, abs(float(q.values[q.state_names[bn].index("High")]) - c["ours"]["blame"][bn]))
print("max abs diff vs pgmpy over", len(json.load(open(sys.stdin.readline().strip()))), "cases:", worst); assert worst < 1e-3, worst
'''
    src = os.path.join(here, "..", "ipc", "bayes.py")
    r = subprocess.run([py, "-c", code], input=f"{src}\n{tmp}\n{tmp}\n", text=True, capture_output=True)
    print(r.stdout.strip() or r.stderr.strip()[-800:])
    assert r.returncode == 0


if __name__ == "__main__":
    check_scenarios(); check_raw()
    if "--pgmpy" in sys.argv:
        check_pgmpy(sys.argv[sys.argv.index("--pgmpy") + 1])
    print("bayes tests passed")
