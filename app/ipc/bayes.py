"""bayes.py · the 20-node Bayesian epistemic router, ported from bayesian_epistemic_routing.py.

Same structure, priors, weights, bases and route thresholds as the original (14 inputs -> 4 bottlenecks ->
HallucinationRisk -> OptimalRoute), plus 4 optional inputs (question kind, document-relative match) whose weight is 0 until a
calibration file fits them. route_query(evidence, params) takes fitted weights / bases / thresholds from
router_calibration/<embedder>__<llm>.json (see calibrate_router.py); without params it is the original network. pgmpy is not needed: the inputs are independent roots and every bottleneck owns a
disjoint set of them, so exact inference is a short enumeration. tests/test_bayes.py cross-checks it against pgmpy.
"""
from itertools import product

ROUTES = ["Serve", "Search", "Clarify"]

# bottleneck -> {input: (prior P(High), weight into the bottleneck, meaning of "High")}
MODULES = {
    "QueryAmbiguity": {
        "QueryLength":       (0.30, 0.30, "very short query (< 4 words)"),
        "QuestionType":      (0.35, 0.35, "open-ended / opinion / compound question"),
        "EntitySpecificity": (0.40, 0.30, "no named entities in the query"),
        "TopicComplexity":   (0.25, 0.30, "multi-hop: needs 2+ separate facts"),
    },
    "ContextDeficit": {
        "FAISSVectorDistance": (0.35, 0.20, "top-1 chunk is far from the query"),
        "RerankerScore":       (0.35, 0.35, "retrieved chunks are not relevant"),
        "ChunkDensity":        (0.40, 0.20, "fewer than 2 of top-k chunks pass the relevance cutoff"),
        "SourceAuthority":     (0.30, 0.15, "chunks come from low-authority sources"),
        "LowRelativeMatch":    (0.30, 0.00, "best sentence barely stands out from the rest of the judgment (calibrated only)"),
        "ListQuestion":        (0.15, 0.00, "asks for a list: all conditions / every direction (calibrated only)"),
        "NumericQuestion":     (0.20, 0.00, "asks for a number, date or count (calibrated only)"),
    },
    "IntentUncertainty": {
        "FollowUpStatus":       (0.30, 0.40, "follow-up that depends on unclear earlier context"),
        "QueryReformulation":   (0.20, 0.35, "user is rephrasing: last answer missed"),
        "UnresolvedReferences": (0.25, 0.45, "pronouns like 'it' / 'that one' with no clear target"),
    },
    "KnowledgeGap": {
        "EntityPopularity":    (0.30, 0.30, "rare / long-tail entity"),
        "TemporalSensitivity": (0.20, 0.30, "asks about recent events"),
        "DomainShift":         (0.25, 0.30, "query is outside the knowledge base's domain"),
        "LawQuestion":         (0.30, 0.00, "general legal question, not a fact of this case (calibrated only)"),
    },
}
CONVERGENCE_WEIGHTS = {"QueryAmbiguity": 0.70, "ContextDeficit": 0.22, "IntentUncertainty": 0.65, "KnowledgeGap": 0.22}
BOTTLENECK_BASE = 0.03
RISK_BASE = 0.03
INPUT_NODES = [n for m in MODULES.values() for n in m]
OPEN_ENDED_TYPES = {"open_ended", "opinion", "compound", "how", "why"}

# raw measurement -> "High" rule. The retrieval cut-offs (FAISS distance, relevance) are calibrated for InLegalBERT
# cosine on legal_kb in retriever.py and passed in through `cuts`; these defaults are the original MiniLM values.
DEFAULT_CUTS = {"faiss": 0.8, "rerank": 0.5, "views": 1000, "domain": 0.3, "relz": 2.0}
THRESHOLDS = {"serve_below": 0.2, "clarify_above": 0.6}


def evidence_rules(cuts=None):
    c = {**DEFAULT_CUTS, **(cuts or {})}
    return {
        "QueryLength":          ("word_count",                lambda v: v < 4),
        "QuestionType":         ("question_type",             lambda v: v in OPEN_ENDED_TYPES),
        "EntitySpecificity":    ("named_entity_count",        lambda v: v == 0),
        "TopicComplexity":      ("sub_question_count",        lambda v: v >= 2),
        "FAISSVectorDistance":  ("top1_faiss_distance",       lambda v: v > c["faiss"]),
        "RerankerScore":        ("top1_rerank_score",         lambda v: v < c["rerank"]),
        "ChunkDensity":         ("relevant_chunk_count",      lambda v: v < 2),
        "SourceAuthority":      ("mean_source_authority",     lambda v: v < 0.5),
        "FollowUpStatus":       ("is_ambiguous_followup",     lambda v: bool(v)),
        "QueryReformulation":   ("similarity_to_prev_query",  lambda v: v > 0.8),
        "UnresolvedReferences": ("unresolved_reference_count", lambda v: v >= 1),
        "EntityPopularity":     ("entity_monthly_views",      lambda v: v < c["views"]),
        "TemporalSensitivity":  ("is_time_sensitive",         lambda v: bool(v)),
        "DomainShift":          ("kb_domain_similarity",      lambda v: v < c["domain"]),
        "LowRelativeMatch":     ("top1_cos_z",                lambda v: v < c["relz"]),
        "ListQuestion":         ("question_kind",             lambda v: v == "list"),
        "NumericQuestion":      ("question_kind",             lambda v: v == "number"),
        "LawQuestion":          ("question_kind",             lambda v: v == "law"),
    }


def extract_evidence(raw: dict, cuts=None) -> dict:
    """Raw measurements -> {node: "Low"|"High"}. Missing (None) measurements are left out and marginalised."""
    ev = {}
    for node, (key, is_high) in evidence_rules(cuts).items():
        v = raw.get(key)
        if v is not None:
            ev[node] = "High" if is_high(v) else "Low"
    return ev


def _cpd_high(bits, weights, base):
    return min(max(base + sum(w for b, w in zip(bits, weights) if b), 0.02), 0.98)


def _prior(name):
    for m in MODULES.values():
        if name in m:
            return m[name][0]
    raise KeyError(name)


def _p_input_high(name, evidence):
    st = evidence.get(name)
    return 1.0 if st == "High" else (0.0 if st == "Low" else _prior(name))


def _bottleneck(names, ws, base, evidence):
    """P(bottleneck=High) = sum over input states with non-zero probability (observed inputs have one state)."""
    states = []
    for n in names:
        p = _p_input_high(n, evidence)
        states.append([(b, q) for b, q in ((1, p), (0, 1 - p)) if q > 0])
    tot = 0.0
    for combo in product(*states):
        pr = 1.0
        for _, q in combo:
            pr *= q
        tot += pr * _cpd_high([b for b, _ in combo], ws, base)
    return tot


def resolve_params(params=None):
    """Original constants, overridden by a calibration file's values where given."""
    p = params or {}
    w = p.get("weights", {})
    return {"weights": {b: {n: w.get(n, v[1]) for n, v in inputs.items()} for b, inputs in MODULES.items()},
            "convergence": {**CONVERGENCE_WEIGHTS, **p.get("convergence", {})},
            "bottleneck_base": p.get("bottleneck_base", BOTTLENECK_BASE), "risk_base": p.get("risk_base", RISK_BASE),
            "thresholds": {**THRESHOLDS, **p.get("thresholds", {})}}


def p_fail(evidence: dict, params=None) -> float:
    """P(HallucinationRisk=High | evidence): the fast path used by calibration."""
    r = resolve_params(params)
    pb = {b: _bottleneck(list(inp), [r["weights"][b][n] for n in inp], r["bottleneck_base"], evidence) for b, inp in MODULES.items()}
    bs = list(MODULES)
    rw = [r["convergence"][b] for b in bs]
    tot = 0.0
    for bits in product([0, 1], repeat=len(bs)):
        pr = 1.0
        for b, bit in zip(bs, bits):
            pr *= pb[b] if bit else 1 - pb[b]
        tot += pr * _cpd_high(bits, rw, r["risk_base"])
    return tot


def route_query(evidence: dict, params=None) -> dict:
    """Exact inference. Returns P(HallucinationRisk=High), the route distribution, the chosen route, and
    `bottlenecks` (P(bottleneck=High)) plus `blame` (P(bottleneck=High | HallucinationRisk=High): who is to blame)."""
    unknown = set(evidence) - set(INPUT_NODES)
    if unknown:
        raise ValueError(f"Unknown evidence variables: {sorted(unknown)}")
    r = resolve_params(params)
    pb = {b: _bottleneck(list(inp), [r["weights"][b][n] for n in inp], r["bottleneck_base"], evidence) for b, inp in MODULES.items()}

    bs, rw = list(MODULES), [r["convergence"][b] for b in MODULES]
    p_risk, joint_high = 0.0, {b: 0.0 for b in bs}
    for bits in product([0, 1], repeat=len(bs)):
        pr = 1.0
        for b, bit in zip(bs, bits):
            pr *= pb[b] if bit else 1 - pb[b]
        ph = pr * _cpd_high(bits, rw, r["risk_base"])
        p_risk += ph
        for b, bit in zip(bs, bits):
            if bit:
                joint_high[b] += ph
    blame = {b: (joint_high[b] / p_risk if p_risk else 0.0) for b in bs}

    dist = {"Serve": 0.5 * (1 - p_risk), "Search": 0.4, "Clarify": 0.1 * (1 - p_risk) + 0.6 * p_risk}
    route = threshold_route(p_risk, r["thresholds"])          # identical to argmax(dist) at the default 0.2 / 0.6
    return {"p_hallucination_high": round(p_risk, 4), "route": route,
            "route_distribution": {k: round(v, 4) for k, v in dist.items()},
            "bottlenecks": {b: round(v, 4) for b, v in pb.items()},
            "blame": {b: round(v, 4) for b, v in blame.items()}}


def threshold_route(p_high: float, thresholds=None) -> str:
    t = {**THRESHOLDS, **(thresholds or {})}
    return "Serve" if p_high < t["serve_below"] else ("Search" if p_high <= t["clarify_above"] else "Clarify")


def _all(state):
    return {n: state for n in INPUT_NODES}


SCENARIOS = [
    ("Everything looks fine", _all("Low"), "Serve"),
    ("Clear query, but retrieval is poor", {**_all("Low"), "FAISSVectorDistance": "High", "RerankerScore": "High", "ChunkDensity": "High"}, "Search"),
    ("Clear query about a recent / rare topic", {**_all("Low"), "TemporalSensitivity": "High", "EntityPopularity": "High"}, "Search"),
    ("Vague, open-ended query with no entities", {**_all("Low"), "QueryLength": "High", "QuestionType": "High", "EntitySpecificity": "High"}, "Clarify"),
    ("Follow-up full of unresolved pronouns", {**_all("Low"), "FollowUpStatus": "High", "UnresolvedReferences": "High", "QueryReformulation": "High"}, "Clarify"),
    ("Everything looks bad", _all("High"), "Clarify"),
]

# static description of the graph for the architecture page
GRAPH = {"modules": {b: {n: {"prior": p, "weight": w, "meaning": m} for n, (p, w, m) in inputs.items()} for b, inputs in MODULES.items()},
         "convergence": CONVERGENCE_WEIGHTS, "thresholds": {"serve_below": 0.2, "clarify_above": 0.6}}
