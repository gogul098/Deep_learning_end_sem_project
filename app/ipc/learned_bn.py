"""learned_bn.py · the LEARNED Bayesian router, ported from prob_reas/learn_bn.py (same maths, same profile format).

Structure: the hand-designed 14 inputs -> 4 latent bottlenecks -> HallucinationRisk (ipc/bayes.py, without the 4
extra inputs the IPC port added, so prob_reas profiles load unchanged). What is learned from labelled answers:
  * cut-offs per signal by information gain (strong retrieval signals get 3-4 levels),
  * every probability table by MAP-EM (bottlenecks latent); a PRIOR profile (default: the hand-set network, or the
    6,000-question prob_reas profile for the same LLM) gives Dirichlet pseudo-counts,
  * the serve threshold tau: largest P(fail) cut-off at which served answers fail at most `target_fail` of the time.
Routing (decide): Serve if P(fail) < tau; else Clarify if the QUESTION is to blame
(P(QueryAmbiguity or IntentUncertainty High | evidence, fail) >= 0.5), else Search.

transfer_profile(): builds a starting profile for a new embedder from a prob_reas profile. Signals measured the same way
keep their cut-offs (cross-encoder reranker score, chunk density, query length...); embedder-dependent ones (vector
distance, domain similarity) are re-cut so their levels keep the same shares on this system's data (quantile mapping);
signals that mean something different here are left unobserved.
"""
from __future__ import annotations

import json
import math
from itertools import product

import numpy as np

from . import bayes as ber

ORIGINAL = ["QueryLength", "QuestionType", "EntitySpecificity", "TopicComplexity",
            "FAISSVectorDistance", "RerankerScore", "ChunkDensity", "SourceAuthority",
            "FollowUpStatus", "QueryReformulation", "UnresolvedReferences",
            "EntityPopularity", "TemporalSensitivity", "DomainShift"]
MODULES = {m: {n: v for n, v in ber.MODULES[m].items() if n in ORIGINAL} for m in ber.MODULES}
MODULE_NAMES = list(MODULES)
INPUTS = [i for m in MODULE_NAMES for i in MODULES[m]]
BOTTLENECKS = list(ber.CONVERGENCE_WEIGHTS)

# raw key, direction (+1: trouble when LARGE, -1: trouble when SMALL, None: yes/no rule)  — as in prob_reas
SIGNALS = {
    "QueryLength":          ("word_count", -1),
    "QuestionType":         ("question_type", None),
    "EntitySpecificity":    ("named_entity_count", -1),
    "TopicComplexity":      ("sub_question_count", +1),
    "FAISSVectorDistance":  ("top1_faiss_distance", +1),
    "RerankerScore":        ("top1_rerank_score", -1),
    "ChunkDensity":         ("relevant_chunk_count", -1),
    "SourceAuthority":      ("mean_source_authority", -1),
    "FollowUpStatus":       ("is_ambiguous_followup", None),
    "QueryReformulation":   ("similarity_to_prev_query", +1),
    "UnresolvedReferences": ("unresolved_reference_count", +1),
    "EntityPopularity":     ("entity_monthly_views", -1),
    "TemporalSensitivity":  ("is_time_sensitive", None),
    "DomainShift":          ("kb_domain_similarity", -1),
}
# IPC hand-set cut-offs (InLegalBERT calibrated values from retriever.CUTS), "value < cut" / "value > cut" form
DEFAULT_CUTS = {"QueryLength": 4, "EntitySpecificity": 1, "TopicComplexity": 1.5, "FAISSVectorDistance": 0.34,
                "RerankerScore": 0.5, "ChunkDensity": 2, "SourceAuthority": 0.5, "QueryReformulation": 0.8,
                "UnresolvedReferences": 0.5, "EntityPopularity": 1000, "DomainShift": 0.64}
LEVELS = {"RerankerScore": 4, "FAISSVectorDistance": 3, "ChunkDensity": 3, "EntityPopularity": 3, "DomainShift": 3}
PRIOR_STRENGTH = 5.0
# measured differently in the IPC port (no page-view data; InLegalBERT similarities between turns are all high):
# left unobserved (marginalised) unless learned from IPC data
NOT_COMPARABLE = ["EntityPopularity", "QueryReformulation"]
EMBEDDER_DEPENDENT = ["FAISSVectorDistance", "DomainShift"]


def state_names(k: int) -> list:
    return {2: ["Low", "High"], 3: ["Low", "Medium", "High"], 4: ["Low", "Medium", "Elevated", "High"]}.get(k, [f"L{i}" for i in range(k)])


def _entropy(p):
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def info_gain(level, y):
    g = _entropy(y.mean())
    for l in np.unique(level):
        m = level == l
        g -= m.mean() * _entropy(y[m].mean())
    return g


def _level(vals, cuts, direction):
    return sum(((vals < c) if direction < 0 else (vals > c)).astype(int) for c in cuts) if cuts else np.zeros(len(vals), dtype=int)


def _vals(rows, key):
    return np.array([r["raw"].get(key) if isinstance(r["raw"].get(key), (int, float)) and not isinstance(r["raw"].get(key), bool)
                     else np.nan for r in rows], dtype=float).round(6)


def learn_cutoffs(rows, y, levels=None, prior_cuts=None, min_rows=300, min_share=0.02):
    """Information-gain cut-offs (prob_reas method). With fewer than `min_rows` measured values a signal keeps its
    prior cut-offs: on small data the gain of a cut is mostly noise."""
    levels = LEVELS if levels is None else levels
    prior_cuts = prior_cuts or {}
    cuts = {}
    for node, (key, direction) in SIGNALS.items():
        if direction is None:
            continue
        k = levels.get(node, 2)
        cuts[node] = prior_cuts.get(node) or {"cuts": [DEFAULT_CUTS[node]], "learned": False, "info_gain": 0.0}
        vals = _vals(rows, key)
        ok = ~np.isnan(vals)
        if ok.sum() < min_rows or np.unique(vals[ok]).size < 2:
            continue
        v, yy = vals[ok], y[ok]
        cand = np.unique(np.quantile(v, np.linspace(0.02, 0.98, 49)))
        chosen, gain = [], 0.0
        for _ in range(k - 1):
            best = (gain, None)
            for q in cand:
                if q in chosen:
                    continue
                lv = _level(v, chosen + [q], direction)
                shares = np.bincount(lv, minlength=len(chosen) + 2) / len(lv)
                if shares.min() < min_share:
                    continue
                g = info_gain(lv, yy)
                if g > best[0] + 1e-3:
                    best = (g, float(q))
            if best[1] is None:
                break
            gain = best[0]
            chosen.append(best[1])
        if chosen:
            cuts[node] = {"cuts": sorted(chosen, reverse=(direction < 0)), "learned": True, "info_gain": round(gain, 4)}
    return cuts


def discretize(raw: dict, cuts: dict, ignore=()) -> dict:
    ev = {}
    for node, (key, direction) in SIGNALS.items():
        if node in ignore:
            continue
        v = raw.get(key)
        if direction is None:
            if node == "QuestionType":
                if v is not None:
                    ev[node] = int(v in ber.OPEN_ENDED_TYPES)
            elif v is not None:
                ev[node] = int(bool(v))
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        c = cuts.get(node, {}).get("cuts", [DEFAULT_CUTS[node]])
        ev[node] = int(_level(np.array([float(v)]), c, direction)[0])
    return ev


def evidence_matrix(rows, cuts, ignore=()):
    X = np.full((len(rows), len(INPUTS)), -1, dtype=int)
    for r_i, r in enumerate(rows):
        ev = discretize(r["raw"], cuts, ignore)
        for c_i, node in enumerate(INPUTS):
            if node in ev:
                X[r_i, c_i] = ev[node]
    return X


def configs(card_list):
    return np.array(list(product(*[range(k) for k in card_list])), dtype=int)


def handset_params(levels=None):
    """The hand-designed network as arrays (extra levels share the 'High' weight evenly) — prob_reas definition."""
    levels = {} if levels is None else levels
    clip = lambda p: min(max(p, 0.02), 0.98)
    pi = []
    for m in MODULE_NAMES:
        for n, (p_high, _, _) in MODULES[m].items():
            k = levels.get(n, 2)
            pi.append(np.array([1 - p_high] + [p_high / (k - 1)] * (k - 1)))
    theta = []
    for m in MODULE_NAMES:
        names = list(MODULES[m])
        w = [MODULES[m][n][1] for n in names]
        ks = [levels.get(n, 2) for n in names]
        theta.append(np.array([clip(ber.BOTTLENECK_BASE + sum(wi * c / (k - 1) for c, wi, k in zip(cfg, w, ks)))
                               for cfg in configs(ks)]))
    w = list(ber.CONVERGENCE_WEIGHTS.values())
    phi = np.array([clip(ber.RISK_BASE + sum(wi for b, wi in zip(c, w) if b)) for c in configs([2] * 4)])
    return {"pi": pi, "theta": theta, "phi": phi}


class FastBN:
    """Exact inference over the 16 bottleneck combinations (prob_reas FastBN)."""

    def __init__(self, params, levels=None):
        self.p = params
        levels = {} if levels is None else levels
        self.slices, start = [], 0
        for m in MODULE_NAMES:
            names = list(MODULES[m])
            self.slices.append((start, start + len(names), configs([levels.get(n, 2) for n in names])))
            start += len(names)
        self.Z = configs([2] * 4)

    def _config_weights(self, X, j):
        a, b, C = self.slices[j]
        w = np.ones((X.shape[0], C.shape[0]))
        for col, i in enumerate(range(a, b)):
            x = X[:, i][:, None]
            prior = self.p["pi"][i][C[:, col]][None, :]
            match = (C[None, :, col] == x).astype(float)
            w *= np.where(x < 0, prior, match)
        return w

    def bottleneck_marginals(self, X):
        out = []
        for j in range(len(MODULE_NAMES)):
            w = self._config_weights(X, j)
            hi = w @ self.p["theta"][j]
            out.append((np.stack([w.sum(1) - hi, hi], 1), w))
        return out

    def joint_z(self, X, y=None):
        marg = self.bottleneck_marginals(X)
        logp = np.zeros((X.shape[0], 16))
        for j, (m, _) in enumerate(marg):
            logp += np.log(np.clip(m[:, self.Z[:, j]], 1e-300, None))
        if y is not None:
            phi = self.p["phi"][None, :]
            logp += np.log(np.where(y[:, None] == 1, phi, 1 - phi))
        ll = np.logaddexp.reduce(logp, axis=1)
        return np.exp(logp - ll[:, None]), ll, marg

    def p_fail(self, X):
        q, _, _ = self.joint_z(X)
        return q @ self.p["phi"]

    def diagnose(self, X):
        q, _, _ = self.joint_z(X, y=np.ones(X.shape[0], dtype=int))
        return {b: q[:, self.Z[:, j] == 1].sum(1) for j, b in enumerate(BOTTLENECKS)}

    def bottlenecks(self, X):
        return {b: m[:, 1] / np.clip(m.sum(1), 1e-300, None) for b, (m, _) in zip(BOTTLENECKS, self.bottleneck_marginals(X))}


def em(X, y, levels=None, prior=None, prior_strength=PRIOR_STRENGTH, max_iter=300, tol=1e-6, verbose=False):
    """MAP-EM (prob_reas). `prior` = params of the prior profile (default: hand-set network); also the starting point."""
    levels = LEVELS if levels is None else levels
    prior = prior or handset_params(levels)
    params = {"pi": [p.copy() for p in prior["pi"]], "theta": [t.copy() for t in prior["theta"]], "phi": prior["phi"].copy()}
    a, last = prior_strength, -np.inf
    for it in range(max_iter):
        bn = FastBN(params, levels)
        q, ll, marg = bn.joint_z(X, y)
        loglik = ll.sum()
        n1, n = (q * y[:, None]).sum(0), q.sum(0)
        params["phi"] = (n1 + a * prior["phi"]) / (n + a)
        new_pi = [p.copy() for p in params["pi"]]
        for j, (start, end, C) in enumerate(bn.slices):
            m, w = marg[j]
            qj = np.stack([q[:, bn.Z[:, j] == b].sum(1) for b in (0, 1)], 1)
            th = params["theta"][j]
            r1 = w * th[None, :] / np.clip(m[:, 1:2], 1e-300, None) * qj[:, 1:2]
            r0 = w * (1 - th)[None, :] / np.clip(m[:, 0:1], 1e-300, None) * qj[:, 0:1]
            N1, N = r1.sum(0), (r1 + r0).sum(0)
            params["theta"][j] = (N1 + a * prior["theta"][j]) / (N + a)
            rc = r1 + r0
            for col, i in enumerate(range(start, end)):
                k = len(prior["pi"][i])
                counts = np.array([(rc * (C[None, :, col] == s)).sum() for s in range(k)])
                new_pi[i] = (counts + a * prior["pi"][i]) / (len(X) + a)
        params["pi"] = new_pi
        if verbose and it % 25 == 0:
            print(f"   EM iteration {it:3d}  log-likelihood {loglik:.2f}")
        if abs(loglik - last) < tol * max(1, abs(loglik)):
            break
        last = loglik
    return params


def choose_threshold(p, y, target_fail):
    best = 0.0
    for t in np.unique(p):
        served = p < t
        if served.any() and y[served].mean() <= target_fail:
            best = float(t)
    return best


def decide(p_fail, why, tau):
    q_side = np.maximum(why["QueryAmbiguity"], why["IntentUncertainty"])
    return np.where(p_fail < tau, "Serve", np.where(q_side >= 0.5, "Clarify", "Search"))


# ----------------------------------------------------------------- profiles
def to_profile(params, cuts, levels, meta):
    return {"meta": meta, "levels": {n: levels.get(n, 2) for n in INPUTS}, "cutoffs": cuts,
            "inputs": {n: params["pi"][i].round(5).tolist() for i, n in enumerate(INPUTS)},
            "bottlenecks": {m: {"parents": list(MODULES[m]), "p_high": params["theta"][j].round(5).tolist()}
                            for j, m in enumerate(MODULE_NAMES)},
            "risk": {"parents": BOTTLENECKS, "p_high": params["phi"].round(5).tolist()}}


def params_from_profile(profile):
    return {"pi": [np.array(profile["inputs"][n]) for n in INPUTS],
            "theta": [np.array(profile["bottlenecks"][m]["p_high"]) for m in MODULE_NAMES],
            "phi": np.array(profile["risk"]["p_high"])}


class Router:
    """A loaded profile, ready to route raw measurements."""

    def __init__(self, profile: dict):
        self.profile = profile
        self.levels = profile["levels"]
        self.bn = FastBN(params_from_profile(profile), self.levels)
        self.ignore = profile.get("meta", {}).get("ignore", [])
        self.tau = profile.get("meta", {}).get("serve_below", 0.2)

    def route(self, raw: dict) -> dict:
        ev = discretize(raw, self.profile["cutoffs"], self.ignore)
        X = np.array([[ev.get(n, -1) for n in INPUTS]])
        p = float(self.bn.p_fail(X)[0])
        why = {b: float(v[0]) for b, v in self.bn.diagnose(X).items()}
        route = str(decide(np.array([p]), {k: np.array([v]) for k, v in why.items()}, self.tau)[0])
        named = {n: state_names(self.levels.get(n, 2))[l] for n, l in ev.items()}
        dist = {"Serve": 0.5 * (1 - p), "Search": 0.4, "Clarify": 0.1 * (1 - p) + 0.6 * p}
        return {"p_hallucination_high": round(p, 4), "route": route, "serve_below": self.tau,
                "route_distribution": {k: round(v, 4) for k, v in dist.items()},
                "bottlenecks": {b: round(float(v[0]), 4) for b, v in self.bn.bottlenecks(X).items()},
                "blame": {k: round(v, 4) for k, v in why.items()}, "evidence": named}


def transfer_profile(prior_profile: dict, rows: list, embedder: str) -> dict:
    """A starting profile for this system from a prob_reas profile (same LLM, different embedder): see module docstring."""
    prof = json.loads(json.dumps(prior_profile))
    for node in EMBEDDER_DEPENDENT:
        key, direction = SIGNALS[node]
        vals = _vals(rows, key)
        vals = vals[~np.isnan(vals)]
        shares = np.array(prof["inputs"][node])                       # P(level) on the prior system
        if len(vals) < 20:
            continue
        cum = np.cumsum(shares)[:-1]                                    # level boundaries, Low -> High
        qs = cum if direction > 0 else 1 - cum                          # +1: trouble when large; -1: when small
        cuts = [float(np.quantile(vals, q)) for q in qs]
        prof["cutoffs"][node] = {"cuts": cuts, "learned": False, "info_gain": 0.0, "quantile_mapped": True}
    meta = prof.setdefault("meta", {})
    meta.update({"transferred_from": meta.get("llm_profile"), "embedder": embedder, "ignore": NOT_COMPARABLE,
                 "note": "prob_reas tables; embedder-dependent cut-offs quantile-mapped; relearn on IPC labels"})
    return prof
