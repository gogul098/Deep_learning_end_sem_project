"""retriever.py · Bayesian retriever: measures 14 pre-generation signals, asks the Bayesian network whether answering
now is safe, and routes the question to Serve / Search / Clarify.   CPU ONLY (never imports or uses CUDA).

  Serve    -> answer from the loaded judgment's own sentences
  Search   -> widen to the offline legal_kb (IPC statute cards + precedent passages) and the procedural KB
  Clarify  -> ask one targeted question, fold the reply into the query, route again (max 2 rounds)

Offline substitutions for the original MiniLM / cross-encoder / spaCy stack (the network itself is unchanged):
  embeddings  InLegalBERT mean-pooling (the same vectors legal_kb was built with; verified cosine 1.0), CPU copy for queries
  reranker    relevance = 0.5 * rescaled cosine + 0.5 * IDF-weighted query-term coverage
  spaCy       regex rules for entities / question type / pronouns (see the helpers below)
"""
from __future__ import annotations

import json, math, os, re, threading
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import bayes
from . import learned_bn
from .reranker import get_reranker

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Cut-offs on InLegalBERT cosine (measured on legal_kb: relevant statute queries top-1 cos 0.72-0.75, off-topic 0.60-0.67,
# nearest-neighbour passage pairs 0.90-0.97). tests/test_retriever.py prints the evidence for them.
CUTS = {"faiss": 0.34, "rerank": 0.5, "domain": 0.64, "views": 1000}
COS_LO, COS_HI = 0.60, 0.85          # cosine range rescaled to [0, 1] for the relevance score
RETRIEVE_K = 5
RERANK_POOL = int(os.environ.get("IPC_RERANK_POOL", "60"))   # candidates the cross-encoder re-orders (best RETRIEVE_K kept);
# eval_retrieval.py on 230 fact questions: answer sentence in the top 5 for 50% (12) / 57% (30) / 62% (60) / 62% (100);
# RRF fusion, 3-sentence windows and footer cleaning did not help
RERANK_RELEVANT = 0.5                # cross-encoder score that counts a chunk as relevant (prob_reas relevance_cutoff)
WIDE_K = 10                          # list questions ("all the conditions"): more sentences + their neighbours
CALIB_DIR = os.path.join(ROOT, "router_calibration")
EMBEDDER = "inlegalbert"
KIND_LIST = re.compile(r"\b(list|all the|all of the|every|each of|enumerate|conditions|directions|what (?:are|were) the (?:main |principal |various |key )?(?:grounds|reasons|charges|arguments|contentions|submissions))\b", re.I)
KIND_NUMBER = re.compile(r"\b(how (?:many|much|long|old)|on what date|what date|which year|when|age|amount|number of|period)\b", re.I)
KIND_LAW = re.compile(r"\b(under administrative law|distinction between|difference between|in general|generally|what is the (?:law|test|principle)|"
                      r"likely remedy|would a (?:high )?court|under writ jurisdiction|explain the)\b", re.I)
KIND_STRATEGY = re.compile(r"\b(could|should|strategy|argue|oppose|challenge|grounds? (?:to|could|for))\b", re.I)


def question_kind(q: str) -> str:
    """list | number | law | strategy | fact: what the question needs (used by the calibrated router and Serve-wide)."""
    if KIND_LIST.search(q):
        return "list"
    if KIND_LAW.search(q):
        return "law"
    if KIND_NUMBER.search(q):
        return "number"
    if KIND_STRATEGY.search(q):
        return "strategy"
    return "fact"


def model_slug(name: str) -> str:
    return "builtin-llama3.2-3b" if name == "builtin" else re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")
MAX_CLARIFY_ROUNDS = 2
PRECEDENT_WEIGHT = 0.7
CASE_SPECIFIC = re.compile(r"\b(he|she|him|her|his|they|them|applicant|accused|appellant|petitioner|respondent|prosecutrix|victim|"
                           r"informant|this case|the case|the judgment|this judgment|the court|the fir)\b", re.I)
AUTHORITY = {"statute": 1.0, "judgment": 0.9, "procedural": 0.9, "precedent": 0.85, "unknown": 0.5}

STOP = set("""a an the of in on at to for from by with and or but if then so as is are was were be been being do does did
have has had it its this that these those he she they them their his her him we you i me my our your what which who whom whose
when where why how whether can could should would will shall may might must not no yes any all some there here also than
about into over under after before between per via upon""".split())


# ───────────────────────────── text helpers ─────────────────────────────
def stem(w: str) -> str:
    for suf in ("ment", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def tokens(text: str) -> List[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1]


def count_words(text: str) -> int:
    return len(re.findall(r"[\w'-]+", text))


# ───────────────────────────── corpus: vectors + BM25 + coverage ─────────────────────────────
class Corpus:
    """A set of passages with (optional) unit vectors and a BM25 index."""

    def __init__(self, texts: List[str], vecs: Optional[np.ndarray] = None, k1: float = 1.5, b: float = 0.75):
        self.texts, self.k1, self.b = texts, k1, b
        self.vecs = None if vecs is None else np.asarray(vecs, dtype="float32")
        self.toks = [tokens(t) for t in texts]
        self.tf = [Counter(t) for t in self.toks]
        self.len = np.array([len(t) for t in self.toks], dtype="float32")
        self.avg = float(self.len.mean()) if len(texts) else 1.0
        df = Counter(w for t in self.tf for w in t)
        n = max(1, len(texts))
        self.df = df
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.max_idf = math.log(1 + (n + 0.5) / 0.5)

    def idf_of(self, w: str) -> float:
        return self.idf.get(w, self.max_idf)

    def bm25(self, q_terms: List[str]) -> np.ndarray:
        s = np.zeros(len(self.texts), dtype="float32")
        for w in set(q_terms):
            if w not in self.idf:
                continue
            idf = self.idf[w]
            for i, tf in enumerate(self.tf):
                f = tf.get(w)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s

    def coverage(self, q_terms: List[str], i: int) -> float:
        qs = set(q_terms)
        if not qs:
            return 0.0
        tot = sum(self.idf_of(w) for w in qs)
        return sum(self.idf_of(w) for w in qs if w in self.tf[i]) / tot if tot else 0.0

    def search(self, q_text: str, q_vec: Optional[np.ndarray], k: int) -> List[dict]:
        qt = tokens(q_text)
        cos = (self.vecs @ q_vec) if (self.vecs is not None and q_vec is not None) else np.zeros(len(self.texts), dtype="float32")
        if len(cos):                             # for the document-relative match signal (how much the best hit stands out)
            self.last_stats = (float(cos.mean()), float(cos.std()), float(cos.max()))
        bm = self.bm25(qt)
        bmn = bm / bm.max() if bm.max() > 0 else bm
        order = np.argsort(-(cos + 0.25 * bmn))[:k]
        out = []
        for i in order:
            c = float(cos[i])
            rel = 0.5 * min(max((c - COS_LO) / (COS_HI - COS_LO), 0.0), 1.0) + 0.5 * self.coverage(qt, int(i))
            out.append({"i": int(i), "cos": c, "bm25": float(bm[i]), "faiss_distance": 1.0 - c, "relevance": rel})
        return out


# ───────────────────────────── CPU query embedder ─────────────────────────────
class QueryEmbedder:
    """InLegalBERT mean-pooled, L2-normalised, always on the CPU. Loaded on first use."""

    def __init__(self, model_dir: str, threads: int = 4):
        self.model_dir, self.threads = model_dir, threads
        self._tok = self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                import torch
                from transformers import AutoModel, AutoTokenizer
                torch.set_num_threads(self.threads)
                self._tok = AutoTokenizer.from_pretrained(self.model_dir)
                self._model = AutoModel.from_pretrained(self.model_dir, dtype=torch.float32).eval()

    def __call__(self, texts: List[str]) -> np.ndarray:
        import torch
        self._load()
        out = []
        with self._lock:
            for i in range(0, len(texts), 16):
                x = self._tok(list(texts[i:i + 16]), padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    h = self._model(**x).last_hidden_state
                m = x["attention_mask"][..., None].float()
                v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
                out.append(v.numpy())
        return np.concatenate(out).astype("float32") if out else np.zeros((0, 768), dtype="float32")


# ───────────────────────────── knowledge base ─────────────────────────────
class KnowledgeBase:
    """legal_kb/: 100 IPC statute cards + precedent passages, with topic centroids for DomainShift."""

    def __init__(self, folder: str = os.path.join(ROOT, "legal_kb")):
        self.docs = [json.loads(l) for l in open(os.path.join(folder, "docs.jsonl"), encoding="utf-8")]
        emb = np.load(os.path.join(folder, "embeddings.npy")).astype("float32")
        self.centroids = np.load(os.path.join(folder, "centroids.npy")).astype("float32")
        pop = os.path.join(folder, "popularity.json")
        self.popularity = {k.lower(): int(v) for k, v in json.load(open(pop)).items()} if os.path.exists(pop) else {}
        self.corpus = Corpus([d["text"] for d in self.docs], emb)

    def domain_similarity(self, qv: np.ndarray) -> float:
        return float(np.max(self.centroids @ qv))

    def search(self, q_text: str, qv: np.ndarray, k: int) -> List[dict]:
        out = []
        for h in self.corpus.search(q_text, qv, k):
            d = self.docs[h["i"]]
            out.append({**h, "id": d["id"], "text": d["text"], "source_type": d.get("source_type", "unknown"),
                        "sections": d.get("section_labels", []), "where": "legal_kb"})
        return out


def procedural_docs() -> List[dict]:
    """Text cards for the procedural KB (CrPC / BNSS facts per IPC section), searchable like any other source."""
    try:
        from procedural_kb import IPC_PROCEDURAL_DATA
    except Exception:
        return []
    docs = []
    for sec, d in IPC_PROCEDURAL_DATA.items():
        text = (f"{sec} IPC, {d.get('title', '')}. Punishment: {d.get('punishment', '')}. "
                f"{'Cognizable' if d.get('cognizable') else 'Non-cognizable'}, {'bailable' if d.get('bailable') else 'non-bailable'}, "
                f"triable by {d.get('triable_by', '')}. Ingredients: {'; '.join(d.get('ingredients', []))}. "
                f"Bail: {d.get('bail_guidance', '')}")
        docs.append({"id": f"procedural-{sec.replace(' ', '_')}", "text": text, "source_type": "procedural",
                     "sections": [sec], "where": "procedural_kb"})
    return docs


# ───────────────────────────── signal heuristics (replace spaCy) ─────────────────────────────
FOLLOWUP_START = re.compile(r"^(and|also|what about|how about|but|so|then|ok|okay|same for)\b", re.I)
OPINION = re.compile(r"\b(best|better|worse|worst|should (i|we)|recommend\w*|prefer\w*|opinion|worth it|pros and cons|good idea)\b", re.I)
OPEN_START = re.compile(r"^(explain|describe|discuss|tell me about|compare|elaborate|overview|what do you think|summari[sz]e)\b", re.I)
HOW_FACTOID = re.compile(r"^how (many|much|old|long|far|often|big|tall|large)\b", re.I)
WH_INSIDE = re.compile(r"\b(and|also|as well as)\b.*\b(what|why|how|when|who|which|whether)\b", re.I)
TIME_WORDS = re.compile(r"\b(latest|current(ly)?|now|today|recent(ly)?|this (year|month|week)|newest|upcoming|as of|so far|nowadays|"
                        r"amend\w*|bns|bnss|bsa|bharatiya|new criminal laws?)\b", re.I)
VAGUE_REF = re.compile(r"\b(the other (?:one|case|thing|point|issue|section|judgment|order|person|party)|that one|this one|the same one|"
                       r"the former|the latter|the rest)\b", re.I)
GENERIC_WORDS = {"other", "one", "ones", "same", "rest", "former", "latter", "thing", "things", "more", "else", "another"}
ROLE_NOUNS = re.compile(r"\b(petitioners?|applicants?|appellants?|accused|respondents?|complainants?|informants?|victims?|witness(?:es)?|"
                        r"SPOs?|police officers?|deceased|prosecutrix)\b", re.I)
REFERRING = {"it", "its", "they", "them", "their", "this", "that", "these", "those", "he", "she", "him", "her", "his"}
PERSONAL = {"he", "she", "him", "her", "his"}
LEGAL_LEXICON = {"bail", "fir", "accused", "applicant", "appellant", "petitioner", "respondent", "prosecutrix", "victim", "witness",
                 "charge", "chargesheet", "conviction", "acquittal", "sentence", "evidence", "confession", "ipc", "crpc", "bnss", "pocso",
                 "section", "court", "judgment", "appeal", "trial", "arrest", "remand", "custody", "investigation", "statement",
                 "testimony", "injury", "weapon", "motive", "alibi", "dowry", "marriage", "consent", "age", "delay"}

LEGAL_STEMS = {stem(w) for w in LEGAL_LEXICON} | {"arrest", "convict", "acquit", "sentenc", "testif", "allege", "grant", "reject", "dismiss", "bail"}


def section_refs(text: str) -> List[str]:
    return [f"Section {m.upper()}" for m in re.findall(r"\b(?:section|sec\.?|s\.|u/s)\s*(\d+[A-Za-z]?)\b", text, re.I)]


def named_entities(text: str) -> List[str]:
    """Capitalised names (not the first word), section numbers, 'X v. Y' case names, quoted terms."""
    found = list(section_refs(text))
    words = re.findall(r"[A-Za-z][\w'.-]*", text)
    for w in words[1:]:
        if w[0].isupper() and w.lower() not in STOP and w.lower() not in {"i"} and len(w) > 1:
            found.append(w)
    found += re.findall(r"\"([^\"]{3,40})\"", text)
    seen, out = set(), []
    for f in found:
        if f.lower() not in seen:
            seen.add(f.lower()); out.append(f)
    return out


def classify_question(text: str, coordinated: bool) -> str:
    t = text.strip().lower()
    if coordinated: return "compound"
    if OPINION.search(t): return "opinion"
    if OPEN_START.search(t): return "open_ended"
    if t.startswith("why"): return "why"
    if t.startswith("how") and not HOW_FACTOID.search(t): return "how"
    return "factoid"


def referring_pronouns(text: str) -> List[str]:
    """Pronouns that point at something, ignoring determiner / complementizer uses of this/that."""
    ws = re.findall(r"[A-Za-z']+", text.lower())
    out = []
    for i, w in enumerate(ws):
        if w not in REFERRING:
            continue
        nxt = ws[i + 1] if i + 1 < len(ws) else ""
        if w in {"this", "that", "these", "those"} and nxt not in {"", "is", "was", "are", "were", "mean", "means", "meant"}:
            continue
        out.append(w)
    return out


def unresolved_pronouns(text: str, anchors: List[str], party_known: bool) -> List[str]:
    """Pronouns with no antecedent: no content word before them in the query and nothing in `anchors` (the previous
    turn's single entity). he/she/him/his/her resolve to the lead party when a judgment with a known party is loaded."""
    ws = re.findall(r"[A-Za-z']+", text.lower())
    out = []
    for p in referring_pronouns(text):
        if p in PERSONAL and party_known:
            continue
        before = [w for w in ws[:ws.index(p)] if w not in STOP and w not in {"did", "does", "do", "was", "were"}]
        if not before:
            out.append(p)
    return [] if len(anchors) == 1 else out


# ───────────────────────────── the retriever ─────────────────────────────
_CASE_INTRO = re.compile(r"\bcase\s+of\s+[A-Z]|\b[A-Z][\w.&\']*(?:\s+[\w.&\']+){0,8}\s+v(?:s)?\.?\s+[A-Z]|\b(?:held|observed|reads?)\s+(?:as\s+)?(?:under|thus|follows)", re.I)
_ORDINAL = re.compile(r"^\W*(?:\d+\.\s*)?(?:firstly|secondly|thirdly|fourthly|fifthly|sixthly|lastly|finally)\b", re.I)


_CUT_ABBREV = re.compile(r"(?:\bv|\bvs|\bors|\banr|\bno|\bnos|\bsecy|\bltd|\bco|\bpvt|\bmr|\bdr|\bsmt|\bshri|\bhon'?ble|\bart|\bsec|\bu/s)\.\s*\d*\W*$", re.I)


_ENUM = re.compile(r"(?:^|[\s;:(])\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|[a-h]|\d{1,2})\)\s*\S", re.I)
_INTRO_LIST = re.compile(r"(?:following|these|the)\s+(?:conditions|directions|terms)\s*:?-?|subject\s+to\s+(?:the\s+)?following", re.I)


_ASKS_CASES = re.compile(r"\b(?:judgments?|cases?|precedents?|decisions?|authorit\w+|rulings?|cited|relied|v(?:s)?\.)\b", re.I)


def _cut_after(s: str) -> bool:
    """'In the case of Hussainara Khatoon Vs.' | 'Home Secy., State of Bihar ...': cut at an abbreviation, continues."""
    return bool(_CUT_ABBREV.search((s or "").strip()))


def _starts_lower(s: str) -> bool:
    """'and the case of State of Haryana vs. Bhajan Lal ...': a fragment the splitter cut off at 'Ors.1'."""
    m = re.match(r"[\s\W\d]*([A-Za-z])", s or "")
    return bool(m and m.group(1).islower())


@dataclass
class DocIndex:
    sentences: List[str]
    corpus: Corpus
    party: Optional[str] = None
    names: set = field(default_factory=set)
    centroid: Optional[np.ndarray] = None

    @classmethod
    def build(cls, sentences: List[str], vecs: np.ndarray) -> "DocIndex":
        v = np.asarray(vecs, dtype="float32")
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        corpus = Corpus(sentences, v)
        head = " ".join(sentences[:12])
        m = re.search(r"([A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,4})\s*[.\s]*(?:\.\.)?\s*(?:Applicant|Appellant|Petitioner|Accused)\b", head)
        names = {w for w, d in corpus.df.items() if d / max(1, len(sentences)) <= 0.15}
        c = v.mean(0)
        return cls(sentences, corpus, m.group(1).strip() if m else None, names, c / max(float(np.linalg.norm(c)), 1e-9))


class BayesianRetriever:
    def __init__(self, kb_dir: str = os.path.join(ROOT, "legal_kb"), models_dir: str = os.path.join(ROOT, "models"),
                 embedder=None, threads: int = 4):
        self.kb = KnowledgeBase(kb_dir)
        self.embed = embedder or QueryEmbedder(os.path.join(models_dir, "inlegalbert"), threads)
        self.doc: Optional[DocIndex] = None
        self._proc: Optional[tuple] = None
        self.calibration: Optional[dict] = None
        self.calibration_file: Optional[str] = None
        self.learned: Optional[learned_bn.Router] = None
        self.reranker = get_reranker()

    def load_calibration(self, llm_name: str) -> Optional[str]:
        """router_calibration/<embedder>__<llm>.json (written by calibrate_router.py) for the chat model in use.
        None = no file for this model: the original hand-set network is used, and Clarify stays suppressed on clear case questions."""
        path = os.environ.get("IPC_ROUTER_PROFILE") or os.path.join(CALIB_DIR, f"{EMBEDDER}__{model_slug(llm_name)}.json")
        self.learned = None
        try:
            self.calibration = json.load(open(path, encoding="utf-8"))
            self.calibration_file = path
            if "bottlenecks" in self.calibration:            # learned network (learn_router.py, prob_reas format)
                self.learned = learned_bn.Router(self.calibration)
        except Exception:
            self.calibration, self.calibration_file = None, None
        return self.calibration_file

    # -- sources --------------------------------------------------------------------------------------------------
    def set_document(self, sentences: List[str], vecs: np.ndarray):
        self.doc = DocIndex.build(sentences, vecs)

    def clear_document(self):
        self.doc = None

    def _procedural(self):
        if self._proc is None:
            docs = procedural_docs()
            vecs = self.embed([d["text"] for d in docs]) if docs else np.zeros((0, 768), dtype="float32")
            self._proc = (docs, Corpus([d["text"] for d in docs], vecs) if docs else None)
        return self._proc

    def domain_similarity(self, qv: np.ndarray) -> float:
        """DomainShift: closeness to the knowledge base's topics, or to the loaded judgment itself (whichever is nearer)."""
        d = self.kb.domain_similarity(qv)
        return max(d, float(self.doc.centroid @ qv)) if self.doc is not None and self.doc.centroid is not None else d

    def rerank_hits(self, q: str, hits: List[dict]) -> List[dict]:
        """Cross-encoder scores ('rerank') and order; unchanged when the reranker is not installed."""
        if self.reranker is None or not hits:
            return hits
        scores = self.reranker.score(q, [h["text"] for h in hits])
        for h, s in zip(hits, scores):
            h["rerank"] = round(float(s), 5)
        return sorted(hits, key=lambda h: -h["rerank"])

    def search_doc(self, q: str, qv: np.ndarray, k: int) -> List[dict]:
        if self.doc is None:
            return []
        return [{**h, "id": f"S{h['i']}", "text": self.doc.sentences[h["i"]], "source_type": "judgment", "where": "judgment"}
                for h in self.doc.corpus.search(q, qv, k)]

    def search_doc_wide(self, q: str, qv: np.ndarray) -> List[dict]:
        """Serve-wide: the top WIDE_K sentences, each followed by what completes it, then the court's numbered reasons, then
        plain neighbours. Order matters: the prompt keeps only the first max_passages, so what a list question needs most
        comes first.
          * a hit inside an enumerated list "(i) ... (viii)" (bail conditions, directions) brings the whole run, page footers
            and footnotes in between included ("(ii) ... released on bail on" | "17 of 19 ..." | "Bond of Rs. 25,000/-"),
            up to 16 sentences, in document order;
          * a hit inside a quoted precedent brings the sentence that names the case (and its cut-off half);
          * one "Firstly / Secondly / Thirdly ..." sentence brings the others (max 8) - they need not share the question's words."""
        hits = self.search_doc(q, qv, WIDE_K)
        S = self.doc.sentences
        have = {h["i"] for h in hits}

        def item(j, rel):
            have.add(j)
            return {"i": j, "cos": 0.0, "bm25": 0.0, "faiss_distance": 1.0, "relevance": rel,
                    "id": f"S{j}", "text": S[j], "source_type": "judgment", "where": "judgment"}

        def block(h):                                         # the whole enumerated run around hit h
            i = h["i"]
            lo, hi, gap = i, i, 0
            while lo > 0 and i - lo < 15 and gap < 6:          # page footers / footnotes sit between items
                lo -= 1
                gap = 0 if _ENUM.search(S[lo]) or _INTRO_LIST.search(S[lo]) else gap + 1
            lo += gap
            gap = 0
            while hi + 1 < len(S) and hi - lo < 15 and gap < 6:
                hi += 1
                gap = 0 if _ENUM.search(S[hi]) else gap + 1
            hi -= gap
            return [item(j, h["relevance"] * 0.9) for j in range(lo, hi + 1) if j not in have]

        def intro(h):                                         # the sentence that names the quoted case
            i = h["i"]
            if _CASE_INTRO.search(S[i]):
                return []
            for j in range(i - 1, max(-1, i - 11), -1):
                if _CASE_INTRO.search(S[j]):
                    ks = ([j - 1] if j > 0 and (_cut_after(S[j - 1]) or _starts_lower(S[j])) else []) + [j]
                    return [item(k, h["relevance"] * 0.85) for k in ks if k not in have]
            return []

        # which completion comes right after each hit depends on what the question asks: cited cases -> the case names
        # first (a list of propositions must not crowd them out); otherwise -> whole enumerated lists first
        cases_first = bool(_ASKS_CASES.search(q))
        out, later = [], []
        for n, h in enumerate(hits):
            out.append(h)
            first, second = (intro, block) if cases_first else (block, intro)
            if n < 8 and (first is intro or _ENUM.search(S[h["i"]])):
                out += first(h)
            later.append(h)
        for h in later[:8]:
            if second is intro or _ENUM.search(S[h["i"]]):
                out += second(h)
        if any(_ORDINAL.search(S[h["i"]]) for h in out):
            near = [h["i"] for h in hits[:4]]
            ords = sorted((j for j in range(len(S)) if _ORDINAL.search(S[j]) and j not in have),
                          key=lambda j: min(abs(j - n) for n in near))[:8]
            out += [item(j, hits[0]["relevance"] * 0.8) for j in ords]
        for h in hits[:4]:
            for j in (h["i"] - 1, h["i"] + 1, h["i"] + 2):
                if 0 <= j < len(S) and j not in have:
                    out.append(item(j, h["relevance"] * 0.9))
        return out

    def with_neighbours(self, hits: List[dict], top: int = 3) -> List[dict]:
        """The sentence after (and before) each of the best `top` judgment hits: the content a question asks about is often
        in the next sentence ('Paragraph 102 reads thus:' -> the categories; 'In the case of X' -> what X held). Also:
          * a hit inside a quoted precedent gets the sentence that NAMES the case ('In the case of Hussainara Khatoon ...
            held as under:-'), up to 10 sentences back - the quote alone does not say whose words they are;
          * a hit cut by the sentence splitter ('... Pratima Mohanty and Ors.1' | 'and the case of ... Bhajan Lal') gets
            its continuation (next sentence starting in lower case) or its start (previous one, when it starts so)."""
        if self.doc is None:
            return hits
        S = self.doc.sentences
        have = {h["id"] for h in hits}
        extra = []

        def add(j, h, w=0.85):
            if 0 <= j < len(S) and f"S{j}" not in have:
                have.add(f"S{j}")
                extra.append({"i": j, "cos": 0.0, "bm25": 0.0, "faiss_distance": 1.0, "relevance": round(h["relevance"] * w, 4),
                              "id": f"S{j}", "text": S[j], "source_type": "judgment", "where": "judgment"})
                return True
            return False

        judg = [h for h in hits if h.get("where") == "judgment"]
        for h in judg[:top]:
            i = h["i"]
            follow = 4 if re.search(r"(?:thus|under|follows|following)\s*:?-?\s*\W*$|:\s*-?\s*\W{0,3}$", S[i].strip()) else 1
            for j in [i + k for k in range(1, follow + 1)] + [i - 1]:     # 'Paragraph 102 reads thus:' -> what follows
                add(j, h)
            if not _CASE_INTRO.search(S[i]):
                for j in range(i - 1, max(-1, i - 11), -1):                # whose words is this quote?
                    if _CASE_INTRO.search(S[j]):
                        add(j, h, 0.8)
                        break
        todo = judg[:top] + list(extra)
        while todo:                                                        # sentences the splitter cut in two (or three)
            h = todo.pop()
            i = h["i"]
            if i + 1 < len(S) and (_starts_lower(S[i + 1]) or _cut_after(S[i])) and add(i + 1, h, 0.8):
                todo.append(extra[-1])
            if i > 0 and (_starts_lower(S[i]) or _cut_after(S[i - 1])) and add(i - 1, h, 0.8):
                todo.append(extra[-1])
        return hits + extra

    def search_wide(self, q: str, qv: np.ndarray, k: int) -> List[dict]:
        """Judgment sentences + legal_kb + procedural cards. Precedent passages are other cases: they are dropped when the
        question is about THIS case (he / the applicant / this case ...) and otherwise down-weighted, so they never
        outrank the judgment's own sentences or a statute card on a fact question."""
        docs, corp = self._procedural()
        proc = []
        if corp is not None:
            proc = [{**h, "id": docs[h["i"]]["id"], "text": docs[h["i"]]["text"], "source_type": "procedural",
                     "sections": docs[h["i"]]["sections"], "where": "procedural_kb"} for h in corp.search(q, qv, 2)]
        kb = self.kb.search(q, qv, k)
        is_prec_query = bool(re.search(r"\b(precedent|precedents|similar case|similar cases|case law|case laws|ruling|rulings|judgment|judgments|cite|cited|citation|citations|authority|authorities|guidelines)\b|\b[A-Z][\w.]+\s+vs?\.\s+[A-Z]", q, re.I))   # a case named in the question ("X v. Y")
        if self.doc is not None and CASE_SPECIFIC.search(q) and not is_prec_query:
            kb = [h for h in kb if h["source_type"] != "precedent"]
        for h in kb:
            if h["source_type"] == "precedent":
                h["relevance"] *= (1.25 if is_prec_query else PRECEDENT_WEIGHT)
        cited = {x.lower() for x in section_refs(q)}
        for h in kb + proc:                      # "section 302" in the question + the card for Section 302 = an exact match
            if cited and h["source_type"] in ("statute", "procedural") and cited & {x.lower() for x in h.get("sections", [])}:
                h["relevance"] = max(h["relevance"], 0.75)
        pool = self.search_doc(q, qv, k) + kb + proc
        if self.reranker is not None:                          # cross-encoder order, same precedent down-weighting
            pool = self.rerank_hits(q, pool)
            for h in pool:
                if h.get("source_type") == "precedent":
                    h["rerank"] *= (1.25 if is_prec_query else PRECEDENT_WEIGHT)
            return sorted(pool, key=lambda h: -h["rerank"])[:k + 2]
        return sorted(pool, key=lambda h: -h["relevance"])[:k + 2]

    # -- signals --------------------------------------------------------------------------------------------------
    def measure(self, query: str, history: List[str], pool: Optional[List[dict]] = None, is_reply: bool = False,
                qv: Optional[np.ndarray] = None) -> dict:
        qv = qv if qv is not None else self.embed([query])[0]
        if pool is not None:
            cand = pool
            chunks = pool[:RETRIEVE_K]
        else:                                                  # retrieve wide, let the cross-encoder pick the best
            cand = self.search_doc(query, qv, RERANK_POOL if self.reranker else RETRIEVE_K) or \
                self.kb.search(query, qv, RERANK_POOL if self.reranker else RETRIEVE_K)
            chunks = self.rerank_hits(query, cand)[:RETRIEVE_K]
        ce = self.reranker is not None and all("rerank" in c for c in chunks) and bool(chunks)
        rel = [c for c in chunks if (c["rerank"] >= RERANK_RELEVANT if ce else c["relevance"] >= CUTS["rerank"])]
        auth_pool = rel or chunks[:3]
        names = named_entities(query)
        q_terms = tokens(query)
        anchors_doc = [w for w in set(q_terms) if self.doc is not None and w in self.doc.names and w in self.doc.corpus.df]
        legal = [w for w in re.findall(r"[a-z]+", query.lower()) if w in LEGAL_LEXICON or stem(w) in LEGAL_STEMS]
        entities = [e for e in dict.fromkeys(names + anchors_doc + legal) if e.lower() not in GENERIC_WORDS]
        coordinated = query.count("?") >= 2 or bool(WH_INSIDE.search(query))
        sub_q = max(1, query.count("?"), 2 if coordinated else 1)
        prev_ent = named_entities(history[-1]) if history else []
        if history and not prev_ent:                     # "Who are the petitioners?" -> "Why were THEY disengaged?"
            roles = list(dict.fromkeys(m.group(1).lower() for m in ROLE_NOUNS.finditer(history[-1])))
            prev_ent = roles[:1] if len(roles) == 1 else roles
        unresolved = unresolved_pronouns(query, prev_ent, self.doc is not None and self.doc.party is not None)
        if VAGUE_REF.search(query) and len(prev_ent) != 1:
            unresolved = unresolved + [VAGUE_REF.search(query).group(1).lower()]

        if history:
            is_follow = bool(FOLLOWUP_START.search(query.strip()) or referring_pronouns(query) or count_words(query) <= 4)
            ambiguous = is_follow and not entities and len(prev_ent) != 1
            sim_prev = 0.0 if is_reply else float(np.dot(qv, self.embed([history[-1]])[0]))
        else:
            ambiguous, sim_prev = False, None

        known_words = set(self.kb.popularity) | ({w for w in (self.doc.corpus.df if self.doc else {})})
        unknown_names = [n for n in names if n.lower() not in known_words and stem(n.lower()) not in known_words]
        views = (CUTS["views"] if not unknown_names else 0) if entities else None

        stats = getattr(self.doc.corpus, "last_stats", None) if (self.doc is not None and pool is None) else None
        top1_z = round((stats[2] - stats[0]) / stats[1], 3) if stats and stats[1] > 1e-6 else None
        raw = {"word_count": count_words(query), "question_type": classify_question(query, coordinated),
               "named_entity_count": len(entities), "sub_question_count": sub_q,
               "top1_faiss_distance": min((c["faiss_distance"] for c in cand), default=None),
               "top1_rerank_score": max((c["rerank"] if ce else c["relevance"] for c in chunks), default=None),
               "relevant_chunk_count": len(rel),
               "mean_source_authority": (float(np.mean([AUTHORITY.get(c["source_type"], .5) for c in auth_pool])) if auth_pool else None),
               "is_ambiguous_followup": ambiguous, "similarity_to_prev_query": sim_prev,
               "unresolved_reference_count": len(unresolved),
               "entity_monthly_views": views, "is_time_sensitive": bool(TIME_WORDS.search(query)),
               "kb_domain_similarity": self.domain_similarity(qv),
               # extra signals (enter the network only through a calibration file; always logged for calibration)
               "top1_cos": round(max((c["cos"] for c in chunks), default=0.0), 4) if chunks else None,
               "top1_cos_z": top1_z, "question_kind": question_kind(query),
               "reranker": "cross-encoder" if ce else "lexical"}
        details = {"entities": entities, "previous_turn_entities": prev_ent, "unresolved_pronouns": unresolved, "chunks": chunks}
        return {"raw": raw, "details": details, "qv": qv}

    def route(self, query: str, history: Optional[List[str]] = None, pool=None, is_reply: bool = False) -> dict:
        m = self.measure(query, history or [], pool, is_reply)
        if self.learned is not None:                          # learned network: P(fail), tau and the blame-based decision
            res = self.learned.route(m["raw"])
            ev = res.pop("evidence")
        else:
            cal = self.calibration or {}
            ev = bayes.extract_evidence(m["raw"], {**CUTS, **cal.get("cuts", {})})
            res = bayes.route_query(ev, cal or None)
        return {**res, "query": query, "raw": m["raw"], "evidence": ev, "details": m["details"], "qv": m["qv"]}


# ───────────────────────────── clarify loop (templates + rules, ported from clarify_loop.py) ─────────────────────────────
CLARIFY_PRIORITY = ["UnresolvedReferences", "FollowUpStatus", "QueryReformulation", "EntitySpecificity", "QueryLength",
                    "TopicComplexity", "QuestionType"]
LEAD_FILLER = re.compile(                                     # prob_reas clarify_loop.LEAD_FILLER
    r"^(?:(?:(?:yes|yeah|no|nope|oh|ah|sorry|well|ok|okay|um+|uh+|so|actually|basically)\s*)+[,!.]\s*"
    r"|(?:oh|ah|um+|uh+)\s+"
    r"|(?:mostly|mainly|probably|definitely|really|i think)\s+(?=\S)"
    r"|i\s*(?:'m|am|was)?\s*(?:meant|mean|meaning|talking about|referring to|asking about|thinking of)\s+"
    r"|(?:it|that|this)\s*(?:'s|is|was)\s+(?:about\s+)?"
    r"|(?:the one|the thing)\s+(?:about|with|for|on)\s+"
    r"|(?:talking|referring|asking)\s+about\s+|about\s+|regarding\s+)",
    re.I)
ONLY_FILLER = re.compile(r"^(?:(?:yes|yeah|yep|no|nope|oh|ah|sorry|well|ok|okay|sure|um+|uh+|hmm+)[\s,.!?]*)*$", re.I)


def clean_reply(reply: str) -> str:
    text = reply.strip()
    for _ in range(4):
        s = LEAD_FILLER.sub("", text, count=1).strip()
        if s == text:
            break
        text = s
    text = re.sub(r"[\s,]*(?:sorry|please|thanks|thank you|i guess|i think)?[\s.!?]*$", "", text, flags=re.I).strip()
    return "" if ONLY_FILLER.match(text) else text


def make_clarifying_question(evidence: dict, details: dict, party: Optional[str] = None) -> tuple:
    high = {k for k, v in evidence.items() if v == "High"}
    reason = next((r for r in CLARIFY_PRIORITY if r in high), "QueryLength")
    raw_opts = [o.strip() for o in details.get("previous_turn_entities", []) if o and o.strip()]
    # "Section 504" and "Section" -> keep "Section 504"; drop bare generic words and duplicates
    opts = [o for o in dict.fromkeys(raw_opts)
            if len(o) > 2 and o.lower() not in GENERIC_WORDS and o.lower() not in {"section", "sections", "act", "case", "court"}
            and not any(o != x and o.lower() in x.lower() for x in raw_opts)]
    pick = f": {' or '.join(opts[:3])}?" if len(opts) >= 2 else "?"
    if reason == "UnresolvedReferences":
        pron = (details.get("unresolved_pronouns") or ["it"])[0]
        if party and pron.lower() in {"he", "him", "his"}:
            q = f'When you say "{pron}", do you mean {party} or someone else?'
        else:
            q = f'When you say "{pron}", what are you referring to{pick}'
    elif reason == "FollowUpStatus":
        q = f"Which earlier topic is this about{pick}"
    elif reason == "QueryReformulation":
        q = "It looks like my last answer missed. What was wrong or missing?"
    elif reason in ("EntitySpecificity", "QueryLength"):
        q = "Could you be more specific: which section, party, witness, document or stage of the case do you mean?"
    elif reason == "TopicComplexity":
        q = "Your question has several parts. Which part should I answer first?"
    else:
        q = "What matters most here: the facts, the legal test, the evidence, or the outcome?"
    return q, reason


TRAIL_FILLER = re.compile(r"[\s,]*(?:sorry|please|thanks|thank you|i guess|i think)?[\s.!?]*$", re.I)
PLURAL_FIX = {"is": "are", "was": "were", "has": "have", "does": "do",
              "isn't": "aren't", "wasn't": "weren't", "doesn't": "don't", "hasn't": "haven't"}
SINGULAR_PRONOUNS = {"it", "this", "that", "its"}
POSSESSIVE = {"its", "their", "his"}


def answer_phrase(reply: str) -> tuple:
    """prob_reas answer_phrase without a parser: the cleaned reply when it is a short phrase (<= 6 words)."""
    text = clean_reply(reply)
    if not text:
        return None, False
    return (text, text.lower().endswith("s") and not text.lower().endswith("ss")) if len(text.split()) <= 6 else (None, False)


def _possessive(phrase: str) -> str:
    if phrase.endswith("'s") or phrase.endswith("'"):
        return phrase
    return phrase + ("'" if phrase.endswith("s") and not phrase.endswith("ss") else "'s")


def _is_possessive(original: str, pron: str) -> bool:
    if pron.lower() in POSSESSIVE:
        return True
    if pron.lower() != "her":
        return False
    m = re.search(r"\bher\s+(\w+)", original, re.I)
    return bool(m) and m.group(1).lower() not in {"to", "in", "on", "at", "for", "with", "about", "from", "and", "or"}


def _replace_pronoun(original: str, pron: str, phrase: str, plural: bool) -> Optional[str]:
    """Swap the pronoun (or vague phrase) for the user's phrase, fixing possessives and is/are agreement (prob_reas)."""
    if " " in pron:                                            # "the other one" -> the phrase
        new, n = re.subn(re.escape(pron), phrase, original, count=1, flags=re.I)
        return (new[0].upper() + new[1:]) if n else None
    possessive = _is_possessive(original, pron)
    words = original.split()
    for i, w in enumerate(words):
        core = re.sub(r"^\W+|\W+$", "", w)
        if core.lower() != pron.lower():
            continue
        lead = w[:w.lower().find(core.lower())]
        trail = w[len(lead) + len(core):]
        new = _possessive(phrase) if possessive else phrase
        if i == 0:
            new = new[0].upper() + new[1:]
        words[i] = lead + new + trail
        if plural and core.lower() in SINGULAR_PRONOUNS:
            for j in (i - 1, i + 1):
                if 0 <= j < len(words):
                    bare = re.sub(r"\W+$", "", words[j])
                    fix = PLURAL_FIX.get(bare.lower())
                    if fix:
                        fix = fix.capitalize() if bare[0].isupper() else fix
                        words[j] = fix + words[j][len(bare):]
        return " ".join(words)
    return None


def merge_clarification(original: str, reply: str, reason: str, details: dict) -> str:
    """prob_reas order: (1) unclear pronoun -> substitute the answer phrase; (2) otherwise append ALL the user's words
    (minus filler) - the retriever needs the detail; (3) nothing usable ('yes', 'sorry') -> keep the original (ask again).
    The merged text is for RETRIEVAL and ROUTING only; the answering model sees the real conversation."""
    if reason == "UnresolvedReferences" and details.get("unresolved_pronouns"):
        phrase, plural = answer_phrase(reply)
        if phrase:
            merged = _replace_pronoun(original, details["unresolved_pronouns"][0], phrase, plural)
            if merged:
                return merged
    extra = TRAIL_FILLER.sub("", clean_reply(reply)).strip()
    if extra:
        core = original.rstrip(" ?.!")
        return f"{core} - {extra}{'?' if original.rstrip().endswith('?') else ''}"
    return original


def resolve_from_history(query: str, history: List[str], party_known: bool, party_name: Optional[str] = None) -> tuple:
    """'When was he arrested?' after a turn that named an entity or where accused party is known."""
    if not history and not party_name:
        return query, False
    needs = [p for p in unresolved_pronouns(query, [], party_known) if p not in {"this", "that", "these", "those"}]
    if not needs:
        return query, False

    # Check last 3 history turns for named entities
    for prev_turn in reversed(history[-3:]):
        prev = named_entities(prev_turn)
        if len(prev) == 1:
            new, n = re.subn(rf"\b{needs[0]}\b", prev[0], query, count=1, flags=re.I)
            if n:
                return (new, True)

    # If pronoun is personal and accused party name is known from header
    if party_name and needs[0].lower() in {"he", "him", "his", "accused", "appellant"}:
        new, n = re.subn(rf"\b{needs[0]}\b", party_name, query, count=1, flags=re.I)
        if n:
            return (new, True)

    return query, False


_VAGUE_WORDS = re.compile(r"\b(?:other|then|more|else|again|same|about (?:it|that|this|them))\b", re.I)   # pronouns the router
# resolved from the conversation (unresolved_reference_count 0) are fine: "Why were they disengaged?"
_WH_START = re.compile(r"^\s*(?:who|what|when|where|which|whose|whom|how|why)\b", re.I)


_SELF_CONTAINED = re.compile(r"^\s*(?:please\s+)?(?:summari[sz]e|summary\s+of|explain|describe|list|give\s+(?:me\s+)?|outline|"
                             r"tell\s+me\s+about|what\s+(?:is|was|were|are))\b.*\b(?:judg\w*|orders?|case|facts|decision|verdict|"
                             r"outcome|holding|ratio|arguments?|submissions?|charges?|evidence|precedents?|issues?|"
                             r"points?|findings?|reasoning|conclusions?|legal\s+\w+)", re.I)   # judg\w*: also typos like "judgemen"


def is_specific(query: str, raw: dict) -> bool:
    """A question clear enough to answer as asked: asking the user to rephrase it would not help.
    'What about the other one?' / 'Tell me more about it.' / 'And then?' are not; 'Who filed the complaint?' is."""
    if _SELF_CONTAINED.search(query) and not _VAGUE_WORDS.search(query):
        return True                                    # "summarise the final judgement": about the whole case, not a follow-up
    if raw.get("is_ambiguous_followup"):
        return False
    wc, ents, refs = raw.get("word_count", 0), raw.get("named_entity_count", 0), raw.get("unresolved_reference_count", 0)
    if wc >= 12 and ents >= 3:                         # a long question naming several things: a pronoun in it resolves inside it
        return True
    if refs:
        return False
    return wc >= 8 or (wc >= 5 and ents >= 1) or (wc >= 4 and bool(_WH_START.match(query)) and not _VAGUE_WORDS.search(query))


@dataclass
class ChatRouter:
    """Per-conversation state: resolved user queries, the open clarification, and the routing decision for each turn."""
    retriever: BayesianRetriever
    history: List[str] = field(default_factory=list)
    pending: Optional[dict] = None

    def handle(self, message: str) -> dict:
        r = self.retriever
        # an answer to our clarifying question is a fragment ("Section 504", "the accused"); a full new question
        # ("Summarise the final judgement", "Who filed the FIR?") starts a new turn instead of being merged into the old one
        new_question = bool(_SELF_CONTAINED.search(message)) or (len(message.split()) >= 4 and bool(_WH_START.match(message)))
        is_reply = self.pending is not None and not new_question
        if is_reply:
            query = merge_clarification(self.pending["query"], message, self.pending["reason"], self.pending["details"])
            rounds = self.pending["rounds"]
        else:
            query, rounds = message, 0
            query, _ = resolve_from_history(query, self.history, r.doc is not None and r.doc.party is not None,
                                            r.doc.party if r.doc else None)

        d = r.route(query, self.history, is_reply=is_reply)
        caveat = False
        is_case_query = bool(re.search(
            r"\b(allegation|allegations|who|what|which|how|why|complainant|accused|victim|witness|"
            r"incident|event|events|facts|charge|charges|offence|offenses|bail\b|quash|convict|sentence|"
            r"order|decision|pw\s*\d+|eyewitness|fir\b|dispute|contention|argument|parties|case\s+about|"
            r"why\s+(?:was|is)?\s*section|what\s+charges|which\s+sections|charges\s+against|charges\s+framed|"
            r"why\s+\d+|why\s+was\s+\d+|what\s+offence|applicable\s+sections)\b",
            query, re.I
        ))
        # The network's P(fail) is reported as computed (it used to be forced to <= 0.08 here for any who/what/which question).
        # Without a calibration file the hand-set network over-asks, so Clarify is still skipped for a clear question about
        # the loaded judgment (no unresolved pronoun, not an ambiguous follow-up). With a calibration file the network decides.
        d["calibrated"] = bool(r.calibration)
        clear = is_specific(query, d["raw"]) or (not is_reply and is_specific(message, d["raw"]))   # "they" -> long party names
        if d["route"] == "Clarify" and clear and r.doc is not None:
            # Asking the user to rephrase only helps when the QUESTION is unclear. A specific question the network rates risky
            # (long, multi-part questions are rare in the generated training set, so they fall back to the prior) is
            # searched wider instead (calibrated), or answered (hand-set network, which over-asks).
            if r.calibration:
                d["route"], d["route_note"] = "Search", "high risk, but the question is specific: searched wider instead of asking"
            elif is_case_query:
                d["route"], d["route_note"] = "Serve", "Clarify skipped: specific question about this judgment (router not calibrated)"

        raw = d["raw"]
        if (d["route"] != "Clarify" and raw.get("unresolved_reference_count") and raw.get("word_count", 99) <= 6
                and not raw.get("named_entity_count") and rounds < MAX_CLARIFY_ROUNDS):
            # "What about the other one?" with nothing earlier to point to: no retrieval can answer this, so ask.
            d["route"], d["route_note"] = "Clarify", "short question with a reference that nothing in the conversation resolves"

        if d["route"] == "Clarify" and r.calibration and r.learned is None:
            # High P(fail) only warrants a question to the user when the QUESTION is the problem. When the blame is on the
            # retrieved context or a knowledge gap, rephrasing would not help: widen the search instead.
            bl = d.get("blame", {})
            if bl.get("QueryAmbiguity", 0) + bl.get("IntentUncertainty", 0) < bl.get("ContextDeficit", 0) + bl.get("KnowledgeGap", 0):
                d["route"], d["route_note"] = "Search", "high risk blamed on context, not on the question: searched wider instead of asking"

        if d["route"] == "Clarify" and rounds >= MAX_CLARIFY_ROUNDS:
            d["route"], caveat = "Search", True

        if os.environ.get("IPC_FORCE_SERVE") == "1":
            # labelling (label_ipc.py): record what the router WOULD do, but always answer, so the label says whether
            # serving now fails - the outcome the network learns (prob_reas labelling)
            d["route_router"], d["route"] = d["route"], "Serve"

        if d["route"] == "Clarify":
            question, reason = make_clarifying_question(d["evidence"], d["details"], r.doc.party if r.doc else None)
            self.pending = {"query": query, "reason": reason, "details": d["details"], "rounds": rounds + 1}
            return {**d, "ask_user": question, "reason": reason, "passages": [], "caveat": False, "resolved_query": query}

        self.pending = None
        qv = d["qv"]
        d["max_passages"] = 8
        if d["route"] == "Search":
            passages = r.search_wide(query, qv, RETRIEVE_K)
        else:
            passages = d["details"]["chunks"]
        if r.doc is not None:
            passages, d["max_passages"] = r.with_neighbours(passages), 13
        if d["raw"].get("question_kind") == "list" and r.doc is not None:
            # the list itself first (wide hits, whole enumerated blocks, the court's numbered reasons), then the neighbours of
            # the best hits: in the other order the neighbours filled the passage budget and cut the list short
            wide = r.search_doc_wide(query, qv)
            seen = {p["id"] for p in wide}
            passages = wide + [p for p in passages if p["id"] not in seen]
            d["route_action"], d["max_passages"] = "Serve-wide", 20
        self.history.append(query)
        return {**d, "passages": passages, "caveat": caveat, "resolved_query": query}
