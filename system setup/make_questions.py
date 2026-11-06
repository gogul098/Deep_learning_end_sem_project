"""make_questions.py · graded questions from Indian Supreme Court judgments, for learning the IPC router
(the IPC equivalent of the prob_reas 6,000-question set: PopQA-style facts, CoQA-style conversations, traps).

  python make_questions.py --aws-years 2023 2024 --n 300 --out ipc_qa      # downloads judgments + metadata (Kaggle)
  python make_questions.py --folder test-case-files --out ipc_qa_local      # local PDFs (no metadata questions)

Source: s3://indian-supreme-court-judgments (AWS Open Data, CC-BY-4.0, no account needed): per year an English
tar of PDFs (data/tar/year=YYYY/english/) and a metadata parquet (metadata/parquet/year=YYYY/metadata.parquet) with
petitioner, respondent, judge, author_judge, decision_date, disposal_nature. 2023+ is outside the classifier's
training years (train <= 2020, dev 2021-22).

Question types (gold answers never come from a language model):
  meta_*      parties, author judge, date, outcome - from the court's METADATA, independent of our extraction code
  fact_*      FIR date, police station, sentence, charges, compensation - read from the text by rules, kept only when
              the judgment states ONE value (an ambiguous gold answer would mislabel the router)
  trap        evidence the judgment never mentions (ballistic report, DNA, CCTV ...): correct = says it is not stated
  chain       3 questions in ONE conversation, later ones with pronouns ("When was THEIR case decided?")
  vague       "What about the other one?" with nothing to refer to: serving it is counted as a failure
Every fact question stores the sentence that contains the gold answer (to measure retrieval: was it found?).
Output: <out>/questions.jsonl (one line per question) and <out>/pdfs/*.pdf
"""
import argparse, io, json, os, random, re, sys, tarfile, urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ipc import case_facts as CF

BUCKET = os.environ.get("IPC_QA_BUCKET", "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
CRIMINAL = re.compile(r"\b(Indian Penal Code|I\.P\.C|IPC|accused|convicted|acquitted|Cr\.?\s?P\.?\s?C|charge-?sheet|"
                      r"Bharatiya Nyaya Sanhita|NDPS|POCSO|bail)\b", re.I)
TRAPS = [  # (words that must be ABSENT from the judgment, question)
    (r"ballistic", "What did the ballistic expert's report conclude about the weapon?"),
    (r"\bDNA\b", "What did the DNA report establish?"),
    (r"CCTV|video footage", "What did the CCTV footage show?"),
    (r"handwriting", "What did the handwriting expert conclude?"),
    (r"narco|polygraph|brain mapping", "What did the narco-analysis test reveal?"),
    (r"call detail|\bCDR", "What did the call detail records show?"),
    (r"post[- ]?mortem|autopsy", "What cause of death did the post-mortem report give?"),
    (r"test identification parade|\bTIP\b", "What happened at the test identification parade?"),
]
VAGUE = ["What about the other one?", "Why?", "Is it valid?", "And then?", "What about that?", "Tell me more about it."]


# ----------------------------------------------------------------------------------------------- gold-answer variants
def date_variants(d: str) -> list:
    """'2023-01-02' / '02-01-2023' / '02.01.2023' -> every way a judgment or an answer writes that date."""
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", d) or None
    if m:
        y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.match(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", d)
        if not m:
            return [d]
        da, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + 2000 if y < 100 else y
    if not (1 <= mo <= 12 and 1 <= da <= 31):
        return [d]
    mn = MONTHS[mo - 1]
    suf = "th" if 11 <= da <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(da % 10, "th")
    out = set()
    for dd in (f"{da:02d}", str(da)):
        for mm in (f"{mo:02d}", str(mo)):
            for sep in (".", "-", "/"):
                out.add(f"{dd}{sep}{mm}{sep}{y}")
    out |= {f"{da} {mn} {y}", f"{da}{suf} {mn} {y}", f"{mn} {da} {y}", f"{da}{suf} {mn}, {y}", f"{mn} {da}, {y}", f"{da:02d} {mn} {y}"}
    return sorted(out)


def party_variants(p: str) -> list:
    p = re.sub(r"\s+", " ", str(p or "")).strip()
    p = re.sub(r"\s*(?:&|and)\s*(?:ors?|anr|others|another)\.?\s*$|\s+etc\.?\s*$", "", p, flags=re.I).strip(" .,")
    if len(p) < 3:
        return []
    out = {p}
    words = [w for w in re.findall(r"[A-Za-z]+", p) if len(w) > 2]
    if len(words) >= 3 and not re.search(r"\b(state|union|government|commissioner|corporation|ltd|limited|bank|board)\b", p, re.I):
        out.add(" ".join(words[:2]))                                  # 'RAMESH KUMAR SINGH' -> 'Ramesh Kumar'
    return sorted(out)


def _author_in_text(text: str):
    """The 'judge' metadata column lists the bench (presiding judge first), not the author; read the author line instead."""
    from label_ipc import AUTHOR
    i = max(text[:20000].upper().find("JUDGMENT"), 0)
    m = AUTHOR.search(text[:20000], i)
    return (m.group(1) or m.group(2)).strip() if m else None


def judge_variants(j: str) -> list:
    j = re.sub(r"(?i)\bhon'?ble\b|\bmr\b\.?|\bmrs\b\.?|\bms\b\.?|\bdr\b\.?|\bjustice\b|\bj\.$|,\s*j\b\.?|\bcji\b|\bchief\b", " ", str(j or ""))
    j = re.sub(r"\s+", " ", j).strip(" .,")
    if len(j) < 4:
        return []
    names = [n.strip() for n in re.split(r",|;| and ", j) if len(n.strip()) > 3]
    out = set()
    for n in names[:1]:                                              # the author judge
        out.add(n)
        last = n.split()[-1]
        if len(last) >= 5:
            out.add(f"justice {last}")
    return sorted(out)


OUTCOME = [(r"partly allowed|partially allowed|allowed in part", ["partly allowed", "partially allowed", "allowed in part"]),
           (r"allowed", ["allowed", "set aside", "succeeds"]),
           (r"dismissed", ["dismissed", "rejected", "fails"]),
           (r"disposed", ["disposed"]),
           (r"remand|remitted", ["remanded", "remitted"])]


def outcome_variants(disposal: str) -> list:
    d = str(disposal or "").lower()
    for pat, gold in OUTCOME:
        if re.search(pat, d):
            return gold
    return []


# ----------------------------------------------------------------------------------------------- facts from the text
def sentences(text: str) -> list:
    flat = CF.flat(text)
    return [s for s in re.split(r"(?<=[.?!])\s+(?=[A-Z(\"“])", flat) if len(s) > 20]


def unique_fact(sents, pattern, group=1, norm=lambda x: x):
    """The value matched by `pattern` when the judgment states exactly ONE distinct value; and its sentence."""
    hits = [(norm(m.group(group)), s) for s in sents for m in re.finditer(pattern, s, re.I)]
    vals = {v for v, _ in hits if v}
    return (hits[0][0], hits[0][1]) if len(vals) == 1 else (None, None)


def text_facts(text: str) -> list:
    sents, out = sentences(text), []
    v, s = unique_fact(sents, r"\bF\.?\s?I\.?\s?R\.?\b[^.]{0,60}?\b(?:dated|dt\.?|on|registered on|lodged on)\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})")
    if v:
        out.append(("fact_fir_date", "On what date was the FIR registered?", date_variants(v), s, "match"))
    skip = {"the", "of", "and", "in", "dated", "under", "concerned", "said", "local", "nearest", "same", "with", "at", "to", "by"}
    ps_norm = lambda x: x if x and x.split()[-1].lower() not in skip and x.split()[0].lower() not in skip else None
    v, s = unique_fact(sents, r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+Police\s+Station\b", norm=ps_norm)   # 'Taloja Police Station'
    if not v:
        v, s = unique_fact(sents, r"\b(?:Police\s+Station|P\.\s?S\.|P/S)\s*[:-]?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b", norm=ps_norm)
    if v:
        out.append(("fact_police_station", "Which police station registered the case?", [v], s, "match"))
    v, s = unique_fact(sents, r"sentenced to (?:undergo\s+)?(?:rigorous |simple )?imprisonment for (life|\w+ \(?\d*\)?\s*years?)",
                       norm=lambda x: re.sub(r"\s+", " ", x.lower()).strip())
    if v:
        num = re.search(r"\d+", v)
        gold = ["life"] if v == "life" else [v] + ([f"{num.group(0)} years"] if num else [])
        out.append(("fact_sentence", "For what term of imprisonment was the accused sentenced?", gold, s, "match"))
    v, s = unique_fact(sents, r"compensation of (?:Rs\.?|₹|INR)\s*([\d,]+)", norm=lambda x: x.replace(",", ""))
    if v and len(v) >= 3:
        out.append(("fact_compensation", "What amount of compensation was awarded?", [f"{int(v):,}", v], s, "match"))
    ch = [(a, n) for a, n in CF.charges(text) if a != "unclear" and n]
    if ch:                                       # IPC/BNS by section number; other Acts by the full sub-section ('3(1)(q)')
        nums = [re.sub(r"\(.*$", "", x) if a in ("IPC", "BNS") else x for a, n in ch for x in n]
        sent = next((s for s in sents if re.search(r"offen[cs]es?\s+(?:punishable\s+)?under", s, re.I)), "")
        out.append(("fact_charges", "Under which sections were the accused charged?", sorted(set(nums)), sent, "sections"))
    return out


# ----------------------------------------------------------------------------------------------- one judgment -> questions
def questions_for(doc_id: str, text: str, meta: dict, rnd: random.Random) -> list:
    qs = []

    def add(qtype, q, answers, mode="match", evidence="", chain=None, step=0):
        if mode in ("match", "sections") and not answers:
            return
        qs.append({"id": f"{doc_id}:{qtype}" + (f":{step}" if chain else ""), "doc": doc_id, "type": qtype, "question": q,
                   "answers": answers, "grade": mode, "evidence": evidence[:400], "chain": chain, "step": step})

    head = text[:5000]
    role = ("appellant" if re.search(r"\bappellant", head, re.I) else "applicant" if re.search(r"\bapplicant", head, re.I)
            else "petitioner")
    pet, res_, judge, date, outc = (party_variants(meta.get("petitioner")), party_variants(meta.get("respondent")),
                                    judge_variants(meta.get("author_judge") or _author_in_text(text)),
                                    date_variants(str(meta.get("decision_date") or "")) if meta.get("decision_date") else [],
                                    outcome_variants(meta.get("disposal_nature")))
    add("meta_petitioner", f"Who is the {role} in this case?", pet)
    add("meta_respondent", "Who is the respondent in this case?", res_)
    add("meta_judge", "Which judge authored this judgment?", judge)
    add("meta_date", "On what date was this judgment delivered?", date)
    add("meta_outcome", "What was the final outcome of this case?", outc)
    for qtype, q, gold, sent, mode in text_facts(text):
        add(qtype, q, gold, mode, sent)
    absent = [(pat, q) for pat, q in TRAPS if not re.search(pat, text, re.I)]
    if absent:
        pat, q = rnd.choice(absent)
        add("trap", q, [], "abstain")
    if pet and (date or outc):                                       # a conversation with pronouns
        chain = f"{doc_id}:chain"
        add("chain", f"Who is the {role} in this case?", pet, chain=chain, step=1)
        if date:
            add("chain", "When was the judgment in their case delivered?", date, chain=chain, step=2)
        if outc:
            add("chain", "And what was the outcome?", outc, chain=chain, step=3)
    if rnd.random() < 0.35:
        add("vague", rnd.choice(VAGUE), [], "vague")
    return qs


# ----------------------------------------------------------------------------------------------- sources
def http_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def aws_metadata(year: int):
    import pandas as pd
    url = f"{BUCKET}/metadata/parquet/year={year}/metadata.parquet"
    with urllib.request.urlopen(url, timeout=120) as r:
        df = pd.read_parquet(io.BytesIO(r.read()))
    look = {}
    for row in df.to_dict("records"):
        for col in ("path", "cnr", "case_id", "nc_display"):
            v = row.get(col)
            if v is not None and str(v).strip() and str(v) != "nan":
                for k in _keys(str(v)):
                    look.setdefault(k, row)
    print(f"  metadata path examples: {list(df['path'].astype(str)[:3])}")
    return look


def _keys(name: str) -> list:
    """Ways a judgment can be named: tar member 'x/y/2023_1_12_34_EN.pdf' vs metadata path '2023_1_12_34' etc."""
    stem = os.path.splitext(os.path.basename(name.strip()))[0].lower()
    ks = {stem, re.sub(r"[^a-z0-9]", "", stem)}
    for suf in ("_en", "-en", "_english", "_eng"):
        if stem.endswith(suf):
            ks |= {stem[:-len(suf)], re.sub(r"[^a-z0-9]", "", stem[:-len(suf)])}
    return [k for k in ks if k]


def aws_tar_urls(year: int) -> list:
    base = f"{BUCKET}/data/tar/year={year}/english"
    try:
        idx = http_json(f"{base}/english.index.json")
        names = sorted({n for n in re.findall(r"[\w.-]+\.tar", json.dumps(idx))})
        if names:
            return [f"{base}/{n}" for n in names]
    except Exception:
        pass
    return [f"{base}/english.tar"]


def pdf_text(data: bytes) -> str:
    import pymupdf
    with pymupdf.open(stream=data, filetype="pdf") as d:
        return "\n".join(p.get_text() for p in d)


def from_aws(years, n, out_pdf, rnd, criminal_share=0.75):
    picked, want_crim = [], int(n * criminal_share)
    why = Counter()
    for year in years:
        meta = aws_metadata(year)
        print(f"year {year}: metadata for {len(meta)} name keys")
        for url in aws_tar_urls(year):
            print("  streaming", url)
            shown = 0
            with urllib.request.urlopen(url, timeout=600) as r, tarfile.open(fileobj=r, mode="r|*") as tar:
                for m in tar:
                    if shown < 3 and m.isfile():
                        print("    tar member example:", m.name); shown += 1
                    if not m.isfile() or not m.name.lower().endswith(".pdf"):
                        why["not a pdf"] += 1
                        continue
                    data = tar.extractfile(m).read()
                    if rnd.random() > 0.35:                            # sample across the year, not just its first files
                        why["sampled out"] += 1
                        continue
                    row = next((meta[k] for k in _keys(m.name) if k in meta), None)
                    if row is None:
                        why["no metadata match (kept, fact questions only)"] += 1
                        row = {}
                    try:
                        text = pdf_text(data)
                    except Exception as e:
                        why[f"pdf error {type(e).__name__}"] += 1
                        continue
                    words = len(text.split())
                    if not 800 <= words <= 25000:
                        why["too short/long (scanned?)" if words < 800 else "too long"] += 1
                        continue
                    crim = len(CRIMINAL.findall(text)) >= 5
                    n_crim = sum(p[3] for p in picked)
                    if (crim and n_crim >= want_crim) or (not crim and len(picked) - n_crim >= n - want_crim):
                        why["criminal/civil quota full"] += 1
                        continue
                    stem = os.path.splitext(os.path.basename(m.name))[0]
                    doc_id = re.sub(r"[^\w.-]", "_", stem if stem.startswith(str(year)) else f"{year}_{stem}")
                    open(os.path.join(out_pdf, f"{doc_id}.pdf"), "wb").write(data)
                    picked.append((doc_id, text, row, crim))
                    if len(picked) % 25 == 0:
                        print(f"    {len(picked)} judgments ({sum(p[3] for p in picked)} criminal)")
                    if len(picked) >= n:
                        print("  skipped:", dict(why))
                        return picked
            print("  skipped so far:", dict(why))
    return picked


def from_folder(folder, out_pdf):
    import hashlib, shutil
    picked, seen = [], set()
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".pdf"):
            continue
        data = open(os.path.join(folder, f), "rb").read()
        h = hashlib.md5(data).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        stem = os.path.splitext(f)[0][:24]
        shutil.copy(os.path.join(folder, f), os.path.join(out_pdf, f"{stem}.pdf"))
        picked.append((stem, pdf_text(data), {}, None))
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aws-years", nargs="+", type=int)
    ap.add_argument("--folder")
    ap.add_argument("--n", type=int, default=300, help="judgments")
    ap.add_argument("--out", default="ipc_qa")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    out_pdf = os.path.join(a.out, "pdfs")
    os.makedirs(out_pdf, exist_ok=True)
    if a.aws_years:
        picked = from_aws(a.aws_years, a.n, out_pdf, rnd)
    elif a.folder:
        picked = from_folder(a.folder, out_pdf)
    else:
        raise SystemExit("give --aws-years or --folder")
    allq = []
    for doc_id, text, meta, _ in picked:
        allq += questions_for(doc_id, text, meta, rnd)
    with open(os.path.join(a.out, "questions.jsonl"), "w", encoding="utf-8") as f:
        for q in allq:
            f.write(json.dumps(q, ensure_ascii=False, default=str) + "\n")
    c = Counter(q["type"] for q in allq)
    print(f"\n{len(picked)} judgments -> {len(allq)} questions -> {os.path.join(a.out, 'questions.jsonl')}")
    for k, v in sorted(c.items()):
        print(f"  {k:<22}{v}")


if __name__ == "__main__":
    main()
