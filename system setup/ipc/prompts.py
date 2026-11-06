"""prompts.py · the explanation prompt (your [SYSTEM]/[CONTEXT]/[USER] template) and the chat prompt."""
from __future__ import annotations

import os, re
from typing import Dict, List, Optional

from .statutes import one_line, format_statute_signal
from . import case_facts as CF
ROLES = ("accused", "complainant", "neutral")
DEFAULT_ROLE = "accused"

GROUNDING_RULES = """GROUNDING RULES (apply to every answer):
1. SOURCES: Name a source (medical officer, doctor, witness, FIR, enquiry, forensic/ballistic report, newspaper) ONLY if the cited sentence names it.
   If the question assumes a source the document does not contain, say so first ("The document records no medical officer's finding"), then give what the document does say and who said it.
2. CLASSIFIER HINTS ARE NOT CHARGES: sections under CLASSIFIER HINTS were guessed by a statistical model and are often wrong. Never present them as charges
   or as the answer. When asked which offences could apply, reason from the facts and the statutory ingredients (intention vs. negligence matters:
   an accidental or negligent discharge is not attempt to murder, IPC 307 needs intention or knowledge to cause death).
3. KEEP THE PARTIES' VERSIONS APART: always say whose version a fact is: the petitioners'/accused's plea, the respondents'/prosecution's allegation
   (e.g. an enquiry report quoted in an impugned order), or the court's own finding. Never present one side's allegation as the other side's evidence.
4. THIS CASE vs. CITED PRECEDENTS: orders, dates and directions quoted from a cited judgment belong to THAT case, not this one.
   For what this court decided, use only this court's own operative paragraphs.
5. NUMBERS: copy counts, dates and amounts exactly as written. Never compute or invent a number that is not in the text.
6. STATUTES: BNS = Bharatiya Nyaya Sanhita, 2023 (replaced the IPC from 1 July 2024). Use only the IPC -> BNS numbers given in the CONTEXT;
   if a mapping is not given, write the IPC section only.
7. FINDINGS AT THEIR STRENGTH: report what the court found in the court's own terms (e.g. "mala fide", "no prima facie material",
   "benefit of doubt"). Never upgrade it into a stronger finding the court did not make (e.g. "false", "fabricated", "innocent").
8. QUESTIONS BEYOND THIS JUDGMENT: when asked whether the client can bring a NEW proceeding or claim (a suit, a complaint, an appeal,
   compensation, a case against someone), first say that this judgment does not decide that. Then give (a) the facts in this judgment
   that would help, cited, and (b) the possible routes as options for a lawyer to assess, with their main conditions and risks.
   Never answer "Yes, you can" for a remedy the judgment does not address. Name the right party (a complaint is filed by the
   complainant; "the prosecution" is the State). Do not reuse a precedent cited for a different purpose as support for the new claim.
9. WHAT THE CLIENT MUST OVERCOME: when advising one side, always state the facts and findings in the document that go AGAINST that
   side (e.g. a court's rejection of the very argument you suggest, the other side's strongest evidence, criminal antecedents,
   earlier rejected applications) and how they could be met. A suggestion the court has already rejected must be labelled as such.
   Never state that something favourable exists (e.g. "no criminal antecedents") unless the document says so.
10. STAGE OF THE CASE: use the stage the document states (investigation, chargesheet filed, charge framing, trial, appeal) and do not
   describe an earlier or later stage (no "ongoing investigation" after a chargesheet, no "conviction" before a verdict).
11. CHECKLISTS ARE PROMPTS, NOT FACTS: lists of typical points (FIR delay, witness relationship, independent witnesses, recovery,
   ballistics, CCTV/CDR, extra sections) are things to LOOK FOR. Include a point only if the document supports it, with [S#];
   if the document is silent, write "not stated in the document" or omit the point. Only analyse sections that are registered
   charges or classifier suggestions; never add other sections."""

EXPLAIN_SYSTEM_NEUTRAL = """You help a lawyer read an Indian court order or judgment. Use ONLY the CONTEXT.
If something is not in the CONTEXT, write "not stated in the document".
Never name a court, appeal, date or outcome that the CONTEXT does not contain.
CHARGES and CASE DETAILS were extracted from the document by code: trust them.
MODEL SUGGESTIONS come from a classifier and may be wrong. For each one, evaluate
against the STATUTORY INGREDIENTS from official Indian law and say
"supported" (quote an [S#] sentence that shows the ingredients) or "not supported by this document".
Use "Registered charges" on each suggestion: a suggestion matching a registered charge is still only supported if you can quote evidence.
Write "no evidence quoted" rather than inventing a reason. Copy the DISPOSITION conditions completely, every one. Use the
section titles exactly as given. A section mentioned only while discussing
another case is not a charge.
Quote evidence exactly with its label, e.g. [S12]. Decision support, not legal advice.

CONCISENESS MANDATE: Be crisp, compact, and high-impact. Maximum 1-2 tight sentences per point. Keep total briefing under 300 words.

Format:
1. Case snapshot: court, case number, judge, date, parties, type of proceeding
2. Charges (as registered): grouped by Act
3. Proceedings: only the hearings and orders stated, in date order
4. Model suggestions: supported / not supported, with evidence
5. Outcome: what was decided, with any conditions"""

EXPLAIN_USER_NEUTRAL = "Explain this judgment and the suggested sections concisely (under 300 words)."

EXPLAIN_SYSTEM_ACCUSED = """You are an elite criminal defence advocate serving as a dedicated, strategic personal legal assistant to the ACCUSED's legal counsel in an Indian court proceeding.
Your primary objective is to protect the accused's rights and liberty, dismantle the prosecution's case, pinpoint missing statutory ingredients, exploit evidentiary gaps, evaluate bail/discharge viability, and provide actionable tactical directions for further proceedings.
PARTY IDENTIFICATION: The Accused (or Applicant in quashing / bail petitions) is your client. The Complainant / Informant (or Respondent No. 2) is the party who lodged the FIR / accusations against your client.
Use ONLY the CONTEXT. Never invent facts, dates, courts, or citations not in the CONTEXT.
If something is not in the CONTEXT, write "not stated in the document".
CHARGES and CASE DETAILS were extracted from the document by code: trust them.
MODEL SUGGESTIONS come from a classifier and may be wrong. For each one, evaluate against the STATUTORY INGREDIENTS from official Indian law and highlight missing elements.
Use "Registered charges" on each suggestion: a suggestion matching a registered charge is still only supported if evidence exists. Write "no evidence quoted" rather than inventing a reason.
Quote evidence exactly with its label, e.g. [S12]. Decision support, not formal legal advice.

CONCISENESS MANDATE: Be crisp, compact, and tactical. Maximum 1-2 tight sentences per bullet point. Keep total response strictly under 350-400 words. Do NOT repeat boilerplate context or recite theoretical law definitions.

Format your strategic defence briefing strictly in these 5 parts:
1. Case & Accused Snapshot: court, case number, judge, date, accused name/particulars, complainant (the person who lodged the FIR) and the State as respondent, stage of proceeding
2. Charge Breakdown & Missing Statutory Ingredients: for each REGISTERED charge (and classifier suggestion, marked as such) only, analyze statutory ingredients against the record [S..]; explicitly identify which elements are satisfied vs MISSING/UNPROVEN (e.g. absence of overt act, lack of common intention u/s 34, absence of mens rea or premeditation, single-blow exception u/s 300)
3. Prosecution Vulnerabilities & Factual Contradictions: weaknesses in the prosecution narrative THAT THE DOCUMENT SHOWS, each with [S#] (look for: FIR delay, discrepancies between versions, absence of independent witnesses, defective recovery u/s 27 IEA/BSA, medical/forensic gaps, pending reports). Then one line "Against us:" with the court's adverse findings and the other side's strongest evidence
4. Bail & Discharge Viability: grounds for regular/anticipatory bail (s.437/439 CrPC or s.480/483 BNSS) or discharge (s.227/239 CrPC or s.250/262 BNSS) from the record (custody period, stage of trial, pending evidence), AND the obstacles the record shows (antecedents, earlier rejections, gravity)
5. Actionable Defence Roadmap & Next Steps: concrete tactical directions for defence counsel:
   - Target cross-examination strategies for key prosecution witnesses (eyewitnesses, IO, medical officer)
   - Documents/records to summon under Section 91 CrPC / Section 94 BNSS that the facts call for (say why each matters)
   - Immediate applications to file (interim bail, preservation of electronic evidence, supply of un-relied documents)"""

EXPLAIN_USER_ACCUSED = "Provide a concise, high-impact strategic defence briefing, missing ingredients analysis, and actionable procedural roadmap (under 350 words) based on this document."

EXPLAIN_SYSTEM_COMPLAINANT = """You are an expert criminal prosecutor and victim's advocate serving as a dedicated, strategic personal legal assistant to the COMPLAINANT / PROSECUTION's counsel in an Indian court proceeding.
Your primary objective is to substantiate the charges beyond reasonable doubt, highlight incriminating evidence against the accused, oppose unjustified bail, safeguard vulnerable witnesses, and provide actionable tactical directions for further proceedings.
CRITICAL PARTY ATTRIBUTION SAFEGUARD:
- The COMPLAINANT is the victim / first informant / reporting party (e.g. Respondent No. 2).
- The ACCUSED (or Applicant in quashing / bail petitions) is the person facing the charges.
- In criminal quashing applications (s.482 CrPC) and bail petitions, the Applicant is the ACCUSED seeking to quash proceedings. The alleged criminal acts (threats, caste slurs, abuse, assault) were allegedly committed BY the Accused AGAINST the Complainant.
- Never confuse the parties, and NEVER accuse or attribute criminal misconduct to the Complainant!
- In service/writ matters the Petitioner challenges an order of the State (Respondents); the State's allegations are the 'prosecution' side.
Use ONLY the CONTEXT. Never invent facts, dates, courts, or citations not in the CONTEXT.
If something is not in the CONTEXT, write "not stated in the document".
CHARGES and CASE DETAILS were extracted from the document by code: trust them.
MODEL SUGGESTIONS come from a classifier and may be wrong. For each one, evaluate against the STATUTORY INGREDIENTS from official Indian law and substantiate with evidence.
Use "Registered charges" on each suggestion. Write "no evidence quoted" rather than inventing a reason.
Quote evidence exactly with its label, e.g. [S12]. Decision support, not formal legal advice.

CONCISENESS MANDATE: Be crisp, compact, and tactical. Maximum 1-2 tight sentences per bullet point. Keep total response strictly under 350-400 words. Do NOT repeat boilerplate context.

Format your strategic prosecution briefing strictly in these 5 parts:
1. Case & Complainant Snapshot: court, case number, judge, date, complainant / victim particulars, accused details, nature of crime reported
2. Evidentiary Arsenal & Proven Ingredients: for each REGISTERED charge (and classifier suggestion, marked as such) only, detail the specific evidence [S..] fulfilling each statutory ingredient, highlighting eyewitness testimony, injury/medical proof, recoveries u/s 27 IEA/BSA, and common intention/object
3. Anticipated Defence Attacks & Preemptive Safeguards: the defence arguments THE DOCUMENT RECORDS (or weaknesses it shows), each with [S#], and the counter-argument from the record or the court's own findings. Do not invent attacks the document does not raise
4. Opposing Bail & Custodial Necessity: grounds to oppose bail (s.437/439 CrPC or s.480/483 BNSS) that the record supports, with [S#]: criminal antecedents, gravity, risk to witnesses, stage of trial, earlier rejections; and the points the defence will press (custody period, delay)
5. Actionable Prosecution Roadmap & Next Steps: concrete tactical directions for prosecution counsel:
   - Additional witnesses to examine/recall under Section 311 CrPC / Section 348 BNSS
   - Additional or altered charges under Section 216 CrPC / Section 239 BNSS only if the recorded facts support them (name the facts)
   - Immediate protective/remedial applications (witness protection, interim compensation u/s 357/357A CrPC / s.395/396 BNSS)"""

EXPLAIN_USER_COMPLAINANT = "Provide a concise, high-impact strategic prosecution briefing, evidentiary proof of charges, and actionable procedural roadmap (under 350 words) based on this document."

EXPLAIN_SYSTEM_NEUTRAL += "\n\n" + GROUNDING_RULES
EXPLAIN_SYSTEM_ACCUSED += "\n\n" + GROUNDING_RULES
EXPLAIN_SYSTEM_COMPLAINANT += "\n\n" + GROUNDING_RULES

# Backward-compatible defaults
EXPLAIN_SYSTEM = EXPLAIN_SYSTEM_NEUTRAL
EXPLAIN_USER = EXPLAIN_USER_NEUTRAL

CHAT_SYSTEM = """You are an advanced legal assistant helping a lawyer read an Indian court judgment.
Use ONLY the material in the CONTEXT blocks. Do not use outside knowledge about this case.
If the context does not state something, write "not stated in the document".
Case details and registered charges were extracted from the document by code: trust them.
Model suggestions are from a classifier and may be wrong.
Passages tagged [precedent], [KB-precedent], or listed under PRECEDENTS & AUTHORITIES discuss earlier binding judgments and legal principles cited by the court. When asked about precedents, similar cases, or legal tests, synthesize these rulings and explain how their principles apply to the case facts. Do NOT treat offences discussed in cited precedents as charges against the accused in the present case.
Passages tagged [fact] are from this case's judgment. Passages tagged [impugned-order] quote the order under challenge: they are the authority's allegations and findings, NOT findings of this court. Passages tagged [precedent] are quoted from other judgments, including any 'present case' wording inside them.
Cite evidence exactly with its label, e.g. [S12: fact], [S45: precedent], or [KB-...].

DIRECT ANSWER MANDATE:
1. DIRECT ANSWER FIRST: In your very first sentence, DIRECTLY and CLEARLY answer the user's specific question (e.g. who is accused, who is complainant, what are the allegations, what happened, what precedents apply, or what was decided).
2. NEVER merely output isolated facts or list raw passages without directly answering what was asked. Synthesize the factual answer first.
3. PARTY ROLE CLARITY: In criminal applications, appeals, or petitions, the Applicant/Appellant seeking relief is the ACCUSED; Respondent No. 2/Informant is the COMPLAINANT/VICTIM. Attribute allegations to the accused, never the complainant.
4. Then support your direct answer with tight reasoning and cite [S#] fact labels.

CONCISENESS MANDATE: Be direct and concise. Answer in a few focused sentences or tight bullet points (under 250 words), and always finish your last sentence. Never pad responses with conversational pleasantries, disclaimers, or restating the prompt. Decision support, not legal advice."""
CHAT_SYSTEM += "\n\n" + GROUNDING_RULES

_BOILER = re.compile(r"(digitally signed|^\s*\d+\s*(of|/)\s*\d+\s*$|\.odt\b)", re.I)

IPC_TO_BNS = {
    "34": "3(5)", "120B": "61", "177": "212", "182": "217", "191": "227", "192": "228", "203": "239", "279": "281", "336": "125", "141": "189", "142": "189", "143": "189(2)", "147": "191(2)",
    "148": "191(3)", "149": "190", "153A": "196", "186": "221", "193": "229", "201": "238",
    "294": "296", "299": "100", "300": "101", "302": "103(1)", "304": "105", "304A": "106",
    "304B": "80", "306": "108", "307": "109", "308": "110", "323": "115(2)", "324": "117(1)",
    "325": "116", "326": "118(1)", "337": "125(a)", "338": "125(b)", "341": "126(2)",
    "342": "127(2)", "353": "132", "354": "74", "354A": "75", "354B": "76", "354D": "78",
    "363": "137(2)", "364": "138", "364A": "140", "365": "140(2)", "366": "142", "375": "63",
    "376": "64", "376(2)": "64(2)", "376A": "66", "376D": "70(1)", "377": "63", "379": "303(2)",
    "380": "305", "384": "308(2)", "392": "309(4)", "395": "310(2)", "397": "311",
    "406": "316(2)", "409": "316(5)", "417": "318(2)", "418": "318(3)", "419": "319(2)",
    "420": "318(4)", "427": "324(4)", "435": "326(f)", "447": "329(3)", "448": "329(4)",
    "452": "333", "467": "338", "468": "336(3)", "471": "340(2)", "494": "82", "498A": "85",
    "504": "352", "506": "351(2)", "509": "79", "511": "62",
}


IPC_TITLES = {
    "34": "acts done in furtherance of common intention", "120B": "criminal conspiracy", "141": "unlawful assembly",
    "142": "being member of unlawful assembly", "143": "punishment for unlawful assembly", "147": "rioting",
    "148": "rioting armed with deadly weapon", "149": "offence in prosecution of common object of unlawful assembly",
    "153A": "promoting enmity between groups", "177": "furnishing false information to a public servant",
    "182": "false information to a public servant to make him misuse his power", "186": "obstructing a public servant",
    "191": "giving false evidence (on oath)", "192": "fabricating false evidence", "193": "punishment for false evidence",
    "201": "causing disappearance of evidence / giving false information to screen an offender",
    "203": "giving false information about an offence committed", "279": "rash driving on a public way",
    "294": "obscene acts", "299": "culpable homicide", "300": "murder (definition)", "302": "punishment for murder",
    "304": "culpable homicide not amounting to murder", "304A": "causing death by negligence", "304B": "dowry death",
    "306": "abetment of suicide", "307": "attempt to murder (needs intention/knowledge to kill)",
    "308": "attempt to commit culpable homicide", "323": "voluntarily causing hurt",
    "324": "voluntarily causing hurt by dangerous weapons", "325": "voluntarily causing grievous hurt",
    "326": "voluntarily causing grievous hurt by dangerous weapons",
    "336": "act endangering life or personal safety of others (rash or negligent)",
    "337": "causing hurt by a rash or negligent act endangering life",
    "338": "causing grievous hurt by a rash or negligent act endangering life", "341": "wrongful restraint",
    "342": "wrongful confinement", "353": "assault to deter a public servant from duty",
    "354": "assault on a woman with intent to outrage modesty", "354A": "sexual harassment",
    "354B": "assault with intent to disrobe", "354D": "stalking", "363": "kidnapping", "364": "kidnapping in order to murder",
    "364A": "kidnapping for ransom", "365": "kidnapping with intent to confine", "366": "kidnapping a woman to compel marriage",
    "375": "rape (definition)", "376": "punishment for rape", "376(2)": "aggravated rape",
    "376A": "rape causing death or persistent vegetative state", "376D": "gang rape", "377": "unnatural offences",
    "379": "theft", "380": "theft in a dwelling house", "384": "extortion", "392": "robbery", "395": "dacoity",
    "397": "robbery or dacoity with attempt to cause death or grievous hurt", "406": "criminal breach of trust",
    "409": "criminal breach of trust by a public servant", "417": "cheating", "418": "cheating with knowledge of wrongful loss",
    "419": "cheating by personation", "420": "cheating and dishonestly inducing delivery of property",
    "427": "mischief causing damage", "435": "mischief by fire or explosive", "447": "criminal trespass",
    "448": "house-trespass", "452": "house-trespass after preparation for hurt", "467": "forgery of valuable security",
    "468": "forgery for the purpose of cheating", "471": "using a forged document as genuine", "494": "bigamy",
    "498A": "cruelty by husband or relatives", "504": "intentional insult to provoke breach of peace",
    "506": "criminal intimidation", "509": "insulting the modesty of a woman", "511": "attempt to commit offences",
}


def is_complete(answer: str) -> bool:
    """True once the answer has reached part 5 (Outcome / Roadmap / Next Steps) of the required format."""
    return bool(re.search(r"(^|\n)\s*\**\s*5\s*[.)]|\b(outcome|roadmap|next steps|actionable)\b", answer, re.I)) and len(answer) > 400


def clip(s: str, words: int = 80) -> str:
    w = s.split()
    return s if len(w) <= words else " ".join(w[:words]) + " …"      # a prefix of the sentence: still an exact quote


def expand_clause(sents: List[str], i: int, max_words: int = 250) -> str:
    """Expands sentence i into its surrounding legal clause / full paragraph.
    Preserves preceding and succeeding context up to paragraph boundaries or word limit."""
    if not sents or not (0 <= i < len(sents)):
        return ""
    start = i
    while start > 0 and (i - start) < 2:
        prev = sents[start - 1].strip()
        curr = sents[start].strip()
        if re.match(r"^\(?\d+[\.\)]\s+", curr) or "\n\n" in prev or len(prev) < 20:
            break
        start -= 1
    end = i
    while end < len(sents) - 1 and (end - i) < 2:
        curr = sents[end].strip()
        nxt = sents[end + 1].strip()
        if re.match(r"^\(?\d+[\.\)]\s+", nxt) or "\n\n" in curr or len(nxt) < 20:
            break
        end += 1

    parts = [sents[k].strip().replace("\n", " ") for k in range(start, end + 1)]
    clause = " ".join(p for p in parts if p)
    return clip(clause, words=max_words)


def sid(i: int) -> str:
    return f"[S{i}]"


_RAW_CACHE: dict = {}
_CURRENT: dict = {}


def use_result(res: Optional[dict]):
    """Remember the analysed result so the extractors can read its original PDF text."""
    _CURRENT["res"] = res



def raw_text(sents: List[str], res: Optional[dict] = None) -> str:
    """The judgment as extracted from the PDF (kept by pipeline.analyse); falls back to the sentences joined."""
    if res is None:
        res = _CURRENT.get("res")
    if res is not None and res.get("sentences_list") is sents and res.get("raw_text"):
        return res["raw_text"]
    k = (len(sents), sents[0][:40] if sents else "", sents[-1][-40:] if sents else "")
    if k not in _RAW_CACHE:
        _RAW_CACHE.clear()
        _RAW_CACHE[k] = "\n".join(sents)
    return _RAW_CACHE[k]


def _extract_case_details(sents: List[str]) -> str:
    d = CF.case_details(raw_text(sents))
    g = lambda k: re.sub(r"\s*DATE\s*:?\s*$", "", (d.get(k) or "").strip()) or "not stated in the document"
    base = " | ".join([g("court"), g("case_no"), g("judge"), g("date"), g("parties")])
    extra = []
    if d.get("accused"):
        extra.append(f"ACCUSED: {d['accused']}")
    if d.get("complainant"):
        extra.append(f"COMPLAINANT: {d['complainant']}")
    if extra:
        base += " | " + " | ".join(extra)
    return base


def _charge_list(sents: List[str]) -> list:
    return CF.charges(raw_text(sents))


def _extract_charges(sents: List[str]) -> str:
    ch = _charge_list(sents)
    if not ch:
        return "not stated in the document"
    return " | ".join(f"{act}: " + ", ".join(f"Section {x}" for x in secs) for act, secs in ch)


def _proceedings(sents: List[str], limit: int = 6) -> str:
    raw = raw_text(sents)
    snips = CF.dated_paragraphs(raw, max_chars=300, finder=CF.FactFinder(raw))
    return "\n".join(f"- {x}" for x in snips[:limit]) or "(none found)"


def _format_suggestions(res: dict, desc: Dict[str, str]) -> str:
    if prediction_status(res)["weak"]:
        return "none: this judgment charges or discusses no IPC/BNS offence (the classifier's forced low-confidence guess is withheld)"
    sents = res["sentences_list"]
    lines = []
    for s in res["sections"]:
        num = s.replace("Section ", "").strip()
        bns = f" (≈ BNS {IPC_TO_BNS[num]})" if num in IPC_TO_BNS else ""
        title = one_line(s, desc)
        score = res["scores"].get(s, 0.0)
        ev = res["evidence_idx"].get(s, [])
        ev_str = " ".join(f'{sid(i)} "{expand_clause(sents, i, max_words=200)}"' for i in ev)
        line = f'IPC {num}{bns} "{title}": {score:.2f} | Registered charges: {CF.check_against_charges(num, _charge_list(sents))}'
        if ev_str:
            line += f" | Evidence: {ev_str}"
        lines.append(line)
    return "\n".join(lines) if lines else "none suggested"


_PRECEDENT_PAT = re.compile(
    r'(?:(?:\b|\()\d{4}\)?\s*(?:\d+\s*)?(?:SCC|SCR|INSC|Cri\s*LJ|AIR)\b|AIR\s*\d{4}\b)|'
    r'\b(?:held in|relied upon|laid down in|in the case of|\bcase\s+of\s+[A-Z]|in\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+vs?\.|\b[A-Z][A-Za-z\s.]+\s+vs?\.\s+[A-Z][A-Za-z\s.]+\b)',
    re.I
)


_OPEN_Q, _CLOSE_Q = "\u201c", "\u201d"
_PREC_INTRO = re.compile(r"\bpara(?:graph)?s?[-\s]*(?:No\.?\s*)?\d|judgment\s+titled|reproduc|\bheld\b|observed|\bv(?:s)?\.\s|"
                         r"reads\s+(?:thus|as\s+(?:under|follows))|as\s+under\s*:|the\s+following\s*:|\bcase\s+of\s+[A-Z]", re.I)
_ORDER_INTRO = re.compile(r"impugned\s+order|order\s+(?:dated|bearing|No\.)|taken\s+note\s+of", re.I)


_PARA_MARK = re.compile(r"(?:^|\n|\s)(\d{1,3}(?:\.\d{1,2})?)\s*(?:\.|\)|/-)\s*(?=\n|[A-Z(\u201c]|$)")
_PREC_PARA = re.compile(r"^\W*(?:In\s+(?:the\s+)?(?:case|matter|landmark\s+(?:judgment|decision))\s+(?:of|titled)|"
                        r"(?:Recently,?\s+)?(?:the\s+)?(?:Supreme\s+Court|Division\s+Bench|[A-Z][a-z]+\s+High\s+Court)\s+in\s+(?:the\s+)?(?:case|matter|judgment|a\s+landmark)|"
                        r"In\s+[A-Z][\w.&' ]{2,60}\s+[Vv]s?\.\s)", re.S)
_THIS_CASE = re.compile(r"\b(?:present\s+case|instant\s+case|this\s+case|Applicant\s+in\s+present|in\s+the\s+present\s+matter)\b", re.I)


def _quote_tags(sents: List[str]) -> Dict[int, str]:
    """Which sentences are not this court's own words:
      'precedent'      inside a quotation from another judgment, or in a paragraph that only narrates a cited case
                       ('18. In the case of X Vs. Y, this Court ... enlarged him on bail.')
      'impugned-order' inside a quotation of the order under challenge
    A sentence that closes a quotation and then starts the court's next paragraph is recorded in _MIXED (split point),
    so its two halves are shown with their own tags."""
    tags, depth, kind, start = {}, 0, None, 0
    para_prec, pending = False, False
    _MIXED.clear()
    for i, s in enumerate(sents):
        # ---- paragraph state (numbered paragraphs of the court's own text)
        marks = list(_PARA_MARK.finditer(s)) if depth == 0 else []
        if pending:
            para_prec, pending = bool(_PREC_PARA.search(s.strip())) and not _THIS_CASE.search(s[:200]), False
        if marks:
            rest = s[marks[-1].end():].strip()
            if len(rest.split()) < 4:
                pending = True
            else:
                para_prec = bool(_PREC_PARA.search(rest)) and not _THIS_CASE.search(rest[:200])
        # ---- quotation state
        o, c = s.count(_OPEN_Q), s.count(_CLOSE_Q)
        if depth == 0 and o > c:
            ctx = " ".join(sents[max(0, i - 2):i + 1])
            kind = "impugned-order" if _ORDER_INTRO.search(ctx) else ("precedent" if _PREC_INTRO.search(ctx) else None)
            start = i
        was_inside = depth > 0 or o > c
        depth = max(0, depth + o - c)
        if depth and i - start > 60:                 # unclosed quotation mark: stop tagging
            depth = 0
        if was_inside and kind:
            tags[i] = kind
            if depth == 0:                           # quotation closed in this sentence: is there court text after it?
                cut = s.rfind(_CLOSE_Q) + 1
                if len(s[cut:].split()) >= 8:
                    _MIXED[" ".join(s.split())] = (kind, cut)
                    para_prec = False
                    after = s[cut:]
                    m = list(_PARA_MARK.finditer(after))
                    if m and len(after[m[-1].end():].split()) >= 4:
                        para_prec = bool(_PREC_PARA.search(after[m[-1].end():].strip()))
        elif para_prec and not _THIS_CASE.search(s):
            tags[i] = "precedent"
        if depth == 0:
            kind = None
    return tags


_MIXED: Dict[str, tuple] = {}


def pieces(text: str) -> List[tuple]:
    """[(tag, text)] for a judgment sentence: one piece, or two when a quotation ends inside it."""
    _quote_map()
    key = " ".join(text.split())
    if key in _MIXED:
        kind, cut = _MIXED[key]
        flat_t = key
        return [(kind, flat_t[:cut].strip()), ("fact", flat_t[cut:].strip())]
    return [(classify_passage_type(text), text)]


def render_sentence(sents: List[str], i: int, max_words: int = 220) -> str:
    """'[S12: fact] sentence + neighbours' for prompts. Neighbours are only joined when they carry the same tag, so a
    precedent sentence never smuggles in court text (or the reverse); split sentences are shown as two tagged pieces."""
    s = sents[i]
    if " ".join(s.split()) in _MIXED:
        return tagged(f"S{i}", s, max_words)
    tag = classify_passage_type(s)
    same = lambda k: (0 <= k < len(sents) and " ".join(sents[k].split()) not in _MIXED and classify_passage_type(sents[k]) == tag
                      and len(sents[k].strip()) >= 20 and not re.match(r"^\(?\d+[\.\)]\s+", sents[k].strip()))
    a = i
    while a > i - 2 and same(a - 1) and not re.match(r"^\(?\d+[\.\)]\s+", sents[a].strip()):
        a -= 1
    b = i
    while b < i + 2 and same(b + 1):
        b += 1
    text = " ".join(sents[k].strip().replace("\n", " ") for k in range(a, b + 1))
    return f"[S{i}: {tag}] {clip(text, max_words)}"


def tagged(sid_label: str, text: str, max_words: int = 110) -> str:
    """'[S75: precedent] ...quoted part.\n[S75: fact] the court's own words' (one line per piece)."""
    return "\n".join(f"[{sid_label}: {t}] {clip(x.replace(chr(10), ' '), max_words)}" for t, x in pieces(text) if x)


_QT_CACHE: dict = {}


def _quote_map() -> Dict[str, str]:
    res = _CURRENT.get("res")
    if not res or not res.get("sentences_list"):
        return {}
    if _QT_CACHE.get("k") != id(res):
        sents = res["sentences_list"]
        _QT_CACHE.clear()
        _QT_CACHE["k"] = id(res)
        tags = _quote_tags(sents)
        m = {" ".join(sents[i].split()): t for i, t in tags.items()}
        for i, x in enumerate(sents):
            # "he has relied on ... Pratima Mohanty and Ors. and ... vs. Bhajan Lal and Ors. to argue that the present case
            # ...": a PARTY citing cases in this case, not a cited judgment speaking - this case's own fact
            if i in tags or not _PRECEDENT_PAT.search(x):
                continue
            if _PREC_PARA.search(x.strip()) or _PARA_MARK.match(x):   # "8. In the case of Emperor Vs." - the court's own
                continue                                               # paragraph about a precedent
            if _CITED_BY_PARTY.search(" ".join(sents[max(0, i - 1):i + 1])) and not _HOLDING.search(x):
                m[" ".join(x.split())] = "fact"
        _QT_CACHE["m"] = m
    return _QT_CACHE["m"]


_CITED_BY_PARTY = re.compile(r"\b(?:relied\s+(?:up)?on\s+(?:the\s+)?(?:\w+\s+){0,3}(?:decisions?|judgments?|cases?|rulings?|authorit\w+|precedents?)|"
                             r"placed\s+reliance|reliance\s+(?:is|was|has\s+been)\s+placed|(?:has|have|had)\s+relied|"
                             r"learned\s+(?:senior\s+)?(?:counsel|advocate|APP|PP|A\.?P\.?P))\b", re.I)
_HOLDING = re.compile(r"\b(?:held|holding|observed|granted|enlarged|directed|acquitted|convicted|quashed|set\s+aside|"
                      r"ruled|opined|laid\s+down)\b", re.I)


def classify_passage_type(text: str) -> str:
    """'precedent' (another judgment), 'impugned-order' (quoted order under challenge) or 'fact' (this case)."""
    t = _quote_map().get(" ".join(text.split()))
    if t:
        return t
    return "precedent" if _PRECEDENT_PAT.search(text) else "fact"


_PENAL_RX = re.compile(r"Indian\s+Penal\s+Code|Penal\s+Code|\bI\.?\s?P\.?\s?C\b|Bharatiya\s+Nyaya\s+Sanhita|\bBNS\b", re.I)
WEAK_SCORE = float(os.environ.get("IPC_WEAK_SCORE", "0.7"))


def prediction_status(res: Optional[dict]) -> dict:
    """weak=True when the judgment names no IPC/BNS offence and the best score is low: the classifier is calibrated to
    always return at least one section, so on a non-criminal judgment (e.g. a service writ) that section is a forced guess."""
    if not res or not res.get("sentences_list"):
        return {"weak": False, "reason": "", "top": 0.0}
    st = res.get("_pred_status")
    if st is None:
        sents = res["sentences_list"]
        top = max((res["scores"].get(s, 0.0) for s in res.get("sections", [])), default=0.0)
        charges = _charge_list(sents)
        penal = bool(_PENAL_RX.search(raw_text(sents, res))) or bool(charges)
        bns_to_ipc = {v.split("(")[0]: k for k, v in IPC_TO_BNS.items()}
        registered = set()
        for act, nums in charges:
            for n in nums:
                base = re.sub(r"\(.*$", "", n).upper()
                if act == "IPC":
                    registered.add(base)
                elif act == "BNS":
                    registered.add(bns_to_ipc.get(base, "BNS" + base))
        predicted = {s.replace("Section ", "").strip().upper() for s in res.get("sections", [])}
        weak, reason = False, ""
        if res.get("sections") and charges and not (predicted & registered):
            weak = True
            listed = "; ".join(f"{act}: {', '.join(n)}" for act, n in charges)
            reason = (f"None of the suggested sections is among the registered charges ({listed}). The classifier was probably "
                      "misled by sections discussed in cited judgments.")
        elif res.get("sections") and not penal and top < WEAK_SCORE:
            weak = True
            reason = (f"The judgment names no IPC/BNS offence and the best score is low ({top:.2f}). The classifier always returns "
                      "at least one section, so this is its nearest guess, likely not applicable.")
        st = res["_pred_status"] = {"weak": weak, "reason": reason, "top": round(top, 3)}
    return st


def _clean_citations(text: str) -> str:
    return text.strip()


def _fact_passages(res: dict, max_sentences: int = 8) -> str:
    sents = res["sentences_list"]
    candidates = list(range(min(res.get("n_head", 8), len(sents)))) + [i for i in res.get("key_idx", []) if i >= res.get("n_head", 8)]
    seen, out = set(), []
    for i in candidates:
        if i in seen or i >= len(sents):
            continue
        seen.add(i)
        st = sents[i]
        if _BOILER.search(st) or len(st.split()) < 5:
            continue
        own = [x for t, x in pieces(st) if t == "fact"]
        if not own:
            continue
        out.append(render_sentence(sents, i, 220) if len(own[0]) == len(" ".join(st.split())) or own[0] == st
                   else f"[S{i}: fact] {clip(own[0], 220)}")
        if len(out) >= max_sentences:
            break
    return "\n".join(out) if out else "(none extracted)"


def _precedent_passages(res: dict, max_sentences: int = 5) -> str:
    """Precedents and earlier judgments cited by the court (discovered by Hyena and pattern matching)."""
    sents = res.get("sentences_list", [])
    scores = res.get("sentence_scores")
    if not sents:
        return "(none cited)"
    if scores and len(scores) == len(sents):
        ranked = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    else:
        ranked = res.get("key_idx", [])

    out = []
    seen = set()
    for i in ranked:
        if i in seen or i >= len(sents):
            continue
        seen.add(i)
        st = sents[i]
        if _BOILER.search(st) or len(st.split()) < 6:
            continue
        if classify_passage_type(st) == "precedent":
            full_clause = expand_clause(sents, i, max_words=220)
            out.append(f"{sid(i)} {full_clause}")
            if len(out) >= max_sentences:
                break
    if len(out) < max_sentences:
        for i, st in enumerate(sents):
            if i in seen or _BOILER.search(st) or len(st.split()) < 6:
                continue
            if classify_passage_type(st) == "precedent":
                seen.add(i)
                full_clause = expand_clause(sents, i, max_words=220)
                out.append(f"{sid(i)} {full_clause}")
                if len(out) >= max_sentences:
                    break
    return "\n".join(out) if out else "(no external precedents cited in the order)"


def _disposition(res: dict, max_words: int = 260) -> List[int]:
    sents = res["sentences_list"]
    out, n = [], 0
    for i in range(len(sents) - 1, -1, -1):
        if _BOILER.search(sents[i]) or len(sents[i]) < 25:
            continue
        n += len(sents[i].split())
        out.append(i)
        if n > max_words:
            break
    return sorted(out)


def _disposition_passages(res: dict) -> str:
    sents = res["sentences_list"]
    d = CF.disposition(raw_text(sents), max_chars=3500)
    d = re.split(r"\bDigitally\s+signed\b", d)[0].strip()          # drop the e-signature stamp and page footers
    return d or "(not stated in the document)"


def build_context(res: dict, desc: Dict[str, str]) -> str:
    use_result(res)
    sents = res["sentences_list"]
    case_details = _extract_case_details(sents)
    charges = _extract_charges(sents)
    suggestions = _format_suggestions(res, desc)
    statute_signal = "(none: no IPC/BNS offence in this judgment)" if prediction_status(res)["weak"] else format_statute_signal(res.get("sections", []), desc)
    facts = _fact_passages(res)
    precedents = _precedent_passages(res)
    disposition = _disposition_passages(res)

    return f"""[CONTEXT]
CASE DETAILS: {case_details}
CHARGES: {charges}
MODEL SUGGESTIONS (classifier guesses, unverified, NOT charges): {suggestions}
STATUTORY INGREDIENTS (Official IndiaCode):
{statute_signal}
PROCEEDINGS (FIR and hearings, extracted by code):
{_proceedings(sents)}
FACT PASSAGES:
{facts}
PRECEDENTS & AUTHORITIES CITED IN ORDER:
{precedents}
DISPOSITION (final paragraphs):
{disposition}"""


def build_explain(res: dict, desc: Dict[str, str], role: str = "accused") -> tuple:
    """-> (system, messages) in the format tailored to the lawyer's perspective."""
    if res:
        use_result(res)
    r = (role or "accused").lower()
    if r == "complainant":
        sys_p = EXPLAIN_SYSTEM_COMPLAINANT
        usr_p = EXPLAIN_USER_COMPLAINANT
    elif r == "neutral":
        sys_p = EXPLAIN_SYSTEM_NEUTRAL
        usr_p = EXPLAIN_USER_NEUTRAL
    else:  # accused / defence
        sys_p = EXPLAIN_SYSTEM_ACCUSED
        usr_p = EXPLAIN_USER_ACCUSED
    return sys_p, [{"role": "user", "content": build_context(res, desc) + "\n\n[USER]\n" + usr_p}]


def _narrative_summary(res: dict, max_sentences: int = 14) -> tuple[str, set[str]]:
    """Hyena's most important factual and precedent sentences in chronological document order (~900 tokens)."""
    sents = res.get("sentences_list", [])
    scores = res.get("sentence_scores")
    used_sids = set()
    if not sents:
        return "(none)", used_sids

    opening_idx = list(range(min(4, len(sents))))
    if scores and len(scores) == len(sents):
        ranked = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    else:
        ranked = res.get("key_idx", [])

    disp_idx = _disposition(res)
    disp_tail = disp_idx[-2:] if disp_idx else [len(sents) - 1]

    chosen = set(opening_idx + disp_tail)
    for i in ranked:
        if len(chosen) >= max_sentences:
            break
        if i in chosen or i >= len(sents):
            continue
        st = sents[i]
        if _BOILER.search(st) or len(st.split()) < 6:
            continue
        chosen.add(i)

    chosen_sorted = sorted(chosen)
    lines = []
    for i in chosen_sorted:
        used_sids.add(f"S{i}")
        lines.append(render_sentence(sents, i, 220))

    return "\n".join(lines) if lines else "(none)", used_sids


def build_static_context(res: dict, desc: Dict[str, str], hide_hints: bool = False) -> tuple[str, set[str]]:
    """Built once per analyzed document: case details, charges, Hyena predictions with evidence, narrative core."""
    use_result(res)
    sents = res["sentences_list"]
    case_details = _extract_case_details(sents)
    charges = _extract_charges(sents)

    ev_used_sids = set()
    sugg_lines = []
    for s in res.get("sections", []):
        num = s.replace("Section ", "").strip()
        bns = f" (≈ BNS {IPC_TO_BNS[num]})" if num in IPC_TO_BNS else ""
        title = one_line(s, desc)
        score = res["scores"].get(s, 0.0)
        is_reg = CF.check_against_charges(num, _charge_list(sents))
        ev_indices = res.get("evidence_idx", {}).get(s, [])
        ev_parts = []
        for i in ev_indices:
            if 0 <= i < len(sents):
                ev_used_sids.add(f"S{i}")
                ev_parts.append(render_sentence(sents, i, 200))
        line = f'- IPC {num}{bns} "{title}" [Confidence: {score:.2f} | Registered charges: {is_reg}]'
        if ev_parts:
            line += "\n  Evidence: " + " ".join(ev_parts)
        sugg_lines.append(line)

    sugg_str = "\n".join(sugg_lines) if sugg_lines else "none suggested"
    weak = prediction_status(res)["weak"]
    if weak or hide_hints:
        sugg_str = ("none: this judgment charges or discusses no IPC/BNS offence (forced low-confidence guess withheld)" if weak else
                    "withheld for this question: answer from the facts and statutory ingredients, not from a classifier")
        ev_used_sids = set()
    summary_str, sum_sids = _narrative_summary(res, max_sentences=12)
    statute_signal = ("(not shown)" if (weak or hide_hints) else format_statute_signal(res.get("sections", []), desc, max_text_len=200))
    all_static_sids = ev_used_sids | sum_sids
    precedents_str = _precedent_passages(res)

    block = f"""[STATIC CASE CONTEXT (case details, charges, proceedings and summary were extracted from the judgment)]
CASE DETAILS: {case_details}
CHARGES REGISTERED: {charges}
CLASSIFIER HINTS (unverified guesses by a statistical model, NOT charges, often wrong):
{sugg_str}
STATUTORY INGREDIENTS (Official IndiaCode):
{statute_signal}
PROCEEDINGS (FIR and hearings, extracted by code):
{_proceedings(sents)}
PRECEDENTS & AUTHORITIES CITED IN ORDER:
{precedents_str}
CASE SUMMARY (Key narrative in document order):
{summary_str}"""
    return block, all_static_sids


_ASKING = re.compile(r"^\s*(?:who|whom|whose|what|when|where|which|why|how|is|are|was|were|do|does|did|can|could|should|would|will|"
                     r"has|have|had|list|summari[sz]e|explain|describe|give|tell|show|compare|draft|write|find|identify)\b", re.I)


def _user_text(content: str) -> str:
    """The user's own words inside a stored chat turn (the turn also carries the passages it was answered from)."""
    t = content.split("[USER QUESTION]\n")[-1]
    return t.split("\n(In the context of this conversation")[0].strip()


def user_notes(turns: List[dict], max_notes: int = 12) -> List[str]:
    """Things the user TOLD the assistant (not questions): client details, dates, corrections. Kept for the whole
    conversation, word for word, in order (a later note overrides an earlier one) - LongMemEval-style memory."""
    notes = []
    for t in turns:
        if t.get("role") != "user":
            continue
        u = " ".join(_user_text(t.get("content", "")).split())
        if u and not u.endswith("?") and not _ASKING.match(u) and len(u.split()) >= 3:
            notes.append(clip(u, 220))
    return notes[-max_notes:]


_CHAT_BUDGET = int(os.environ.get("IPC_NUM_CTX", "8192")) - int(os.environ.get("IPC_CHAT_TOKENS", "1000")) - 400


def _tokens(text: str) -> int:
    return len(text or "") // 3                          # conservative for legal English (about 3.5-4 characters per token)


def _distill_older_history(turns: List[dict], max_bullets: int = 8) -> tuple[str, List[dict]]:
    """Separates chat history into an immediate verbatim window and a compressed memory log for older turns.
    Statements the user made (notes, corrections) are pinned verbatim for the whole conversation, separately."""
    if len(turns) < 4:
        return "", turns

    immediate_turns = turns[-4:]
    older_turns = turns[:-4]

    lines = []
    for i in range(0, len(older_turns) - 1, 2):
        q = _user_text(older_turns[i].get("content", ""))
        ans = older_turns[i + 1].get("content", "").split("\n")[0].strip()
        q_clean = clip(q.replace(chr(10), " "), 50)
        ans_clean = clip(ans.replace(chr(10), " "), 90)
        lines.append(f'• Q: "{q_clean}" -> A: {ans_clean}')

    if not lines:
        return "", immediate_turns

    memory_block = "[PRIOR CONVERSATION MEMORY (Key Q&As from earlier in this session)]:\n" + "\n".join(lines[-max_bullets:])
    return memory_block, immediate_turns


def _notes_block(notes: List[str]) -> str:
    return ("[NOTES THE USER GAVE IN THIS CONVERSATION, oldest first (the user's own information, not from the judgment). "
            "When two notes are about the same thing, ONLY THE LATEST ONE IS VALID: a 'correction' replaces the earlier note. "
            "Use them when asked]:\n" + "\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1)))


_OFFENCE_Q = re.compile(r"\b(offen[cs]es?|charge[ds]?|IPC|BNS|penal|section[s]?|punishable|liable)\b", re.I)


_ORDER_Q = re.compile(r"\b(conditions?|direct(?:ed|ions?)|order(?:ed|s)?|relief|final|outcome|decided|decision|dispos|granted|"
                      r"quash(?:ed)?|set aside|remand|result|verdict|allowed|dismissed|list all|"
                      r"summar\w*|overall|conclu\w*)\b", re.I)   # any summary must state the real outcome


def build_chat(question: str, routed: dict, res: Optional[dict], desc: Dict[str, str], turns: List[dict], role: str = "accused",
               resolved: Optional[str] = None) -> tuple:
    """Chat prompt: merges static case context, older memory log, dynamic passages, and recent dialogue turns."""
    r = (role or "accused").lower()
    if r == "complainant":
        role_directive = (
            "PERSPECTIVE: You are acting as a dedicated, strategic personal legal assistant to the COMPLAINANT / PROSECUTION's counsel.\n"
            "PARTY ATTRIBUTION: The Complainant is the victim / first informant reporting the crime. The Accused (Applicant) is the party facing the charges. Never attribute the criminal acts to the complainant.\n"
            "Analyze all facts, evidence, and legal issues from the prosecution's tactical perspective: establish and substantiate charges beyond "
            "reasonable doubt, highlight incriminating evidence and witness corroboration [S..], evaluate grounds to oppose bail or seek custodial interrogation, "
            "and suggest proactive procedural steps (summoning key witnesses under s.311 CrPC / s.348 BNSS, pressing for additional/altered charges under s.216 CrPC / s.239 BNSS, "
            "and victim compensation). Be direct, actionable, and quote evidence [S..] rigorously. "
            "Also state what the defence has on record against these points (rule 9)."
        )
    elif r == "neutral":
        role_directive = (
            "PERSPECTIVE: You are an objective judicial legal assistant helping a lawyer read an Indian court judgment.\n"
            "Provide balanced, direct, and factual answers based strictly on the CONTEXT."
        )
    else:  # accused / defence
        role_directive = (
            "PERSPECTIVE: You are acting as a dedicated, strategic personal legal assistant to the ACCUSED's DEFENCE counsel.\n"
            "PARTY ATTRIBUTION: The Accused (Applicant) is your client facing the allegations. The Complainant is the opposing party who lodged the complaint.\n"
            "Analyze all facts, evidence, and legal issues from the defence's tactical perspective: identify missing statutory ingredients, expose contradictions, "
            "delays in FIR, lack of independent corroboration, and procedural defects [S..]. Formulate strong arguments for regular/anticipatory bail or discharge, "
            "and outline sharp cross-examination angles and documents to summon under Section 91 CrPC / Section 94 BNSS. Be direct, actionable, and quote evidence [S..] rigorously. "
            "Only raise a weakness the document shows, and always state what the record has against the accused (rule 9): a defence argument "
            "the court has rejected must be flagged as rejected, with how it could still be used (e.g. at trial)."
        )

    system_prompt = f"{CHAT_SYSTEM}\n\n{role_directive}"


    # Dynamic passages
    dynamic_lines = [f"[DYNAMIC PASSAGES FOR THIS QUESTION (route: {routed['route']}, risk of wrong answer: {routed['p_hallucination_high']:.2f})]"]
    added = 0
    excluded_sids = set()
    # The case context goes into the first ANSWERED turn. Earlier turns can exist without it: after a clarifying question
    # the conversation already holds the vague question and the clarification (that used to drop the context entirely).
    context_sent = any("[STATIC CASE CONTEXT" in (t.get("content") or "") for t in turns)
    if res and not context_sent:
        static_block, static_sids = build_static_context(res, desc, hide_hints=bool(_OFFENCE_Q.search(resolved or question)))
        excluded_sids = static_sids
    else:
        static_block = ""

    for p in routed.get("passages", []):
        pid = p["id"]
        # Deduplication: suppress sentence if already present in static block
        if pid in excluded_sids and p["where"] == "judgment":
            continue
        text = clip(p["text"].replace("\n", " "), 110)
        if p["where"] == "judgment":
            dynamic_lines.append(tagged(pid, p["text"], 110))
        else:
            stype = p.get("source_type", "library")
            dynamic_lines.append(f"[KB-{stype}: {pid}] {text}")
        added += 1
        if added >= routed.get("max_passages", 8):
            break

    if res and _ORDER_Q.search(question):
        dynamic_lines.append("[OPERATIVE ORDER OF THIS COURT (end of the judgment, extracted by code; quote every condition from here). "
                             "This is the OUTCOME of this case: never state a different outcome (e.g. 'dismissed' when it says 'allowed')] "
                             + _disposition_passages(res))
        added += 1

    # "What evidence points to Section 504?": the sentences the section model linked to that section. They are shown even when
    # the prediction is weak (then the classifier hints are hidden), because the user asked about this section by name.
    if res and res.get("evidence_idx"):
        asked = {f"Section {m}" for m in re.findall(r"\bsec(?:tion|\.)?\s*(\d{1,3}[A-Z]?)\b", question, re.I)}
        for sec in sorted(asked & set(res["evidence_idx"])):
            idx = [i for i in res["evidence_idx"][sec] if 0 <= i < len(res["sentences_list"])][:3]
            if idx:
                dynamic_lines.append(f"[SECTION MODEL EVIDENCE for {sec} (sentences the section-prediction model linked to it; "
                                     f"{sec} is the model's suggestion, NOT a charge in this case unless the judgment says so)]")
                dynamic_lines += [tagged(f"S{i}", res["sentences_list"][i], 110) for i in idx]
                added += 1

    if _OFFENCE_Q.search(question):
        dynamic_lines.append("[IPC -> BNS REFERENCE (official correspondence; BNS in force from 1 July 2024)] " +
                             "; ".join(f"IPC {k} ({IPC_TITLES[k]}) = BNS {v}" if k in IPC_TITLES else f"IPC {k} = BNS {v}"
                                       for k, v in IPC_TO_BNS.items()))
        dynamic_lines.append("NOTE: choose a section only when the facts match its title. An ACCIDENTAL or NEGLIGENT act is never "
                             "murder (302), attempt to murder (307) or culpable homicide: use the rash/negligent-act sections "
                             "(336, 337, 338, 304A). Match each section to the specific allegation it covers.")
        added += 1

    dyn_block = ""
    if added > 0:
        if routed.get("caveat"):
            dynamic_lines.append("NOTE: the question stayed unclear after two clarifying rounds; answer cautiously and state what is uncertain.")
        if routed["route"] == "Search":
            dynamic_lines.append("NOTE: passages marked [KB-...] come from the general legal library, not this case's facts.")
        dyn_block = "\n".join(dynamic_lines)

    q_line = f"[USER QUESTION]\n{question}"
    if resolved and resolved.strip().lower() != question.strip().lower():
        q_line += f"\n(In the context of this conversation the question means: {resolved})"
    # Earlier turns stay intact (stable KV-cache prefix); the static context is added to whichever turn first needs it.
    parts = [p for p in (static_block, dyn_block) if p]
    user = ("\n\n".join(parts) + "\n\n" if parts else "") + q_line
    notes = user_notes(turns)
    if notes:                                     # the user's own notes, latest last, right next to the question
        user = ("\n\n".join(parts) + "\n\n" if parts else "") + _notes_block(notes) + "\n\n" + q_line
    msgs = turns + [{"role": "user", "content": user}]
    if _tokens(system_prompt) + sum(_tokens(m.get("content", "")) for m in msgs) <= _CHAT_BUDGET:
        return system_prompt, msgs
    # The conversation no longer fits the model's window (every stored turn carries its passages). Instead of letting the
    # server cut the oldest messages - the case context and the user's first notes - send a compact form: the case
    # context again, the user's notes verbatim, older Q&As as one-liners, the last two exchanges without their passages.
    if res:
        static_block, static_sids = build_static_context(res, desc, hide_hints=bool(_OFFENCE_Q.search(resolved or question)))
    memory_block, recent = _distill_older_history(turns)
    if notes:
        memory_block = _notes_block(notes) + ("\n\n" + memory_block if memory_block else "")
    recent = [{"role": t["role"], "content": _user_text(t.get("content", "")) if t.get("role") == "user" else clip(t.get("content", ""), 150)}
              for t in turns[-4:]]
    parts = [p for p in (static_block, memory_block, dyn_block) if p]
    user = "\n\n".join(parts) + "\n\n" + q_line
    return system_prompt, recent + [{"role": "user", "content": user}]


# ───────────────────────────── no chat model: answers built directly from the extracted material ─────────────────────────────
_PROC = re.compile(r"\b(FIR|First Information|chargesheet|charge-sheet|Trial Court|Sessions|Special Court|High Court|appeal|Appeal|"
                   r"remand|arrest|bail)\b")


def extractive_explain(res: dict, desc: Dict[str, str], role: str = "accused") -> str:
    use_result(res)
    r = (role or "accused").lower()
    sents = res["sentences_list"]
    q = lambda i: f'{sid(i)} "{clip(sents[i].replace(chr(10), " "), 60)}"'
    proc = [i for i, s in enumerate(sents) if _PROC.search(s)][:6]

    if r == "complainant":
        out = [
            "⚖️ STRATEGIC PROSECUTION BRIEFING (Extractive Summary)",
            "",
            "1. Case & Complainant Snapshot",
            "   " + _extract_case_details(sents),
            "",
            "2. Charges (as registered)",
            "   " + _extract_charges(sents),
            "",
            "3. Evidentiary Arsenal & Proven Ingredients (by Section)",
        ]
        for s in res["sections"]:
            num = s.replace("Section ", "").strip()
            bns = f" (≈ BNS {IPC_TO_BNS[num]})" if num in IPC_TO_BNS else ""
            out.append(f'   • IPC {num}{bns} "{one_line(s, desc)}": Confidence {res["scores"][s]:.2f} | Registered: {CF.check_against_charges(num, _charge_list(sents))}')
            ev_lines = [f"      - {q(i)}" for i in res["evidence_idx"].get(s, [])]
            out += ev_lines if ev_lines else ["      - (no direct evidence sentence extracted)"]
        out += [
            "",
            "4. Procedural & Custodial Factors (Record Highlights)",
        ]
        out += ["   " + q(i) for i in proc] or ["   not stated in the document"]
        out += [
            "",
            "5. Actionable Roadmap for Prosecution Counsel",
            "   • Scrutinize Section 311 CrPC / Section 348 BNSS to examine un-examined witnesses if material gaps exist.",
            "   • Evaluate whether evidence discloses grounds to alter or add charges under Section 216 CrPC / Section 239 BNSS.",
            "   • Oppose bail applications by emphasizing seriousness of allegations, severity of punishment, and potential witness tampering.",
            f"   • Disposition: {_disposition_passages(res)[:400]}",
            "",
            "(No chat model is selected; assembled directly from extracted passages. Decision support only.)"
        ]
        return "\n".join(out)
    elif r == "accused":
        out = [
            "🛡️ STRATEGIC DEFENCE BRIEFING (Extractive Summary)",
            "",
            "1. Case & Accused Snapshot",
            "   " + _extract_case_details(sents),
            "",
            "2. Charges (as registered)",
            "   " + _extract_charges(sents),
            "",
            "3. Statutory Scrutiny & Potential Defence Angles (by Section)",
        ]
        for s in res["sections"]:
            num = s.replace("Section ", "").strip()
            bns = f" (≈ BNS {IPC_TO_BNS[num]})" if num in IPC_TO_BNS else ""
            out.append(f'   • IPC {num}{bns} "{one_line(s, desc)}": Model Score {res["scores"][s]:.2f} | Registered: {CF.check_against_charges(num, _charge_list(sents))}')
            ev_lines = [f"      - Evidence cited: {q(i)}" for i in res["evidence_idx"].get(s, [])]
            out += ev_lines if ev_lines else ["      - No prima facie evidence sentence linked"]
            out.append("      - Defence Angle: Verify whether all essential ingredients (mens rea, overt act, common intention) are strictly established.")
        out += [
            "",
            "4. Procedural Record & Bail Footprint",
        ]
        out += ["   " + q(i) for i in proc] or ["   not stated in the document"]
        out += [
            "",
            "5. Actionable Roadmap for Defence Counsel",
            "   • Explore grounds for Regular / Anticipatory Bail (s.437/439 CrPC or s.480/483 BNSS) or Discharge (s.227/239 CrPC or s.250/262 BNSS).",
            "   • Formulate cross-examination on contradictions between FIR and subsequent witness statements.",
            "   • Apply under Section 91 CrPC / Section 94 BNSS for production of crucial un-relied documents (CCTV, CDR, station diary entries).",
            f"   • Disposition: {_disposition_passages(res)[:400]}",
            "",
            "(No chat model is selected; assembled directly from extracted passages. Decision support only.)"
        ]
        return "\n".join(out)
    else:
        out = [
            "1. Case snapshot",
            "   " + _extract_case_details(sents),
            "",
            "2. Charges (as registered)",
            "   " + _extract_charges(sents),
            "",
            "3. Proceedings",
        ]
        out += ["   " + q(i) for i in proc] or ["   not stated in the document"]
        out += ["", "4. Model suggestions"]
        for s in res["sections"]:
            num = s.replace("Section ", "").strip()
            bns = f" (≈ BNS {IPC_TO_BNS[num]})" if num in IPC_TO_BNS else ""
            out.append(f'   IPC {num}{bns} "{one_line(s, desc)}": {res["scores"][s]:.2f} - supported | Registered: {CF.check_against_charges(num, _charge_list(sents))}')
            out += [f"      {q(i)}" for i in res["evidence_idx"].get(s, [])]
        out += ["", "5. Outcome", f"   {_disposition_passages(res)[:400]}"]
        out += ["", "(No chat model is selected, so this is assembled from the extracted passages without a written explanation. "
                "Decision support only.)"]
        return "\n".join(out)


def extractive_chat(routed: dict) -> str:
    if routed["route"] == "Clarify":
        return routed["ask_user"]
    out = [f"Route: {routed['route']} (risk {routed['p_hallucination_high']:.2f}). Most relevant passages:"]
    for p in routed["passages"][:3]:
        tag = f"[{p['id']}: {classify_passage_type(p['text'])}]" if p["where"] == "judgment" else f"[KB-{p.get('source_type', 'lib')}:{p['id']}]"
        out.append(f'{tag} "{clip(p["text"].replace(chr(10), " "), 70)}"')
    return "\n".join(out)


CLARIFY_SYSTEM = (
    "You are an offline legal assistant helping a lawyer read an Indian court judgment. "
    "The user's query is ambiguous or lacks necessary context to answer safely. "
    "Your objective is to ask ONE direct, polite, concise clarifying question. "
    "Do NOT answer the question yet. Do NOT invent legal opinions. Just ask the single clarifying question."
)


def build_clarify_prompt(query: str, routed: dict, res: Optional[dict]) -> tuple[str, list[dict]]:
    """Builds a micro-prompt for the LLM to synthesize a natural, case-aware clarifying question."""
    high_signals = [k for k, v in routed.get("evidence", {}).items() if v == "High"]
    reason = routed.get("reason", "Ambiguous question")
    case_context = ""
    if res:
        use_result(res)
        sents = res.get("sentences_list", [])
        case_details = _extract_case_details(sents)
        charges = _extract_charges(sents)
        case_context = f"\nCase Context: {case_details}\nRegistered Charges: {charges}"

    user_msg = f"""The user asked: "{query}"

Bayesian Epistemic Analysis:
- Dominant ambiguity cause: {reason}
- High-risk signals: {", ".join(high_signals) or "Vague / missing context"}
{case_context}

Task: Formulate ONE concise, natural clarifying question (under 25 words) to help the user specify what they mean."""
    return CLARIFY_SYSTEM, [{"role": "user", "content": user_msg}]
