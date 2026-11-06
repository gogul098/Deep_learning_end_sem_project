"""Facts that code can read straight from an Indian court order, without any model:
case details from the header, registered charges grouped by Act, every cited section,
and an IPC -> BNS lookup for the model's suggestions.

These are filled into the explanation by code, so the chat model never has to guess them."""
import re

# ---------------------------------------------------------------- text helpers
def flat(text: str) -> str:
    """Join PDF lines into running text (undo hyphenation and line breaks)."""
    text = re.sub(r"-\n\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- case details
def case_details(raw: str, head_lines: int = 60) -> dict:
    """Court, case number, parties, judge(s) and date from the first page of the order."""
    lines = [l.strip() for l in raw.splitlines()[:head_lines] if l.strip()]
    head = "\n".join(lines)
    d = {}
    m = re.search(r"IN\s+THE\s+(SUPREME\s+COURT\s+OF\s+INDIA|HIGH\s+COURT\s+OF[^\n]+?)(?:\n|$)", head, re.I)
    if m:
        d["court"] = re.sub(r"\s+", " ", m.group(1)).strip().title()
    m = re.search(r"\b((?:[A-Z][A-Z.()/ ]*?)\s(?:APPLICATION|APPEAL|PETITION|REVISION|CASE|WRIT PETITION)\s*"
                  r"(?:\(\w+\)\s*)?NO\.?\s*[\w/-]+\s+OF\s+\d{4})", head, re.I)
    if m:
        d["case_no"] = re.sub(r"\s+", " ", m.group(1)).strip()

    role_kw = re.compile(r"\b(Applicant|Appellant|Petitioner|Respondent|Accused|Complainant|First Informant|Informant)s?\b", re.I)
    role_line = re.compile(r"^[.\s\u2026]*(?:Applicant|Appellant|Petitioner|Respondent|Accused|Complainant|First Informant|Informant)s?\b|^[.\s]*$", re.I)
    stop_header = re.compile(r"^(?:CORAM|DATE|DATED|PRONOUNCED|Mr\.|Ms\.|Advocate|Advocates|Appearance|\.{4,})", re.I)
    clean = lambda x: re.sub(r"^\d+\.\s*", "", re.sub(r"\s*(?:\.{2,}|\u2026).*$|\s{2,}.*$", "", x)).strip(" .,")
    counsel = re.compile(r"^(?:Through|Mr\.?\s|Mrs\.?\s|Ms\.?\s|Shri\s|Smt\.?\s)|\bAdvocates?\b|\bAAG\b|\bAPP\b|\bwith$", re.I)
    header_end = re.compile(r"^(?:Whether|judgment is pronounced|Uploaded|Pronounced|Reserved|CORAM|IA No|.*\bOF\s+\d{4}\s*$)", re.I)

    def numbered(block):
        """'1. Jaswant Singh, age 26 years S/o ...' lines -> ['Jaswant Singh', ...] (parties listed one per number)."""
        out = []
        for x in block:
            m = re.match(r"^\d{1,2}\.\s*([A-Z][^,]{2,80}?)(?:,|\s+(?:age|aged|S/o|D/o|W/o|R/o|through)\b|$)", x)
            if m and not counsel.search(x):
                out.append(m.group(1).strip(" ."))
        return out

    def summarise(names):
        return names[0] if len(names) == 1 else (f"{names[0]} & {names[1]}" if len(names) == 2 else f"{names[0]} & {len(names) - 1} others ({', '.join(names[1:])})")

    for i, l in enumerate(lines):
        if re.fullmatch(r"(?:Versus|Vs\.?|V/s\.?|v\.)", l, re.I):
            before_lines = lines[max(0, i - 4):i]
            after_lines = []
            for x in lines[i + 1:min(len(lines), i + 10)]:
                if stop_header.search(x) and not role_kw.search(x):
                    break
                after_lines.append(x)

            start = max([k + 1 for k in range(max(0, i - 16), i) if header_end.search(lines[k])] or [max(0, i - 16)])
            b_names = numbered(lines[start:i])
            a_names = numbered(after_lines)
            before = next((x for x in reversed(before_lines) if not role_line.search(x) and not counsel.search(x)
                           and not re.match(r"^\d+\s+OF\s+\d{4}$", x)), "")
            after = next((x for x in after_lines if not role_line.search(x) and not x.startswith("..") and not counsel.search(x)), "")
            if b_names:
                before = summarise(b_names)
                d["first_parties"] = b_names
            if a_names:
                after = a_names[0] + (" & Ors." if len(a_names) > 1 else "")

            b_role = ""
            for x in before_lines:
                m = role_kw.search(x)
                if m:
                    b_role = m.group(1).title()

            a_role = ""
            for x in after_lines:
                m = role_kw.search(x)
                if m:
                    a_role = m.group(1).title()

            b_clean, a_clean = clean(before), clean(after)
            if b_clean and a_clean:
                d["parties"] = f"{b_clean} v. {a_clean}"
                if b_role:
                    d["before_role"] = b_role
                if a_role:
                    d["after_role"] = a_role

                is_criminal_app = bool(re.search(r"\b(APPLICATION|APPEAL|PETITION|REVISION|BAIL)\b", d.get("case_no", ""), re.I))
                is_state_appellant = bool(re.search(r"\bState\b", b_clean, re.I))

                if is_state_appellant:
                    d["prosecution"] = b_clean
                    d["accused"] = a_clean
                    d["complainant"] = "Complainant / First Informant (Victim)"
                elif b_role in ("Applicant", "Appellant", "Petitioner", "Accused") or is_criminal_app:
                    lbl = b_role or "Accused"
                    d["accused"] = f"{b_clean} ({lbl})"
                    d["prosecution"] = a_clean
                    if "& Anr" in a_clean or "& Ors" in a_clean or "Respondent No.2" in head:
                        d["complainant"] = "Respondent No. 2 / Complainant (First Informant)"
                    else:
                        d["complainant"] = "Complainant / First Informant (Victim)"
            break

    if "parties" not in d:          # one-line form: "State of Maharashtra vs. Suresh Sharma"
        for l in lines[:25]:
            m = re.fullmatch(r"(.{3,80}?)\s+(?:Versus|Vs\.?|V/s\.?|v\.)\s+(.{3,80})", l, re.I)
            if m and not re.search(r"\b(?:SCC|AIR|held|case of)\b", l):
                p1, p2 = m.group(1).strip(" ."), m.group(2).strip(" .")
                d["parties"] = f"{p1} v. {p2}"
                if re.search(r"\bState\b", p1, re.I):
                    d["prosecution"] = p1
                    d["accused"] = p2
                    d["complainant"] = "Complainant / First Informant (Victim)"
                elif re.search(r"\bState\b", p2, re.I):
                    d["accused"] = p1
                    d["prosecution"] = p2
                    d["complainant"] = "Respondent No. 2 / Complainant (Victim)"
                break
    m = re.search(r"CORAM\s*:\s*([^\n]+)", head, re.I)
    if m:
        d["judge"] = re.sub(r"\s+", " ", m.group(1)).strip()
    m = re.search(r"(?:DATE|DATED|PRONOUNCED ON|DECIDED ON)\s*:?\s*([^\n]+)", head, re.I)
    if m:
        d["date"] = re.sub(r"\s+", " ", m.group(1)).strip()
    return d


# ---------------------------------------------------------------- statutes
ACTS = [  # (short name, how the Act is written in orders)
    ("BNS", r"Bharatiya\s+Nyaya\s+Sanhita|\bBNS\b"),
    ("BNSS", r"Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita|\bBNSS\b"),
    ("BSA", r"Bharatiya\s+Sakshya\s+Adhiniyam|\bBSA\b"),
    ("IPC", r"Indian\s+Penal\s+Code|Penal\s+Code|\bI\.?\s?P\.?\s?C\b"),
    ("CrPC", r"Code\s+of\s+Criminal\s+Procedure|\bCr\.?\s?P\.?\s?C\b"),
    ("POCSO Act", r"Protection\s+of\s+Children\s+from\s+Sexual\s+Offences|POCSO"),
    ("SC/ST Act", r"\bSC\s+(?:and|&)\s+ST\b|Scheduled\s+Castes\s+and\s+(?:the\s+)?Scheduled\s+Tribes"),
    ("Child Marriage Act", r"Child\s+Marriage\s+(?:Restraint|Prohibition)\s+Act"),
    ("NDPS Act", r"Narcotic\s+Drugs|\bNDPS\b"),
    ("Arms Act", r"Arms\s+Act"),
    ("Dowry Prohibition Act", r"Dowry\s+Prohibition"),
    ("IT Act", r"Information\s+Technology\s+Act"),
]
NUM = r"\d+[A-Z]{0,2}(?:\s*\(\s*[0-9a-z]+\s*\))*"
SEC_LIST = re.compile(rf"\b(?:Sections?|Secs?\.?|u/ss?\.?|S\.)\s*({NUM}(?:\s*(?:,|and|&|/)\s*(?:and\s+)?{NUM})*)", re.I)


def _act_after(text: str, end: int, window: int = 120):
    """The Act named right after a section list ('Sections 4, 6 of the POCSO Act')."""
    seg = text[end:end + window]
    seg = re.split(r"\bSections?\b", seg, maxsplit=1, flags=re.I)[0]   # stop at the next section list
    best = None
    for name, pat in ACTS:
        m = re.search(pat, seg, re.I)
        if m and (best is None or m.start() < best[1]):
            best = (name, m.start())
    if best:
        return best[0]
    if re.match(r"\s*of\s+the\s+(?:said\s+)?Act\b", seg, re.I):
        return "the Act (named earlier)"
    m = re.match(r"\s*(?:of|under)\s+(?:the\s+)?([A-Z][\w&().' ]{2,80}?\bAct\b(?:,?\s*\d{4})?)", seg)   # any other named Act
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return None


def _subclauses(seg: str, end: int, nums: list) -> list:
    """'Sections 3(1)(q), (r) and (s)' -> 3(1)(q), 3(1)(r), 3(1)(s): bare '(r)' after a section continues its parent."""
    m = re.match(r"((?:\s*(?:,|and|&)\s*\(\s*[0-9a-z]{1,4}\s*\))+)", seg[end:end + 60])
    if not m or not nums or "(" not in nums[-1]:
        return nums
    parent = nums[-1][:nums[-1].rfind("(")]
    return nums + [f"{parent}({s})" for s in re.findall(r"\(\s*([0-9a-z]{1,4})\s*\)", m.group(1))]


def _nums(s: str):
    return [re.sub(r"\s+", "", n) for n in re.findall(NUM, s)]


def charges(raw: str) -> list:
    """Registered offences: the sentence 'offences punishable under Sections ... of <Act>; Sections ... of <Act> ...'.
    Returns [(act, [sections]), ...]; empty if the order does not state them in that form."""
    t = flat(raw)
    m = re.search(r"offen[cs]es?\s+(?:punishable\s+)?under\s+(.{0,700}?)(?:\.\s+\d+\.\s|\.\s+[A-Z][a-z])", t, re.I)
    if not m:
        return []
    seg, out = m.group(1), []
    for sm in SEC_LIST.finditer(seg):
        nums = _subclauses(seg, sm.end(), _nums(sm.group(1)))
        tail = re.match(r"((?:\s*(?:,|and|&)\s*\(\s*[0-9a-z]{1,4}\s*\))+)", seg[sm.end():sm.end() + 60])
        out.append((_act_after(seg, sm.end() + (tail.end() if tail else 0)) or "unclear", nums))
    return out


def cited_sections(raw: str) -> list:
    """Every section mention with its Act (or 'unclear') and context. Mentions are NOT charges."""
    t, out = flat(raw), []
    for m in SEC_LIST.finditer(t):
        out.append({"act": _act_after(t, m.end()) or "unclear", "sections": _nums(m.group(1)),
                    "context": t[max(0, m.start() - 70): m.end() + 70]})
    return out


# ---------------------------------------------------------------- IPC -> BNS (common offences)
# Source: the IPC-BNS correspondence published with the BNS (2023). Check against the official table before relying on it.
IPC_TO_BNS = {
    "34": "3(5)", "120B": "61(2)", "147": "191(2)", "148": "191(3)", "149": "190", "201": "238",
    "302": "103(1)", "304": "105", "304A": "106(1)", "304B": "80", "306": "108", "307": "109",
    "323": "115(2)", "324": "118(1)", "325": "117(2)", "326": "118(2)", "341": "126(2)", "342": "127(2)",
    "354": "74", "354A": "75", "354D": "78", "363": "137(2)", "364": "140(1)", "366": "87",
    "376": "64", "376D": "70(1)", "379": "303(2)", "380": "305", "392": "309(4)", "395": "310(2)",
    "406": "316(2)", "409": "316(5)", "420": "318(4)", "447": "329(3)", "448": "329(4)",
    "465": "336(2)", "467": "338", "468": "336(3)", "471": "340(2)", "498A": "85", "504": "352",
    "506": "351(2)", "509": "79", "511": "62",
}


def _base(sec: str) -> str:
    return re.match(r"\d+[A-Z]*", sec.replace(" ", "")).group(0)


def check_against_charges(ipc_section: str, charge_list: list) -> str:
    """'matches registered charge BNS 64(1)(2)' / 'not among registered charges' / 'no charges stated'."""
    n = re.sub(r"(?i)^section\s*", "", ipc_section).strip()
    bns = IPC_TO_BNS.get(_base(n))
    if not charge_list:
        return "registered charges not stated in a recognised form"
    for act, secs in charge_list:
        for s in secs:
            if act == "IPC" and _base(s) == _base(n):
                return f"matches registered charge IPC {s}"
            if act == "BNS" and bns and _base(s) == _base(bns):
                return f"matches registered charge BNS {s} (IPC {n} ≈ BNS {bns})"
    return "not among the registered charges" + (f" (IPC {n} ≈ BNS {bns})" if bns else "")


# ---------------------------------------------------------------- which sentences are this case's facts
PRECEDENT_RE = re.compile(r"\bSCC\b|\bAIR\b|\bSCR\b|\bOnLine\b|\bdecided\s+on\b|\bin\s+the\s+case\s+of\b|"
                          r"\bv\.\s|\bvs\.?\s|\bversus\b|Hon'ble\s+Supreme\s+Court|\bApex\s+Court\b|\bheld\s+thus\b", re.I)
CONDITION_RE = re.compile(r"^\s*\(?(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\)\s*|^\s*(?:Applicant|Accused)\s+shall\b", re.I)
LAW_TALK_RE = re.compile(r"\bpresum|\bprovides?\s+that\b|\bstates?\s+that\s+where\b|\blegislat|\bstatutory\b", re.I)
QUOTE_RE = re.compile(r"[“\"](.{40,}?)[”\"]", re.S)


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


class FactFinder:
    """Classifies a sentence of one document as 'fact' | 'precedent' | 'condition' | 'law'.
    A sentence counts as 'precedent' if it sits inside a quotation that is introduced by a case citation
    (e.g. '... in the case of X Vs. State ... reproduced below: “...”'), even when the sentence itself names no case.
    Quotes of the court's own earlier orders in the same case are kept as facts."""

    def __init__(self, raw: str):
        t = flat(raw)
        self.norm = _alnum(t)
        # map positions in the flat text to positions in the alnum-only text
        idx, k = [], 0
        for ch in t:
            idx.append(k)
            if re.match(r"[A-Za-z0-9]", ch):
                k += 1
        # top-level quotations, tracking nesting of curly quotes; straight quotes toggle only at depth 0.
        # Text after the document's last closing quote is the court's own (handles an unclosed outer quote).
        last_close = max(t.rfind("”"), t.rfind('"'))
        self.spans, depth, start, straight = [], 0, None, False
        for i, ch in enumerate(t):
            if i > last_close:
                break
            if ch == "“" and not straight:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "”" and depth > 0:
                depth -= 1
                if depth == 0:
                    self._add(t, idx, start, i)
            elif ch == '"' and depth == 0:
                if not straight:
                    straight, start = True, i
                else:
                    straight = False
                    self._add(t, idx, start, i)
        if depth > 0 and start is not None:
            self._add(t, idx, start, last_close)

    def _add(self, t, idx, a, b):
        if b - a > 40 and PRECEDENT_RE.search(t[max(0, a - 400): a]):
            self.spans.append((idx[a], idx[b]))

    def position(self, sentence: str) -> int:
        """Where the middle of the sentence sits in the document (-1 if not found). Tries several 40-character windows, so a sentence
        whose section numbers were hidden by the model's masking is still found."""
        key = _alnum(sentence)
        if len(key) < 20:
            return -1
        w = min(40, len(key))
        for off in sorted({0, (len(key) - w) // 2, len(key) - w, len(key) // 4, (3 * len(key)) // 4 - w // 2}):
            off = max(0, min(off, len(key) - w))
            pos = self.norm.find(key[off:off + w])
            if pos >= 0:
                return pos - off + len(key) // 2      # middle of the sentence (robust at quote boundaries)
        return -1

    def kind(self, sentence: str) -> str:
        pos = self.position(sentence)
        if pos >= 0 and any(a <= pos <= b for a, b in self.spans):
            return "precedent"
        return sentence_kind(sentence)


def sentence_kind(s: str) -> str:
    """Kind of a sentence on its own: 'precedent' | 'condition' | 'law' | 'fact'."""
    if PRECEDENT_RE.search(s):
        return "precedent"
    if CONDITION_RE.search(s):
        return "condition"
    if LAW_TALK_RE.search(s):
        return "law"
    return "fact"


DATE = r"\d{1,2}(?:st|nd|rd|th)?[./\s-]\s*(?:\d{1,2}|[A-Z][a-z]+)[./\s-]\s*\d{2,4}"
HEARING_RE = re.compile(rf"\bOn\s+{DATE},?\s+after\s+hearing|\bheard\s+on\s+{DATE}|(?<=\d\.\s)Today\b|"
                        rf"\bFIR\b[^.]{{0,60}}?\blodged\s+on\s+{DATE}|\bStand\s+over\w*\s+to\s+[^.;]+", re.I)


def dated_paragraphs(raw: str, max_chars: int = 450, finder=None) -> list:
    """Snippets that record this case's FIR, hearings and adjournments, in document order
    (skipping any that sit inside a quoted precedent when a FactFinder is given)."""
    t = flat(raw)
    out = []
    for m in HEARING_RE.finditer(t):
        snip = t[m.start(): m.start() + max_chars]
        if finder is None or finder.kind(snip[:120]) != "precedent":
            out.append(snip)
    return out


def disposition(raw: str, max_chars: int = 3500) -> str:
    """The operative part: from the last 'Hence' / 'For the reasons' / 'In view of' to the end."""
    t = flat(raw)
    t = re.split(r"\bDigitally\s+signed\b", t)[0]                       # drop the e-signature stamp
    depth, inside, opened = 0, [], 0                                      # quoted text (cited judgments) is not this order
    for k, ch in enumerate(t):
        if ch == "\u201c":
            depth, opened = depth + 1, (k if depth == 0 else opened)
        elif ch == "\u201d":
            depth = max(depth - 1, 0)
        if depth and k - opened > 4000:                                    # a quotation closed with a plain " : stop counting it
            depth = 0
        inside.append(depth > 0)
    starts = [m.start() for m in re.finditer(r"\b(?:Hence|For the (?:above |aforesaid )?reasons|In view of the above|In view thereof|"
                                             r"In the result|Accordingly|granted bail subject to|subject to (?:the )?following conditions)\b", t, re.I)
              if not inside[m.start()]]
    seg = t[starts[-1]:] if starts else t[-max_chars:]
    if len(seg) > max_chars:                                              # the end of an order is what matters: keep the tail
        seg = "\u2026 " + seg[-max_chars:]
    return seg
