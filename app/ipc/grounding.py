"""grounding.py · checks a finished chat answer against the exact text the model was given (CPU, no model, ~1 ms).

The Bayesian router decides BEFORE generation and cannot see reasoning errors made with the right passages in hand.
This check runs AFTER generation and catches the error types seen in the stress tests:
  numbers      a figure, date or amount that appears nowhere in the context or the question   ("shortage of 19 rounds")
  units        the right quantity with the wrong unit                                         ("three and a half months" vs "weeks")
  sections     a BNS number that does not exist in the reference table                       ("BNS 337")
  precedent    a claim about THIS case backed only by sentences quoted from other judgments   ("granted bail on parity")
The engine regenerates once with the problems named (see Engine.chat); remaining problems are shown as a warning.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+(?:[./]\d+)*)(?![\w])")
CITE_BLOCK = re.compile(r"\[[^\]]*\bS\d+[^\]]*\]|\(S\d+[^)]*\)")
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "a": 1, "an": 1}
QTY = re.compile(r"\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|an?)"
                 r"(\s+and\s+(?:a\s+)?half)?\s+(day|week|month|year)s?\b", re.I)
THIS_CASE = re.compile(r"\b(the applicant|the petitioners?|the accused|the appellant|present case|this case|this order|"
                       r"the court (?:granted|held|found|directed|allowed|ordered))\b", re.I)
ABOUT_OTHER = re.compile(r"\b(cited|precedent|in the case of|referred|relied|similar|another case|other cases?|co-accused in)\b", re.I)


PAGE_JUNK = re.compile(r"\bPage\s+\d{1,3}\s+of\s+\d{1,3}\b|\b\d{1,3}\s+of\s+\d{1,3}\b(?!\s*\d)(?:\s+\S+\.(?:odt|docx?|pdf)\b)?|\S+\.(?:odt|docx?)\b", re.I)
STOPW = set("the a an of in on at to for from by with and or was were is are be been as that this which its their his her".split())


def clean(text: str) -> str:
    """Page footers and file names split phrases in PDF text ('three and half 2 of 9 APL.1266.2022.odt weeks')."""
    return re.sub(r"\s+", " ", PAGE_JUNK.sub(" ", text))


def _counted(body: str, m) -> str:
    """The word a number counts: '19 rounds' -> 'roun'; '' when none follows."""
    nxt = re.match(r"\s+([A-Za-z]+)", body[m.end():m.end() + 30])
    w = nxt.group(1).lower() if nxt else ""
    return "" if (not w or w in STOPW) else w


def _norm(n: str) -> str:
    n = n.replace(",", "")
    parts = re.split(r"([./])", n)
    return "".join(p.lstrip("0") or "0" if p not in "./" else p for p in parts)


def _numbers(text: str) -> set:
    return {_norm(m.group(1)) for m in NUM.finditer(text)}


def _quantities(text: str) -> list:
    out = []
    for m in QTY.finditer(text):
        v = m.group(1).lower()
        val = float(WORDNUM.get(v, v)) if v in WORDNUM or re.fullmatch(r"[\d.]+", v) else None
        if val is None:
            continue
        if m.group(2):
            val += 0.5
        out.append((val, m.group(3).lower(), m.group(0)))
    return out


def check(answer: str, context: str, question: str = "", sentences: Optional[List[str]] = None,
          tag_of: Optional[Callable[[str], List[str]]] = None, bns_values: Optional[set] = None,
          ipc_to_bns: Optional[dict] = None) -> List[str]:
    """Problems found in `answer`; [] when it is consistent with `context`. `tag_of(sentence)` returns the tags of a
    judgment sentence (['fact'], ['precedent'], ['precedent', 'fact'] ...)."""
    issues: List[str] = []
    context = clean(context)
    body = CITE_BLOCK.sub(" ", answer)                                  # citation labels are not claims
    known = _numbers(context) | _numbers(question)
    ctx_words = {}                                                       # number -> words found right next to it in the context
    for m in NUM.finditer(context):
        near = re.findall(r"[A-Za-z]+", context[max(0, m.start() - 40):m.end() + 40].lower())
        ctx_words.setdefault(_norm(m.group(1)), set()).update(w[:4] for w in near)

    # 1. numbers that are not in the context
    sec_nums = {m.group(2) for m in re.finditer(r"\b(BNS|IPC|Sections?|Sec\.?|S\.)\s*(\d+[A-Z]?)", body, re.I)}
    bad = []
    for m in NUM.finditer(body):
        raw = m.group(1)
        line_start = body.rfind("\n", 0, m.start()) + 1
        if re.fullmatch(r"\s*(?:[-*•]\s*)?", body[line_start:m.start()]) and body[m.end():m.end() + 1] in ".)":
            continue                                                     # list numbering "1." "2)"
        if raw in sec_nums:
            continue
        n = _norm(raw)
        if n in _numbers(question):
            continue
        what = _counted(body, m)
        if n in known and (not what or what[:4] in ctx_words.get(n, set())):
            continue                                                     # same figure, counting the same thing
        bad.append(f"{raw} {what}".strip() if n in known else raw)
    if bad:
        issues.append("these figures are not in the document as stated: " + ", ".join(dict.fromkeys(bad))
                      + ". Copy numbers exactly as the document gives them; never compute or infer a figure.")

    # 2. the right quantity with the wrong unit
    ctx_q = _quantities(context)
    for val, unit, text in _quantities(body):
        if any(v == val and u == unit for v, u, _ in ctx_q):
            continue
        other = [t for v, u, t in ctx_q if v == val and u != unit]
        if other:
            issues.append(f'"{text}" does not match the document, which says "{other[0]}".')

    # 3. BNS numbers that are not BNS sections
    if bns_values:
        bases = {re.sub(r"\(.*$", "", v) for v in bns_values}
        for m in re.finditer(r"\bBNS\s*(?:Section\s*)?(\d+[A-Z]?)", body, re.I):
            n = m.group(1).upper()
            if n not in bases and not re.search(rf"\bBNS\s*(?:Section\s*)?{n}\b", context, re.I):
                hint = f" (IPC {n} corresponds to BNS {ipc_to_bns[n]})" if ipc_to_bns and n in ipc_to_bns else ""
                issues.append(f"BNS {n} is not a BNS section in the reference table{hint}. Label IPC and BNS numbers correctly.")

    # 3b. an IPC section of this case labelled as BNS ("Section 307 of the BNS", "307 BNS") - a different offence in the BNS
    if ipc_to_bns:
        seen = set()
        for m in re.finditer(r"\bSec(?:tion|\.)?\s*(\d+[A-Z]?)\s*(?:of\s+(?:the\s+)?)?(?:Bharatiya\s+Nyaya\s+Sanhita|BNS)\b(?!\s*\d)"
                             r"|\b(\d+[A-Z]?)\s+(?:of\s+(?:the\s+)?)?(?:Bharatiya\s+Nyaya\s+Sanhita|BNS)\b(?!\s*\d)", body, re.I):
            n = (m.group(1) or m.group(2)).upper()
            b = ipc_to_bns.get(n)
            if n in seen or not b or b == n:
                continue
            seen.add(n)
            in_ipc = re.search(rf"(?:IPC|Indian Penal Code)[^.\n]{{0,60}}\b{n}\b|\b{n}\b[^.\n]{{0,60}}(?:IPC|Indian Penal Code)", context, re.I)
            in_bns = re.search(rf"\bBNS\s*(?:Section\s*)?{n}\b|\b{n}\s+BNS\b", context, re.I)
            if in_ipc and not in_bns:
                issues.append(f"Section {n} in this case is an IPC section, not BNS (its BNS equivalent is {b}). "
                              f"Write 'Section {n} IPC' (or 'BNS {b}' only when giving the equivalent).")

    # 4. claims about this case supported only by quotations from other judgments
    if sentences and tag_of:
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", answer):
            ids = [int(x) for x in re.findall(r"\bS(\d{1,4})\b", s)]
            ids = [i for i in ids if 0 <= i < len(sentences)]
            if not ids or not THIS_CASE.search(s) or ABOUT_OTHER.search(CITE_BLOCK.sub(" ", s)):
                continue
            tags = [set(tag_of(sentences[i])) for i in ids]
            if all(t == {"precedent"} for t in tags):
                issues.append(f'this statement is about the present case but cites only passages quoted from OTHER judgments '
                              f'({", ".join("S" + str(i) for i in ids)}): "{s.strip()[:140]}". Say what the cited case held, '
                              f"and state separately what THIS court did, from this court's own paragraphs.")
                break
    return list(dict.fromkeys(issues))


FEEDBACK = ("An automatic check of your answer against the document found these problems:\n{issues}\n\n"
            "Rewrite the complete answer to the same question. Use only numbers, dates, units and section numbers that appear "
            "in the CONTEXT. Keep what other judgments held separate from what this court decided. If the context does not "
            "support a statement, drop it or say it is not stated in the document. Do not mention this check.")
