"""silver.py · silver IPC labels from a judgment's raw text (the sections it cites with an IPC context)."""
import re

NUM = r"\d{1,3}(?!\d)(?:\s*-?\s*[A-Z]{1,2}\b)?(?:\s*\(\s*[0-9a-z]{1,3}\s*\))*"
JOIN = r"\s*(?:,|/|&|\band\b|\bor\b|r/w|read\s+with|alongwith|along\s+with|with)\s*(?:(?:[Ss]ections?(?:\(s\))?|[Ss]ecs?\.?|S\.)\s*)?"
SEC = (r"(?:\b[Ss]ections?(?:\s*\(s\))?|\b[Ss]ecs?\.?|(?<![A-Za-z.])S\.|\b[Uu]/[Ss]{1,2}\.?|\b[Ss]{2}\.|\b[Uu]nder\s+[Ss]ection)")
CITE = re.compile(rf"{SEC}\s*{NUM}(?:{JOIN}{NUM})*", re.I)
BARE = re.compile(rf"(?<![\w/.]){NUM}(?:{JOIN}{NUM})*(?=\s*(?:of\s+(?:the\s+)?)?(?:Indian\s+Penal\s+Code|I\.\s?P\.\s?C|IPC)\b)", re.I)
IPC = re.compile(r"Indian\s+Penal\s+Code|I\.\s?P\.\s?C|\bIPC\b|Penal\s+Code", re.I)
OTHER = re.compile(r"Cr\.?\s?P\.?\s?C|Criminal\s+Procedure|Evidence\s+Act|N\.?D\.?P\.?S|Arms\s+Act|POCSO|Prevention\s+of|Constitution|"
                   r"Article|Act,?\s+\d{4}|\bBNSS?\b|Bharatiya|Negotiable|Customs|Income|Companies|Dowry", re.I)
NUMS = re.compile(r"(\d{1,3})(?!\d)\s*-?\s*([A-Z]{1,2})?\b")


def silver_labels(text: str, labels) -> set:
    t = re.sub(r"\s+", " ", text)
    valid = set(labels)
    out = set()
    spans = [(m.start(), m.end()) for m in CITE.finditer(t)] + [(m.start(), m.end()) for m in BARE.finditer(t)]
    for a, b in spans:
        after, before = t[b:b + 45], t[max(0, a - 15):a]
        ctx = t[a:b] + after
        if not IPC.search(ctx):
            continue
        cut = OTHER.search(after)                    # 'Section 302 IPC and 27 Evidence Act': stop at the other Act
        ipc = IPC.search(after)
        if cut and ipc and cut.start() < ipc.start():
            continue
        for n, suf in NUMS.findall(t[a:b]):
            for cand in (f"Section {n}{(suf or '').upper()}", f"Section {n}"):
                if cand in valid:
                    out.add(cand)
                    break
    return out
