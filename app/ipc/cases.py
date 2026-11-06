"""cases.py · saved judgments and chat history, on this computer only.

  cases/<id>.json        a judgment: name, file name, briefings per perspective
  cases/<id>.txt         its text (a saved judgment reopens without the original file; the analysis comes from the result cache)
  cases/chats/<id>.json  one conversation (like a chat in any LLM app): title, the visible messages, the model's memory
A judgment can have many conversations. The case id is a hash of the judgment text, so the same file maps to the same judgment.
"""
from __future__ import annotations

import hashlib, json, os, re, secrets, threading, time
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIR = os.environ.get("IPC_CASES_DIR", os.path.join(ROOT, "cases"))
CHATS = os.path.join(DIR, "chats")
_LOCK = threading.Lock()


def case_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def _path(cid: str, ext: str) -> str:
    if not cid.isalnum():
        raise ValueError("bad case id")
    return os.path.join(DIR, f"{cid}.{ext}")


def load(cid: str) -> Optional[dict]:
    try:
        with open(_path(cid, "json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def text(cid: str) -> Optional[str]:
    try:
        with open(_path(cid, "txt"), encoding="utf-8") as f:
            return f.read()
    except (OSError, ValueError):
        return None


def save(c: dict) -> None:
    """Atomic write (a crash mid-write never leaves a half file)."""
    with _LOCK:
        os.makedirs(DIR, exist_ok=True)
        c["updated"] = time.time()
        _atomic(_path(c["id"], "json"), c)


def title_from_text(doc: str) -> str:
    """A name for pasted text: the 'X versus Y' line near the top, else the first line."""
    lines = [l.strip() for l in doc.splitlines() if l.strip()][:80]
    party = lambda l: not re.match(r"^[.\s]*\.\.", l)          # skip ".. Applicant" / "..Respondents" labels
    for i, l in enumerate(lines):
        if re.fullmatch(r"(?:versus|vs\.?|v/s|v\.?)", l, re.I):        # "Versus" on its own line between the parties
            before = next((x for x in reversed(lines[:i]) if party(x)), "")
            after = next((x for x in lines[i + 1:] if party(x)), "")
            if before and after:
                return f"{before} v. {after}"[:80]
    for l in lines:
        if (re.search(r"\b(?:versus|vs\.?|v/s)\s", l, re.I) or re.search(r"\s[vV]\.?\s", l)) and 8 < len(l) < 160:
            return l[:80]
    return ""


def open_case(doc: str, name: str, hide_cited: bool) -> dict:
    """The saved case for this judgment text, created on first open."""
    cid = case_id(doc)
    c = load(cid)
    if c is None:
        os.makedirs(DIR, exist_ok=True)
        with open(_path(cid, "txt"), "w", encoding="utf-8") as f:
            f.write(doc)
        file = name if name and name != "pasted text" else ""
        title = title_from_text(doc)
        first = next((l.strip() for l in doc.splitlines() if l.strip()), "pasted text")[:60]
        name = title or file or "Pasted · " + first             # "A v. B" reads better than a hash-like file name
        c = {"id": cid, "name": name, "file": file, "created": time.time(), "hide_cited": hide_cited,
             "explain": {}}
    elif name and name != "pasted text" and not c.get("file"):
        c["file"] = name
    c["opened"] = time.time()
    save(c)
    return c


def _atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


# ───────────────────────────── conversations ─────────────────────────────
def _chat_path(chat_id: str) -> str:
    if not chat_id.isalnum():
        raise ValueError("bad chat id")
    return os.path.join(CHATS, chat_id + ".json")


def load_chat(chat_id: str) -> Optional[dict]:
    try:
        with open(_chat_path(chat_id), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_chat(ch: dict, touch: bool = True) -> None:
    with _LOCK:
        if touch:
            ch["updated"] = time.time()
        _atomic(_chat_path(ch["id"]), ch)


def new_chat(case_id_: str) -> dict:
    now = time.time()
    return {"id": secrets.token_hex(8), "case_id": case_id_, "title": "", "created": now, "updated": now,
            "messages": [], "turns": [], "history": []}


def chats_for(case_id_: str) -> List[dict]:
    return [c for c in all_chats() if c.get("case_id") == case_id_]


def all_chats() -> List[dict]:
    _migrate()
    out = []
    try:
        names = [n for n in os.listdir(CHATS) if n.endswith(".json")]
    except OSError:
        return out
    for n in names:
        ch = load_chat(n[:-5])
        if ch:
            out.append(ch)
    return sorted(out, key=lambda c: c.get("updated") or 0, reverse=True)


def delete_chat(chat_id: str) -> None:
    with _LOCK:
        try:
            os.remove(_chat_path(chat_id))
        except OSError:
            pass


def title_from_question(q: str) -> str:
    q = re.sub(r"\s+", " ", q).strip().rstrip("?.!")
    return (q[:1].upper() + q[1:])[:60] + ("…" if len(q) > 60 else "") if q else "New chat"


def _migrate() -> None:
    """Earlier builds kept one chat inside the judgment file: move it into its own conversation."""
    try:
        names = [n for n in os.listdir(DIR) if n.endswith(".json")]
    except OSError:
        return
    for n in names:
        c = load(n[:-5])
        if c and ("chat" in c or "turns" in c):
            msgs = c.pop("chat", []) or []
            turns, hist = c.pop("turns", []) or [], c.pop("history", []) or []
            if msgs:
                ch = new_chat(c["id"])
                ch.update(messages=msgs, turns=turns, history=hist, title=title_from_question(msgs[0].get("q", "")),
                          created=msgs[0].get("t", time.time()))
                ch["updated"] = msgs[-1].get("t", time.time())
                save_chat(ch, touch=False)
            save(c)


def summary(c: dict) -> dict:
    return {"id": c["id"], "name": c.get("name") or "judgment", "file": c.get("file", ""), "updated": c.get("updated"), "opened": c.get("opened"),
            "briefings": sorted(c.get("explain", {}))}


def list_cases() -> List[dict]:
    out = []
    try:
        names = [n for n in os.listdir(DIR) if n.endswith(".json")]
    except OSError:
        return out
    for n in names:
        c = load(n[:-5])
        if c:
            out.append(summary(c))
    return sorted(out, key=lambda s: s.get("opened") or s.get("updated") or 0, reverse=True)


def delete(cid: str) -> None:
    for ch in chats_for(cid):
        delete_chat(ch["id"])
    with _LOCK:
        for ext in ("json", "txt"):
            try:
                os.remove(_path(cid, ext))
            except OSError:
                pass
