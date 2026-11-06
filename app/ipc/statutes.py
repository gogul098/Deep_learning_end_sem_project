"""statutes.py · Official IndiaCode retrieval & statutory signal for the LLM.

Fetches authoritative headings, definitions, and BNS correspondences from:
  BASE = "https://indiacode.ecourtsindia.com/api/v1"
Caches results locally in `legal_kb/statutes_cache.json` for 100% offline availability and zero-latency inference.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import requests

BASE = "https://indiacode.ecourtsindia.com/api/v1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE_FILE = os.path.join(ROOT, "legal_kb", "statutes_cache.json")

# In-memory cache loaded on import
_CACHE: Dict[str, dict] = {}


def _load_cache():
    global _CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _CACHE = json.load(f)
        except Exception as e:
            _CACHE = {}


def _save_cache():
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_CACHE, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


_load_cache()

# Built-in emergency offline fallback headings (used only if neither IndiaCode nor checkpoint has data)
HEADINGS = {
    "21": "Public servant", "53": "Punishments", "96": "Things done in private defence",
    "100": "When the right of private defence of the body extends to causing death",
    "102": "Commencement and continuance of the right of private defence of the body", "107": "Abetment of a thing",
    "120": "Concealing design to commit offence punishable with imprisonment", "120A": "Definition of criminal conspiracy",
    "121": "Waging, or attempting to wage war, or abetting waging of war, against the Government of India",
    "121A": "Conspiracy to commit offences punishable by section 121", "141": "Unlawful assembly",
    "143": "Punishment for unlawful assembly",
    "153A": "Promoting enmity between different groups on grounds of religion, race, place of birth, residence, language, etc.",
    "186": "Obstructing public servant in discharge of public functions", "193": "Punishment for false evidence",
    "212": "Harbouring offender",
    "218": "Public servant framing incorrect record or writing with intent to save person from punishment or property from forfeiture",
    "228": "Intentional insult or interruption to a public servant sitting in any stage of a judicial proceeding",
    "294": "Obscene acts and songs", "299": "Culpable homicide", "300": "Murder", "308": "Attempt to commit culpable homicide",
    "309": "Attempt to commit suicide", "337": "Causing hurt by act endangering life or personal safety of others",
    "338": "Causing grievous hurt by act endangering life or personal safety of others",
    "353": "Assault or criminal force to deter public servant from discharge of his duty", "364A": "Kidnapping for ransom, etc.",
    "365": "Kidnapping or abducting with intent secretly and wrongfully to confine person", "375": "Rape",
    "376A": "Punishment for causing death or resulting in persistent vegetative state of victim", "376D": "Gang rape",
    "405": "Criminal breach of trust", "415": "Cheating", "417": "Punishment for cheating",
    "418": "Cheating with knowledge that wrongful loss may thereby be caused to a person whose interest the offender is bound to protect",
    "427": "Mischief causing damage to the amount of fifty rupees",
    "435": "Mischief by fire or explosive substance with intent to cause damage",
    "449": "House-trespass in order to commit offence punishable with death",
    "450": "House-trespass in order to commit offence punishable with imprisonment for life",
    "457": "Lurking house-trespass or house-breaking by night in order to commit offence punishable with imprisonment",
    "466": "Forgery of record of Court or of public register, etc.", "477A": "Falsification of accounts",
    "494": "Marrying again during lifetime of husband or wife",
    "498": "Enticing or taking away or detaining with criminal intent a married woman", "499": "Defamation",
    "500": "Punishment for defamation",
    "511": "Punishment for attempting to commit offences punishable with imprisonment for life or other imprisonment",
}


_NET_DOWN = False


def get_section(act: str, section: str, timeout: int = 5) -> dict:
    """Retrieves authoritative statute details from IndiaCode API with persistent local caching.

    Returns:
        dict with keys: act, section, heading, text, offence, bns, bns_relation
    """
    act_clean = act.lower().strip()
    sec_clean = re.sub(r"(?i)\bsections?\b", "", section).strip().upper()
    key = f"{act_clean}:{sec_clean}"

    # 1. Check local cache first
    if key in _CACHE and _CACHE[key].get("heading"):
        return _CACHE[key]

    # 2. Fetch from IndiaCode API (skipped for the rest of the session after one failure: the app is meant to run offline,
    #    and a blocked network would otherwise cost a timeout on every uncached section)
    global _NET_DOWN
    try:
        if _NET_DOWN:
            raise OSError("offline")
        url = f"{BASE}/{act_clean}/section/{sec_clean}"
        r = requests.get(url, timeout=min(timeout, 3))
        if r.status_code == 200:
            data = r.json()
            bns_sec = None
            bns_rel = None
            for c in data.get("corresponds_to", []):
                if c.get("act") == "bns":
                    bns_sec = c.get("section")
                    bns_rel = c.get("relation")
                    break

            res = {
                "act": data.get("act", {}).get("short_title", act.upper()),
                "section": data.get("section", {}).get("number", sec_clean),
                "heading": data.get("section", {}).get("heading", ""),
                "text": data.get("section", {}).get("text", ""),
                "bns": bns_sec,
                "bns_relation": bns_rel,
                "offence": data.get("classification", [{}])[0] if data.get("classification") else None,
            }
            _CACHE[key] = res
            _save_cache()
            return res
    except Exception:
        _NET_DOWN = True

    # 3. Offline fallback synthesis if not yet in cache
    fallback_heading = HEADINGS.get(sec_clean, "")
    res = {
        "act": f"The Indian Penal Code, 1860" if act_clean == "ipc" else act_clean.upper(),
        "section": sec_clean,
        "heading": fallback_heading,
        "text": "",
        "bns": None,
        "bns_relation": None,
        "offence": None,
    }
    return res


def one_line(section: str, desc: Optional[dict] = None) -> str:
    """'Section 302' -> authoritative statute heading.

    Priority:
      1. Official IndiaCode heading (from API / local cache)
      2. Checkpoint description (if non-empty)
      3. Built-in HEADINGS fallback table
    """
    sec_num = re.sub(r"(?i)\bsections?\b", "", section).strip().upper()

    # 1. Primary: Official IndiaCode
    info = get_section("ipc", sec_num)
    if info.get("heading"):
        return info["heading"]

    # 2. Checkpoint description
    if desc:
        d = (desc.get(section) or "").strip()
        if d:
            return d

    # 3. Built-in heading fallback
    return HEADINGS.get(sec_num, "statute text not available in the local knowledge base")


def format_statute_signal(sections: List[str], desc: Optional[dict] = None, max_text_len: int = 350) -> str:
    """Formats an authoritative statutory signal block containing the exact statutory definitions and elements.

    Passed to the LLM to verify whether document facts fulfill the legal ingredients.
    """
    if not sections:
        return "(none)"

    lines = []
    for s in sections:
        sec_num = re.sub(r"(?i)\bsections?\b", "", s).strip().upper()
        data = get_section("ipc", sec_num)

        heading = data.get("heading") or one_line(s, desc)
        bns = f" | Modern Equivalent: BNS Section {data['bns']}" if data.get("bns") else ""
        text = data.get("text", "").strip()

        if text:
            # Clean and truncate if excessively verbose
            text_clean = " ".join(text.split())
            if len(text_clean) > max_text_len:
                text_clean = text_clean[:max_text_len] + "..."
            lines.append(f'• IPC Section {sec_num} ("{heading}"{bns}):\n  Statute Ingredients: "{text_clean}"')
        else:
            lines.append(f'• IPC Section {sec_num} ("{heading}"{bns})')

    return "\n".join(lines) if lines else "(none)"
