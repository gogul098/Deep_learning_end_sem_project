"""engine.py · the one object every front end (CLI, TUI, web server) talks to.

Every long operation is a generator of plain-dict events, so a terminal or a browser can render it live:
  {"type": "stage", "name", "status": "start"|"end", "ms", "info"}     analysis / routing stages
  {"type": "result", "result": {...}}                                   sections, scores, evidence
  {"type": "route", "route", "p_fail", "blame", ...}                    the Bayesian router's decision for a chat question
  {"type": "token", "text"}                                             streamed answer text
  {"type": "done", "stats": {...}}  /  {"type": "error", "message"}
"""
from __future__ import annotations

import json, os, queue, re, subprocess, threading, time
from typing import Dict, Iterator, List, Optional

from . import prompts
from .llm import BuiltinLlama, OllamaLLM, ollama_models, ollama_unload_all
from . import grounding
from . import cases
from .statutes import one_line

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SETTINGS = os.path.join(ROOT, "settings.json")
GROUNDED_MIN = 0.30          # best passage relevance below this: answer "not stated" instead of asking a model


def gpu_stats() -> dict:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu",
                                       "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, text=True, timeout=1.5).strip()
        n, u, t, temp, util = [x.strip() for x in out.split(",")][:5]
        return {"ok": True, "name": n, "used_gb": round(float(u) / 1024, 2), "total_gb": round(float(t) / 1024, 2),
                "pct": round(float(u) / float(t) * 100), "temp": int(temp), "util": int(util)}
    except Exception:
        return {"ok": False, "name": "no NVIDIA GPU", "used_gb": 0, "total_gb": 0, "pct": 0, "temp": 0, "util": 0}


def load_settings() -> dict:
    try:
        return json.load(open(SETTINGS))
    except Exception:
        return {}


def save_settings(d: dict):
    try:
        json.dump(d, open(SETTINGS, "w"), indent=1)
    except Exception:
        pass


class Engine:
    def __init__(self, models_dir: Optional[str] = None, verifier: Optional[str] = None, model: Optional[str] = None,
                 device: Optional[str] = None, role: Optional[str] = None):
        from .pipeline import MODELS, Analyzer
        s = load_settings()
        self.settings = s
        self.verifier = verifier or s.get("verifier", "llama")
        self.model_name = model or s.get("model", "builtin")
        self.role = role or s.get("role", "accused")
        self.an = Analyzer(models_dir or MODELS, device=device, verifier=self.verifier)
        self.lock = threading.RLock()                      # one GPU job at a time
        self.res: Optional[dict] = None
        self.doc_name: Optional[str] = None
        self.turns: List[dict] = []
        self.retriever = None
        self.router = None
        self.last_route: Optional[dict] = None
        self.case: Optional[dict] = None                   # the saved judgment (ipc/cases.py): briefings on disk
        self.conv: Optional[dict] = None                   # the open conversation (chat history entry) for that judgment
        self._llm = None
        self.loaded = False

    # ───────────────────────────── lifecycle ─────────────────────────────
    def load(self, emit=lambda e: None):
        if self.an.gpu and self.verifier == "llama":
            gone = ollama_unload_all()                  # a model Ollama kept resident would leave no room for the verifier
            if gone:
                emit({"type": "info", "message": f"unloaded from Ollama to free VRAM: {', '.join(gone)}"})
        self.an.load(emit)
        self.loaded = True
        # warm the CPU retriever AFTER the GPU models finished loading, in this thread (a background load raced with the
        # Llama loader and produced empty 'meta' weights). Takes ~1-2 s.
        self._warm_retriever()
        return self

    def _warm_retriever(self):
        try:
            self._get_retriever().embed(["warm up"])
        except Exception:
            pass

    def _get_retriever(self):
        if self.retriever is None:
            from .retriever import BayesianRetriever
            self.retriever = BayesianRetriever(models_dir=self.an.models_dir)
            self.retriever.load_calibration(self.model_name)
        return self.retriever

    # ───────────────────────────── models ─────────────────────────────
    def models(self) -> List[dict]:
        out = [{"name": "builtin", "kind": "builtin", "label": "Llama-3.2-3B (built in, 4-bit)", "size_gb": 2.1, "active": self.model_name == "builtin"}]
        for m in ollama_models():
            out.append({**m, "kind": "ollama", "label": m["name"], "active": self.model_name == m["name"]})
        out.append({"name": "none", "kind": "none", "label": "No chat model (extractive answers only)", "size_gb": 0, "active": self.model_name == "none"})
        return out

    def set_model(self, name: str) -> dict:
        names = {m["name"] for m in self.models()}
        if name not in names:
            raise ValueError(f"unknown model {name!r}; available: {sorted(names)}")
        with self.lock:
            if self._llm is not None and getattr(self._llm, "kind", "") == "ollama":
                self._llm.unload()
            self._llm = None
            self.model_name = name
            if self.retriever is not None:
                self.retriever.load_calibration(name)          # the router is calibrated per chat model
            self.settings["model"] = name
            save_settings(self.settings)
        return {"model": name}

    def set_verifier(self, mode: str) -> dict:
        if mode not in ("llama", "none"):
            raise ValueError("verifier must be 'llama' or 'none'")
        with self.lock:
            self.verifier = self.an.verifier = mode
            if mode == "none" and self.model_name != "builtin":
                self.an.unload_verifier()
            self.settings["verifier"] = mode
            save_settings(self.settings)
        return {"verifier": mode}

    def set_role(self, role: str) -> dict:
        r = role.lower().strip()
        if r not in prompts.ROLES:
            raise ValueError(f"unknown role {role!r}; choose from {prompts.ROLES}")
        with self.lock:
            self.role = r
            self.settings["role"] = r
            save_settings(self.settings)
        return {"role": r}

    def chat_llm(self):
        if self.model_name == "none":
            return None
        if self._llm is None:
            self._llm = BuiltinLlama(self.an) if self.model_name == "builtin" else OllamaLLM(self.model_name)
        return self._llm

    def _vram_handover(self, for_kind: str):
        """6 GB card: the verifier and a large Ollama model do not fit together."""
        if os.environ.get("IPC_KEEP_RESIDENT") == "1":      # big / two GPUs (Kaggle T4 x2): keep both loaded
            return
        import gc, torch
        if for_kind == "ollama":
            if hasattr(self._llm, "reset_cache"):
                self._llm.reset_cache()
            self.an.unload_verifier()
            if self.an.gpu:
                gc.collect()
                torch.cuda.empty_cache()
        elif for_kind == "analyse":
            if hasattr(self._llm, "reset_cache"):
                self._llm.reset_cache()
            if self._llm is not None and getattr(self._llm, "kind", "") == "ollama":
                self._llm.unload()
            if self.an.gpu:
                ollama_unload_all()                     # ensure Ollama leaves 0 resident VRAM before verifier runs
                gc.collect()
                torch.cuda.empty_cache()

    def unload(self):
        """Release all models and KV caches from GPU VRAM."""
        with self.lock:
            if hasattr(self._llm, "reset_cache"):
                self._llm.reset_cache()
            if self._llm is not None and getattr(self._llm, "kind", "") == "ollama":
                self._llm.unload()
            self.an.unload_verifier()
            ollama_unload_all()
            if self.an.gpu:
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()

    # ───────────────────────────── analysis ─────────────────────────────
    def analyse(self, text: Optional[str] = None, path: Optional[str] = None, name: Optional[str] = None,
                hide_cited: bool = True, use_cache: bool = True, chat_id: Optional[str] = None) -> Iterator[dict]:
        q: queue.Queue = queue.Queue()
        box: dict = {}

        def work():
            try:
                with self.lock:
                    from local_model_text import read_document
                    q.put({"type": "stage", "name": "read", "status": "start"})
                    t = time.time()
                    doc = read_document(path) if path else (text or "")
                    q.put({"type": "stage", "name": "read", "status": "end", "ms": round((time.time() - t) * 1000),
                           "info": {"chars": len(doc)}})
                    if not doc.strip():
                        raise ValueError("the document is empty (a scanned PDF needs OCR first)")
                    self._vram_handover("analyse")
                    res = self.an.analyse(doc, hide_cited=hide_cited, emit=q.put, use_cache=use_cache)
                    self._set_document(res, name or (os.path.basename(path) if path else "pasted text"))
                    self._open_case(doc, self.doc_name, hide_cited, chat_id)
                    box["res"] = res
            except Exception as e:                                   # noqa: BLE001
                box["err"] = e
            finally:
                q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (ev := q.get()) is not None:
            yield ev
        if "err" in box:
            yield {"type": "error", "message": f"{type(box['err']).__name__}: {box['err']}"}
            return
        yield {"type": "result", "result": self.public(box["res"])}
        yield {"type": "done", "stats": {k: box["res"].get(k) for k in ("seconds", "doc_tokens", "sentences", "llm_tokens", "peak_vram_gb", "cached", "verifier")}}

    def _set_document(self, res: dict, name: str):
        from .retriever import ChatRouter
        self.res, self.doc_name, self.turns, self.last_route = res, name, [], None
        if hasattr(self._llm, "reset_cache"):
            self._llm.reset_cache()
        br = self._get_retriever()
        br.set_document(res["sentences_list"], self.an.last_vectors)
        self.router = ChatRouter(br)

    def public(self, res: dict) -> dict:
        s = res["sentences_list"]
        ev = {c: [{"id": f"S{i}", "text": s[i]} for i in ids] for c, ids in res["evidence_idx"].items()}
        desc = self.an.ck["statute_desc"]
        prompts.use_result(res)
        ps = prompts.prediction_status(res)
        return {"document": self.doc_name, "sections": res["sections"], "scores": res["scores"], "candidates": res["candidates"],
                "evidence": ev, "passages": res["passages"], "statutes": {c: one_line(c, desc) for c in res["candidates"]},
                "doc_tokens": res["doc_tokens"], "sentences": res["sentences"], "llm_tokens": res["llm_tokens"],
                "seconds": res["seconds"], "peak_vram_gb": res["peak_vram_gb"], "cached": res["cached"], "verifier": res["verifier"],
                "weak_prediction": ps["weak"], "weak_reason": ps["reason"],
                "case": ({"id": self.case["id"], "explain": self.case.get("explain", {})}
                         if self.case is not None and res is self.res else None),
                "conv": self.conv_public() if res is self.res else None}

    # ───────────────────────────── explanation (your template) ─────────────────────────────
    def _explain(self, role: Optional[str] = None) -> Iterator[dict]:
        if self.res is None:
            yield {"type": "error", "message": "No judgment is loaded. Open one first."}
            return
        active_role = (role or self.role or "accused").lower()
        desc = self.an.ck["statute_desc"]
        system, messages = prompts.build_explain(self.res, desc, role=active_role)
        text = []
        for ev in self._generate(system, messages, fallback=lambda: prompts.extractive_explain(self.res, desc, role=active_role), max_tokens=int(os.environ.get("IPC_EXPLAIN_TOKENS", "1200"))):
            if ev["type"] == "token":
                text.append(ev["text"])
            if ev["type"] == "done" and not prompts.is_complete("".join(text)) and self.model_name != "none":
                # some models stop early: ask once for the missing parts (the 5-part format is the contract)
                yield {"type": "token", "text": "\n\n"}
                cont = messages + [{"role": "assistant", "content": "".join(text)},
                                   {"role": "user", "content": "You stopped early. Finish the remaining numbered parts "
                                                               "(4. Bail / Custodial Assessment, 5. Actionable Roadmap & Next Steps) in 2-3 tight bullets. Do not repeat what you wrote."}]
                yield from self._generate(system, cont, fallback=lambda: "", max_tokens=400)
                return
            yield ev

    def _generate(self, system, messages, fallback, max_tokens=700) -> Iterator[dict]:
        llm = self.chat_llm()
        t0 = time.time()
        if llm is None:
            text = fallback()
            for i in range(0, len(text), 60):
                yield {"type": "token", "text": text[i:i + 60]}
            yield {"type": "done", "stats": {"model": "none", "seconds": round(time.time() - t0, 1)}}
            return
        try:
            with self.lock:
                self._vram_handover(llm.kind)
                first = None
                for piece in llm.stream(system, messages, max_tokens=max_tokens):
                    if first is None:
                        first = round(time.time() - t0, 2)
                    yield {"type": "token", "text": piece}
            yield {"type": "done", "stats": {"model": llm.name, "first_token_s": first, **getattr(llm, "stats", {})}}
        except Exception as e:                                       # noqa: BLE001
            yield {"type": "error", "message": f"{llm.name}: {type(e).__name__}: {e}"}

    # ───────────────────────────── chat with the document ─────────────────────────────
    def _chat(self, message: str, role: Optional[str] = None) -> Iterator[dict]:
        message = message.strip()
        if not message:
            return
        active_role = (role or self.role or "accused").lower()
        yield {"type": "stage", "name": "route", "status": "start"}
        t = time.time()
        if self.router is None:                              # no judgment yet: the knowledge base alone
            from .retriever import ChatRouter
            self.router = ChatRouter(self._get_retriever())
        try:
            routed = self.router.handle(message)
        except Exception as e:                               # noqa: BLE001
            yield {"type": "error", "message": f"routing failed: {type(e).__name__}: {e}"}
            return
        self.last_route = routed
        best = max((p["relevance"] for p in routed["passages"]), default=0.0)
        yield {"type": "stage", "name": "route", "status": "end", "ms": round((time.time() - t) * 1000)}
        yield {"type": "route", "route": routed["route"], "p_fail": routed["p_hallucination_high"],
               "distribution": routed["route_distribution"], "blame": routed["blame"], "bottlenecks": routed["bottlenecks"],
               "high_signals": [k for k, v in routed["evidence"].items() if v == "High"], "raw": _jsonable(routed["raw"]),
               "query": routed["query"], "caveat": routed.get("caveat", False), "best_relevance": round(best, 3),
               "calibrated": routed.get("calibrated", False), "route_note": routed.get("route_note"),
               "route_action": routed.get("route_action"), "evidence": routed.get("evidence"),
               "role": active_role,
               "passages": [{"id": p["id"], "where": p["where"], "relevance": round(p["relevance"], 3), "text": p["text"][:400]}
                            for p in routed["passages"]]}
        if routed["route"] == "Clarify":
            fallback_text = routed.get("ask_user", "Could you please clarify your question?")
            llm = self.chat_llm()
            if llm is None or os.environ.get("IPC_CLARIFY_LLM", "0") != "1":
                # prob_reas: a template question aimed at the exact gap (instant, and the reply merge knows what was asked)
                yield {"type": "token", "text": fallback_text}
                yield {"type": "done", "stats": {"route": "Clarify", "generator": "template", "reason": routed.get("reason")}}
                self.turns += [{"role": "user", "content": message}, {"role": "assistant", "content": fallback_text}]
                return
            if False:
                yield {"type": "token", "text": fallback_text}
                yield {"type": "done", "stats": {"route": "Clarify", "generator": "template"}}
                return

            system, messages = prompts.build_clarify_prompt(message, routed, self.res)
            c_text = []
            try:
                for ev in self._generate(system, messages, fallback=lambda: fallback_text, max_tokens=100):
                    if ev.get("type") == "token":
                        c_text.append(ev["text"])
                    yield ev
            except Exception:
                yield {"type": "token", "text": fallback_text}
                c_text = [fallback_text]
                yield {"type": "done", "stats": {"route": "Clarify", "generator": "fallback"}}
            if c_text:
                self.turns += [{"role": "user", "content": message}, {"role": "assistant", "content": "".join(c_text)}]
            return
        import re
        is_case_query = bool(self.res) or bool(self.turns) or bool(re.search(r"\b(why\s+(?:was|is)?\s*section|what\s+charges|which\s+sections|charges\s+against|charges\s+framed|why\s+\d+|why\s+was\s+\d+|what\s+offence|applicable\s+sections|who|what|which|how|accused|deceased|victim|witness|weapon|injury|wound|order|bail|fir|court|judge|evidence)\b", message, re.I))
        if best < GROUNDED_MIN and not is_case_query:
            yield {"type": "token", "text": "That is not stated in the judgment, and the offline library has nothing relevant to it."
                   + ("" if self.res else " (No judgment is loaded.)")}
            yield {"type": "done", "stats": {"route": routed["route"], "grounded": False}}
            return
        desc = self.an.ck["statute_desc"]
        system, messages = prompts.build_chat(message, routed, self.res, desc, self.turns, role=active_role,
                                              resolved=routed.get("resolved_query"))
        answer = []
        limit = int(os.environ.get("IPC_CHAT_TOKENS", "1000"))
        for ev in self._generate(system, messages, fallback=lambda: prompts.extractive_chat(routed), max_tokens=limit):
            if ev["type"] == "token":
                answer.append(ev["text"])
            yield ev
        final = "".join(answer)
        # grounding check after the answer (CPU, ~1 ms): figures, units, BNS labels, present-case claims backed only by precedents.
        if final and self.model_name != "none" and os.environ.get("IPC_VERIFY", "1") != "0":
            issues = self._grounding_issues(final, system, messages, message)
            if issues:
                yield {"type": "revise", "issues": issues}           # UI replaces the draft with the corrected answer
                fix = messages + [{"role": "assistant", "content": final},
                                  {"role": "user", "content": grounding.FEEDBACK.format(issues="\n".join("- " + i for i in issues))}]
                again = []
                for ev in self._generate(system, fix, fallback=lambda: "", max_tokens=limit):
                    if ev["type"] == "token":
                        again.append(ev["text"])
                    yield ev
                if again:
                    final = "".join(again)
                    left = self._grounding_issues(final, system, messages, message)
                    if left:
                        yield {"type": "verify", "issues": left}       # UI shows a small "check the citations" note
        if final:
            # the full history is kept (build_chat sends a compact form when it no longer fits the window)
            self.turns = self.turns + [messages[-1], {"role": "assistant", "content": final}]

    def _grounding_issues(self, answer: str, system: str, messages: list, question: str) -> list:
        ctx = system + "\n" + "\n".join(m["content"] for m in messages)
        sents = self.res["sentences_list"] if self.res else None
        if self.res:
            prompts.use_result(self.res)
        return grounding.check(answer, ctx, question, sents, lambda s: [t for t, _ in prompts.pieces(s)],
                               set(prompts.IPC_TO_BNS.values()), prompts.IPC_TO_BNS)

    # ───────────────────────────── saved judgments + chat history ─────────────────────────────
    def _open_case(self, doc: str, name: str, hide_cited: bool, chat_id: Optional[str] = None):
        """Register the judgment and open a conversation on it: the requested one, else a new (empty) chat."""
        try:
            c = cases.open_case(doc, name, hide_cited)
        except Exception:                                    # noqa: BLE001  (saving is best-effort, never blocks analysis)
            self.case = self.conv = None
            return
        self.case = c
        if c.get("name"):
            self.doc_name = c["name"]
        ch = cases.load_chat(chat_id) if chat_id else None
        self._use_chat(ch if ch and ch.get("case_id") == c["id"] else self._fresh_chat())

    def _fresh_chat(self) -> dict:
        for ch in cases.chats_for(self.case["id"]):          # an empty chat of this judgment is reused, not duplicated
            if not ch.get("messages"):
                return ch
        return cases.new_chat(self.case["id"])              # written to disk with its first question

    def _use_chat(self, ch: dict):
        """The model's memory and the router's history are the conversation's own."""
        self.conv = ch
        self.turns = list(ch.get("turns", []))
        if self.router is not None:
            self.router.history = list(ch.get("history", []))
            self.router.pending = None
        if hasattr(self._llm, "rewind_to_base"):
            self._llm.rewind_to_base()

    def conv_public(self) -> Optional[dict]:
        ch = self.conv
        return None if ch is None else {"id": ch["id"], "title": ch.get("title") or "New chat", "messages": ch.get("messages", [])}

    def new_chat(self) -> dict:
        if self.case is None:
            raise ValueError("open a judgment first")
        with self.lock:
            self._use_chat(self._fresh_chat())
        return {"conv": self.conv_public()}

    def open_chat(self, chat_id: str) -> Iterator[dict]:
        """Switch to a saved conversation; a conversation on another judgment re-opens that judgment (result cache, ~1 s)."""
        ch = cases.load_chat(chat_id)
        if ch is None:
            yield {"type": "error", "message": "this chat no longer exists"}
            return
        if self.case is not None and self.res is not None and ch.get("case_id") == self.case["id"]:
            with self.lock:
                self._use_chat(ch)
            yield {"type": "result", "result": self.public(self.res)}
            yield {"type": "done", "stats": {"cached": True, "switched": True}}
            return
        c, text = cases.load(ch["case_id"]), cases.text(ch["case_id"])
        if c is None or text is None:
            yield {"type": "error", "message": "the judgment of this chat is no longer saved"}
            return
        yield from self.analyse(text=text, name=c.get("name"), hide_cited=c.get("hide_cited", True), chat_id=chat_id)

    def chat_list(self) -> List[dict]:
        names = {c["id"]: c for c in cases.list_cases()}
        cur = self.conv["id"] if self.conv else None
        out = [{"id": ch["id"], "title": ch.get("title") or "New chat", "updated": ch.get("updated"), "case_id": ch.get("case_id"),
                "judgment": (names.get(ch.get("case_id")) or {}).get("name", "judgment"), "messages": len(ch.get("messages", [])),
                "current": ch["id"] == cur}
               for ch in cases.all_chats() if ch.get("messages")]
        if self.conv is not None and not self.conv.get("messages"):        # the open, still empty "New chat"
            out.insert(0, {"id": self.conv["id"], "title": "New chat", "updated": time.time(), "case_id": self.conv["case_id"],
                           "judgment": self.doc_name or "judgment", "messages": 0, "current": True})
        return out

    def rename_chat(self, chat_id: str, title: str) -> dict:
        ch = self.conv if self.conv and self.conv["id"] == chat_id else cases.load_chat(chat_id)
        if ch is None:
            raise ValueError("no such chat")
        ch["title"] = re.sub(r"\s+", " ", title).strip()[:80] or ch.get("title") or "New chat"
        if ch.get("messages"):
            cases.save_chat(ch, touch=False)
        return {"id": chat_id, "title": ch["title"]}

    def delete_chat(self, chat_id: str) -> dict:
        cases.delete_chat(chat_id)
        if self.conv is not None and self.conv["id"] == chat_id and self.case is not None:
            with self.lock:
                self._use_chat(self._fresh_chat())
            return {"switched": True, "conv": self.conv_public()}
        return {"switched": False}

    def _save_conv(self):
        ch = self.conv
        if ch is None:
            return
        try:
            ch["turns"] = self.turns[-40:]
            ch["history"] = list(self.router.history[-40:]) if self.router else []
            cases.save_chat(ch)
        except Exception:                                    # noqa: BLE001
            pass

    def chat(self, message: str, role: Optional[str] = None) -> Iterator[dict]:
        """Answer a question (see _chat) and add it to the open conversation."""
        conv, rec, text = self.conv, {"q": message.strip(), "role": (role or self.role or "accused").lower(), "t": time.time()}, []
        for ev in self._chat(message, role=role):
            k = ev.get("type")
            if k == "token":
                text.append(ev["text"])
            elif k == "revise":
                text, rec["fixed"] = [], True
            elif k == "verify":
                rec["warn"] = True
            elif k == "route":
                rec.update(route=ev.get("route"), p_fail=ev.get("p_fail"))
            elif k == "error":
                rec["error"] = ev.get("message")
            yield ev
        rec["a"] = "".join(text).strip()
        if conv is not None and conv is self.conv and rec["q"] and (rec["a"] or rec.get("error")):
            conv.setdefault("messages", []).append(rec)
            if not conv.get("title"):
                conv["title"] = cases.title_from_question(rec["q"])
                yield {"type": "title", "id": conv["id"], "title": conv["title"]}
            self._save_conv()

    def explain(self, role: Optional[str] = None) -> Iterator[dict]:
        """The briefing for one perspective (see _explain), saved with the judgment."""
        case, r, text, stats = self.case, (role or self.role or "accused").lower(), [], {}
        for ev in self._explain(role=role):
            if ev.get("type") == "token":
                text.append(ev["text"])
            elif ev.get("type") == "done":
                stats = ev.get("stats", {}) or stats
            yield ev
        if case is not None and case is self.case and "".join(text).strip():
            case.setdefault("explain", {})[r] = {"text": "".join(text), "model": stats.get("model"), "t": time.time()}
            try:
                cases.save(case)
            except Exception:                            # noqa: BLE001
                pass

    def clear_chat(self):
        """Clears the conversational chat history while keeping the loaded judgment and models intact."""
        self.turns = []
        if self.router:
            self.router.history = []
            self.router.pending = None
        if self.conv is not None:
            self.conv["messages"] = []
            self._save_conv()
        if hasattr(self._llm, "rewind_to_base"):
            self._llm.rewind_to_base()

    def kb_search(self, query: str, k: int = 6) -> List[dict]:
        """Search the offline legal_kb (statute cards + precedent passages) and the procedural cards. CPU only."""
        br = self._get_retriever()
        qv = br.embed([query])[0]
        pool = br.search_wide(query, qv, k)
        return [{"id": p["id"], "where": p["where"], "relevance": round(p["relevance"], 3), "text": p["text"][:600]} for p in pool
                if p["where"] != "judgment"]

    def sentence(self, i: int) -> Optional[str]:
        return self.res["sentences_list"][i] if self.res and 0 <= i < len(self.res["sentences_list"]) else None

    # ───────────────────────────── status ─────────────────────────────
    def state(self) -> dict:
        r = self.res
        return {"loaded": self.loaded, "model": self.model_name, "verifier": self.verifier, "role": self.role,
                "device": self.an.device, "load_seconds": self.an.load_seconds, "gpu": gpu_stats(),
                "ollama": bool(ollama_models()),
                "document": None if r is None else {"name": self.doc_name, "tokens": r["doc_tokens"], "sentences": r["sentences"],
                                                    "sections": r["sections"], "seconds": r["seconds"]}}


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    return out
