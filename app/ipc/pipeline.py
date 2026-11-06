"""pipeline.py · the analysis pipeline: InLegalBERT -> Hyena encoders -> shortlist -> Llama verifier -> fusion.

Same maths as LegalPredictor.__call__ in models/final/legal_hybrid.py (which is imported and reused, not edited), with:
  * InLegalBERT on the GPU in fp16 when it fits (18 s -> 0.3 s on the CPU fallback the old app forced on 6 GB cards)
  * the verifier reads the shared prompt prefix of its two candidate orders once (KV cache reused, then cropped)
  * a pre-quantised 4-bit Llama cache (models/llama-4bit) so start-up skips the 6.4 GB -> 4-bit conversion
  * sentence indices kept ([S12] ids), per-stage events with timings, verifier = llama | none, on-disk result cache
"""
from __future__ import annotations

import hashlib, json, os, re, sys, time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.environ.get("LEGAL_MODELS_DIR", os.path.join(ROOT, "models"))
CACHE = os.environ.get("IPC_CACHE_DIR", os.path.join(ROOT, ".cache", "results"))
CACHE_VERSION = 3


def _paths(models_dir=MODELS):
    p = json.load(open(os.path.join(models_dir, "paths.json")))
    return {k: v if os.path.isabs(v) else os.path.join(models_dir, v) for k, v in p.items()}


def _lh(final_dir):
    if final_dir not in sys.path:
        sys.path.insert(0, final_dir)
    import legal_hybrid as LH
    return LH


class Analyzer:
    def __init__(self, models_dir: str = MODELS, device: Optional[str] = None, verifier: str = "llama",
                 shared_prefix: bool = True, embed_on_gpu: Optional[bool] = None):
        self.models_dir = models_dir
        self.p = _paths(models_dir)
        self.LH = _lh(self.p["final"])
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gpu = self.device.startswith("cuda")
        self.verifier = verifier if verifier in ("llama", "none") else "llama"
        self.shared_prefix = shared_prefix
        if embed_on_gpu is None:
            # Small GPUs (< 7 GB VRAM like 6 GB RTX 3050): keep InLegalBERT on CPU so VRAM is reserved for Llama & KV cache
            self.embed_on_gpu = self.gpu and torch.cuda.get_device_properties(0).total_memory >= 7 * 1024**3
        else:
            self.embed_on_gpu = embed_on_gpu and self.gpu
        self.ck = self.tok = self.llm = self.encoders = self.emb = None
        self.last_vectors: Optional[np.ndarray] = None
        self.load_seconds: Dict[str, float] = {}

    # ───────────────────────────── loading ─────────────────────────────
    def load(self, emit: Callable = lambda e: None):
        LH, p, dev = self.LH, self.p, torch.device(self.device)
        t = time.time()
        self.ck = torch.load(os.path.join(p["final"], "hybrid_model.pt"), map_location="cpu", weights_only=False)
        ck = self.ck
        ampere = self.gpu and torch.cuda.get_device_capability(dev)[0] >= 8
        ck["dtype"] = (torch.bfloat16 if ampere else torch.float16) if self.gpu else torch.float32
        self.encoders = []
        for sd in ck["encoders"]:
            e = LH.LegalHyenaEncoder(**ck["encoder_config"]); e.load_state_dict(sd); self.encoders.append(e.to(dev).eval())
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(os.path.join(p["final"], "llama_lora"))
        self.load_seconds["checkpoint"] = round(time.time() - t, 1)
        t = time.time()
        self.emb = LH.SentenceEmbedder.load(p["legal_encoder"], self.device if self.embed_on_gpu else "cpu", batch=8 if self.embed_on_gpu else 16)
        # Lazy-load verifier on GPU: loaded on-demand in verify() or chat to keep startup VRAM under ~200MB
        if self.verifier == "llama" and not self.gpu:
            self.ensure_verifier()
        emit({"type": "loaded", "seconds": self.load_seconds, "vram_gb": self.vram()})
        return self

    def vram(self):
        return round(torch.cuda.memory_allocated() / 2**30, 2) if self.gpu else None

    def ensure_verifier(self):
        if self.llm is not None:
            return
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        t = time.time()
        dtype, dev = self.ck["dtype"], torch.device(self.device)
        q_dir = os.path.join(self.models_dir, "llama-4bit")
        if self.gpu:
            if os.path.exists(os.path.join(q_dir, "config.json")):
                base = AutoModelForCausalLM.from_pretrained(q_dir, device_map={"": dev.index or 0}, dtype=dtype)
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    self.p["base_model"], dtype=dtype, device_map={"": dev.index or 0},
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
                try:
                    base.save_pretrained(q_dir)          # first run only: later loads skip the conversion
                except Exception as e:
                    print("could not write the 4-bit cache:", e)
        else:
            base = AutoModelForCausalLM.from_pretrained(self.p["base_model"], dtype=dtype).to(dev)
        self.llm = PeftModel.from_pretrained(base, os.path.join(self.p["final"], "llama_lora")).eval()
        self.load_seconds["verifier"] = round(time.time() - t, 1)

    def unload_verifier(self):
        """Free the verifier's VRAM (e.g. before an Ollama model needs the whole card)."""
        if self.llm is not None:
            self.llm = None
            import gc; gc.collect()
            if self.gpu:
                torch.cuda.empty_cache()

    # ───────────────────────────── verifier ─────────────────────────────
    @torch.no_grad()
    def _score(self, system, user, cands, cache=None, fed=None):
        """Same procedure as legal_hybrid.score_candidates, optionally starting from an already-read prefix."""
        from transformers import DynamicCache
        LH = self.LH
        body, head = LH.llm_body(self.llm), LH.llm_head(self.llm)
        dev = head.weight.device
        _, _, prompt = LH.chat_ids(self.tok, system, user)
        cache, fed = (cache if cache is not None else DynamicCache()), list(fed or [])
        answer, probs = "", []
        for i, c in enumerate(cands):
            answer += ("\n" if i else "") + f"{c}:"
            ids, yes, no = LH._decision_point(self.tok, prompt + answer)
            if ids[:len(fed)] != fed or len(ids) <= len(fed):
                cache, fed = DynamicCache(), []
            out = body(input_ids=torch.tensor([ids[len(fed):]], device=dev), past_key_values=cache, use_cache=True)
            fed = ids
            lg = head(out.last_hidden_state[:, -1]).float()[0]
            p = torch.sigmoid(torch.logsumexp(lg[yes], 0) - torch.logsumexp(lg[no], 0)).item()
            probs.append(p)
            answer += " yes" if p >= 0.5 else " no"
        return probs

    @torch.no_grad()
    def verify(self, passages, cands, evidence, n_orders):
        LH, ck = self.LH, self.ck
        orders = [cands if k % 2 == 0 else cands[::-1] for k in range(n_orders)]
        users = [LH.verify_user(passages, o, ck["statute_desc"], evidence) for o in orders]
        acc = {c: 0.0 for c in cands}
        prefix = 0
        cache = None
        if self.shared_prefix and n_orders > 1:
            from transformers import DynamicCache
            ids = [LH.chat_ids(self.tok, ck["verify_system"], u)[0] for u in users]
            prefix = next((i for i in range(min(map(len, ids))) if len({x[i] for x in ids}) > 1), min(map(len, ids))) - 8
            if prefix > 64:
                body = LH.llm_body(self.llm)
                cache = DynamicCache()
                body(input_ids=torch.tensor([ids[0][:prefix]], device=LH.llm_head(self.llm).weight.device), past_key_values=cache, use_cache=True)
                prefix_ids = ids[0][:prefix]
            else:
                prefix = 0
        for o, u in zip(orders, users):
            ps = self._score(ck["verify_system"], u, o, cache, prefix_ids if prefix else None)
            if prefix:
                cache.crop(prefix)
            for c, p in zip(o, ps):
                acc[c] += p / n_orders
        del cache
        if self.gpu:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        return acc, len(LH.chat_ids(self.tok, ck["verify_system"], users[0])[0]), prefix

    @torch.no_grad()
    def embed_standalone(self, sentences: List[str]) -> np.ndarray:
        """One independent InLegalBERT mean-pooled unit vector per sentence: exactly how the CPU retriever embeds a query
        (and how legal_kb was built), so cosines between questions and sentences are on the scale its cut-offs were set on."""
        m, tok, dev = self.emb.model, self.emb.tok, self.emb.device
        out = np.zeros((len(sentences), self.emb.dim), dtype="float32")
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        for b in range(0, len(order), 32):
            ids = order[b:b + 32]
            x = tok([sentences[i] for i in ids], padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                h = m(**x).last_hidden_state.float()
            mk = x["attention_mask"][..., None].float()
            out[ids] = torch.nn.functional.normalize((h * mk).sum(1) / mk.sum(1), dim=-1).cpu().numpy()
        return out

    # ───────────────────────────── analysis ─────────────────────────────
    def cache_key(self, text: str, hide_cited: bool, verifier: str) -> str:
        h = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
        return f"{h}-{int(hide_cited)}-{verifier}-v{CACHE_VERSION}"

    def analyse(self, text: str, hide_cited: bool = True, emit: Callable = lambda e: None, use_cache: bool = True) -> dict:
        LH, ck, dev = self.LH, self.ck, torch.device(self.device)
        cfg = ck["inference"]
        verifier = self.verifier
        key = self.cache_key(text, hide_cited, verifier)
        cpath = os.path.join(CACHE, key + ".json")
        if use_cache and os.path.exists(cpath):
            r = json.load(open(cpath))
            self.last_vectors = np.load(os.path.join(CACHE, key + ".npy"))
            r["cached"] = True
            emit({"type": "stage", "name": "cache", "status": "end", "ms": 0, "info": {"hit": True}})
            return r

        t_all = time.time()
        if self.gpu:
            torch.cuda.reset_peak_memory_stats()

        def stage(name, fn, info=lambda r: {}):
            emit({"type": "stage", "name": name, "status": "start"})
            t = time.time(); r = fn(); ms = round((time.time() - t) * 1000)
            emit({"type": "stage", "name": name, "status": "end", "ms": ms, "info": info(r)})
            return r

        raw_sents = stage("split", lambda: LH.split_sentences(text), lambda r: {"sentences": len(r)})
        if hide_cited:
            masked_sents, flags = stage("hide", lambda: LH.view_sentences(raw_sents, "full"))
        else:
            masked_sents, flags = raw_sents, [False] * len(raw_sents)
        doc = LH.prepare_doc(masked_sents, self.tok, 65536, flags)
        raw_doc_sents = raw_sents[:len(doc.sentences)]

        with torch.no_grad(), torch.autocast("cuda", dtype=ck["dtype"], enabled=self.gpu):
            vecs = stage("embed", lambda: self.emb(doc.sentences), lambda r: {"vectors": int(r.shape[0]), "dim": int(r.shape[1]),
                                                                              "device": "gpu" if self.embed_on_gpu else "cpu"})
            self.last_vectors = self.embed_standalone(raw_doc_sents)          # for the CPU retriever using original wording
            p_enc, out = stage("hyena", lambda: LH.ensemble_probs(self.encoders, vecs.to(dev)),
                               lambda r: {"encoders": len(self.encoders), "top": [ck["label_names"][j] for j in torch.topk(r[0].float(), 3).indices.tolist()]})
            cand_idx = LH.shortlist(p_enc, cfg["k_min"], cfg["k_max"], cfg["min_p"])
            key_n = lambda s: (int(re.findall(r"\d+", s)[0]) if re.findall(r"\d+", s) else 0, s)
            cands = sorted([ck["label_names"][j] for j in cand_idx], key=key_n)
            emit({"type": "stage", "name": "shortlist", "status": "end", "ms": 0, "info": {"candidates": cands}})

            idx, n_head = LH.select_with_opening(out["scores"], doc.counts, cfg["passage_tokens"],
                                                 head_frac=cfg["head_tokens"] / max(1, cfg["passage_tokens"]))
            passages = LH.opening_selection_text(doc, idx, n_head)
            ev_idx = {ck["label_names"][j]: out["attn"][j].float().topk(min(2, len(doc.sentences))).indices.tolist() for j in cand_idx}
            evidence = {c: [LH.clip_words(raw_doc_sents[t]) for t in ti] for c, ti in ev_idx.items()}

            llm_tokens, prefix = 0, 0
            if verifier == "llama":
                self.ensure_verifier()
                p_llm, llm_tokens, prefix = stage(
                    "verify", lambda: self.verify(passages, cands, evidence, max(1, cfg.get("verify_orders", 1))),
                    lambda r: {"tokens": r[1], "shared_prefix_tokens": r[2], "scores": {c: round(v, 3) for c, v in r[0].items()}})
            else:
                p_llm = {}
        pe = {ck["label_names"][j]: float(p_enc[j]) for j in range(len(ck["label_names"]))}
        if verifier == "llama":
            final = {c: LH.fuse(pe[c], p_llm.get(c), cfg["alpha"]) for c in ck["label_names"]}
            th = [cfg["threshold"]] * len(ck["label_names"])          # one threshold: it generalised best (as in local_model.py)
        else:
            final = dict(pe)
            th = [ck.get("calibration", {}).get("enc_threshold", 0.45)] * len(ck["label_names"])
        secs = [c for j, c in enumerate(ck["label_names"]) if final[c] >= th[j]]
        if not secs and cfg.get("at_least_one"):
            secs = [max(final, key=final.get)]
        secs = sorted(secs, key=key_n)

        sec_ev_idx = {c: ev_idx.get(c, []) for c in secs}
        res = {
            "sections": secs, "scores": {c: round(final[c], 3) for c in secs},
            "candidates": {c: {"encoder": round(pe[c], 3), "llm": round(p_llm.get(c, 0.0), 3) if verifier == "llama" else None,
                               "final": round(final[c], 3)} for c in cands},
            "evidence": evidence, "evidence_idx": ev_idx, "passages": passages,
            "key_idx": idx, "n_head": n_head, "sentences_list": raw_doc_sents,
            "sentence_scores": [round(float(v), 4) for v in out["scores"].tolist()],
            "doc_tokens": len(doc.ids), "sentences": len(doc.sentences), "llm_tokens": llm_tokens,
            "verifier": verifier, "shared_prefix_tokens": prefix,
            "seconds": round(time.time() - t_all, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if self.gpu else None,
            "cached": False,
            "raw_text": text,          # original PDF text for the case-facts extractor (case details, charges, hearings)
        }
        try:
            os.makedirs(CACHE, exist_ok=True)
            json.dump(res, open(cpath, "w")); np.save(os.path.join(CACHE, key + ".npy"), self.last_vectors)
        except Exception as e:
            print("result cache not written:", e)
        return res
