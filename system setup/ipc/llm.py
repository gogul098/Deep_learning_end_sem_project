"""llm.py · chat models behind one interface: stream(system, messages) -> text pieces.

  OllamaLLM     any model from `ollama list` (HTTP on 127.0.0.1:11434), streamed
  BuiltinLlama  the local Llama-3.2-3B base (4-bit) with the verifier LoRA switched OFF for chat
  None          no chat model: the engine answers extractively from the retrieved passages
"""
from __future__ import annotations

import json, os, threading, time, urllib.request
from typing import Iterator, List, Optional

def ollama_url(raw: Optional[str] = None) -> str:
    """Where to CONNECT to Ollama. OLLAMA_HOST is often set to a LISTEN address ("0.0.0.0", ":11434", "0.0.0.0:11434")
    so Ollama serves the network; connecting to 0.0.0.0 fails on Windows (WinError 10049), so use 127.0.0.1 instead."""
    from urllib.parse import urlsplit
    raw = (raw if raw is not None else os.environ.get("OLLAMA_HOST", "")).strip()
    if not raw:
        return "http://127.0.0.1:11434"
    u = urlsplit(raw if "://" in raw else "http://" + raw)
    host = (u.hostname or "").strip("[]")
    if host in ("", "0.0.0.0", "::", "*", "localhost"):          # localhost can resolve to ::1 first on Windows
        host = "127.0.0.1"
    try:
        port = u.port or 11434
    except ValueError:
        port = 11434
    return f"{u.scheme or 'http'}://{host}:{port}"


OLLAMA = ollama_url()


def ollama_models(host: str = OLLAMA, timeout: float = 2.0) -> List[dict]:
    """[{name, size_gb, params, quant, family}] from /api/tags; [] if Ollama is not running."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            data = json.load(r)
    except Exception:
        return []
    out = []
    for m in data.get("models", []):
        d = m.get("details", {})
        out.append({"name": m["name"], "size_gb": round(m.get("size", 0) / 2**30, 1), "params": d.get("parameter_size", ""),
                    "quant": d.get("quantization_level", ""), "family": d.get("family", "")})
    return out


def ollama_unload_all(host: str = OLLAMA) -> List[str]:
    """Ask Ollama to drop every model it keeps resident (they come back on demand). Frees VRAM for the verifier."""
    gone = []
    try:
        with urllib.request.urlopen(f"{host}/api/ps", timeout=2) as r:
            running = [m["name"] for m in json.load(r).get("models", [])]
        for name in running:
            req = urllib.request.Request(f"{host}/api/generate", json.dumps({"model": name, "keep_alive": 0}).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15).read()
            gone.append(name)
        if gone:
            time.sleep(1.5)
    except Exception:
        pass
    return gone


class OllamaLLM:
    kind = "ollama"

    def __init__(self, model: str, host: str = OLLAMA, num_ctx: int = None, keep_alive: str = "10m", num_gpu: int = None):
        # 8192: the explanation prompt is ~5-7k tokens; at Ollama's 4096 it was silently cut from the front.
        num_ctx = num_ctx or int(os.environ.get("IPC_NUM_CTX", "8192"))
        # num_gpu = how many layers go to the GPU (rest run on CPU from RAM). None = Ollama decides from free VRAM.
        if num_gpu is None and os.environ.get("IPC_NUM_GPU"):
            num_gpu = int(os.environ["IPC_NUM_GPU"])
        self.model, self.host, self.num_ctx, self.keep_alive, self.num_gpu = model, host, num_ctx, keep_alive, num_gpu
        self.name = model
        self.stats: dict = {}

    def stream(self, system: str, messages: List[dict], max_tokens: int = 1100, temperature: float = 0.0) -> Iterator[str]:
        body = {"model": self.model, "stream": True, "keep_alive": self.keep_alive,
                "messages": [{"role": "system", "content": system}] + messages,
                "options": {"num_ctx": self.num_ctx, "num_predict": max_tokens, "temperature": temperature, "top_p": 0.9}}
        if self.num_gpu is not None:
            body["options"]["num_gpu"] = self.num_gpu
        req = urllib.request.Request(f"{self.host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if not line.strip():
                    continue
                j = json.loads(line)
                if j.get("error"):
                    raise RuntimeError(j["error"])
                piece = j.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if j.get("done"):
                    ec, ed = j.get("eval_count", 0), j.get("eval_duration", 0)
                    self.stats = {"tokens": ec, "tokens_per_s": round(ec / (ed / 1e9), 1) if ed else None,
                                  "prompt_tokens": j.get("prompt_eval_count", 0), "seconds": round(time.time() - t0, 1)}

    def unload(self):
        try:
            req = urllib.request.Request(f"{self.host}/api/generate", json.dumps({"model": self.model, "keep_alive": 0}).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            pass


class BuiltinLlama:
    kind = "builtin"
    name = "builtin"

    def __init__(self, analyzer):
        self.an = analyzer
        self.stats: dict = {}
        self.cache = None
        self.cached_tokens = None
        self.base_seq_len = 0

    def reset_cache(self):
        """Invalidate the KV cache (e.g. when switching PDFs or freeing VRAM)."""
        self.cache = None
        self.cached_tokens = None
        self.base_seq_len = 0
        if self.an.gpu:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

    def rewind_to_base(self):
        """Roll back chat conversation back to the base document prefix (for 'New Chat')."""
        if self.cache is not None and self.base_seq_len > 0:
            try:
                self.cache.crop(self.base_seq_len)
                if self.cached_tokens is not None:
                    self.cached_tokens = self.cached_tokens[:, :self.base_seq_len]
            except Exception:
                self.reset_cache()

    def stream(self, system: str, messages: List[dict], max_tokens: int = 700, temperature: float = 0.3) -> Iterator[str]:
        import torch
        from transformers import DynamicCache, TextIteratorStreamer
        self.an.ensure_verifier()
        llm, tok = self.an.llm, self.an.tok
        dev = next(llm.parameters()).device
        msgs = [{"role": "system", "content": system}] + messages
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        inp_ids = ids["input_ids"].to(dev)
        ids = {k: v.to(dev) for k, v in ids.items()}

        cached_len = 0
        if self.cache is not None and self.cached_tokens is not None:
            c_len = self.cached_tokens.shape[1]
            if inp_ids.shape[1] >= c_len and torch.equal(inp_ids[:, :c_len], self.cached_tokens.to(dev)):
                cached_len = c_len
                use_cache = self.cache
            else:
                self.cache = DynamicCache()
                use_cache = self.cache
                self.base_seq_len = 0
        else:
            self.cache = DynamicCache()
            use_cache = self.cache
            self.base_seq_len = 0

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        err: list = []
        gen_result: list = []

        def run():
            try:
                with torch.no_grad(), llm.disable_adapter():       # the LoRA was trained to answer yes/no: switch it off for chat
                    out = llm.generate(
                        **ids,
                        past_key_values=use_cache,
                        return_dict_in_generate=True,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0,
                        temperature=max(temperature, 1e-3),
                        top_p=0.9,
                        streamer=streamer,
                        pad_token_id=tok.eos_token_id,
                    )
                    gen_result.append(out)
            except Exception as e:                                  # noqa: BLE001
                err.append(e); streamer.end()

        t0, n = time.time(), 0
        th = threading.Thread(target=run, daemon=True); th.start()
        for piece in streamer:
            n += 1
            yield piece
        th.join()
        if err:
            raise err[0]

        if gen_result:
            res = gen_result[0]
            self.cache = res.past_key_values
            self.cached_tokens = res.sequences
            if self.base_seq_len == 0:
                self.base_seq_len = int(inp_ids.shape[1])

        self.stats = {
            "tokens": n,
            "tokens_per_s": round(n / max(time.time() - t0, 1e-3), 1),
            "prompt_tokens": int(inp_ids.shape[1]),
            "cached_prefix_tokens": cached_len,
            "seconds": round(time.time() - t0, 1),
        }

    def unload(self):
        self.reset_cache()
