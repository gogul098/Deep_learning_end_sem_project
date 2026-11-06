"""client.py · HttpEngine: the Engine interface over a running server (app.py), so the TUI can attach to it instead of loading
a second copy of the models into the GPU."""
from __future__ import annotations

import json, os, urllib.error, urllib.parse, urllib.request
from typing import Iterator, List, Optional


class HttpEngine:
    def __init__(self, url: str = "http://127.0.0.1:7860"):
        self.url = url.rstrip("/")
        self.remote = True

    # -- plumbing --------------------------------------------------------------------------------------------------
    def _json(self, path: str, body: Optional[dict] = None, timeout: float = 10):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data, {"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode()[:300]) from None

    def _sse(self, path: str, data: Optional[bytes] = None, headers: Optional[dict] = None) -> Iterator[dict]:
        req = urllib.request.Request(self.url + path, data if data is not None else b"", {"Content-Type": "application/json", **(headers or {})}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                for line in r:
                    if line.startswith(b"data: "):
                        yield json.loads(line[6:])
        except urllib.error.HTTPError as e:
            yield {"type": "error", "message": e.read().decode()[:300]}

    @staticmethod
    def alive(url: str, timeout: float = 1.0) -> bool:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/api/state", timeout=timeout) as r:
                json.load(r)
            return True
        except Exception:
            return False

    # -- Engine interface ------------------------------------------------------------------------------------------
    def load(self, emit=lambda e: None):
        return self

    def state(self) -> dict:
        return self._json("/api/state")

    def models(self) -> List[dict]:
        return self._json("/api/models")

    def set_model(self, name: str) -> dict:
        return self._json("/api/model", {"name": name})

    def set_verifier(self, mode: str) -> dict:
        return self._json("/api/verifier", {"mode": mode})

    def set_role(self, role: str) -> dict:
        return self._json("/api/role", {"role": role})

    def analyse(self, text: Optional[str] = None, path: Optional[str] = None, name: Optional[str] = None,
                hide_cited: bool = True, use_cache: bool = True) -> Iterator[dict]:
        if path:
            with open(path, "rb") as f:
                data = f.read()
            yield from self._sse("/api/analyse", data, {"Content-Type": "application/octet-stream", "X-Filename": os.path.basename(path),
                                                        "X-Hide-Cited": "1" if hide_cited else "0", "X-No-Cache": "0" if use_cache else "1"})
        else:
            yield from self._sse("/api/analyse", json.dumps({"text": text or "", "name": name, "hide_cited": hide_cited, "no_cache": not use_cache}).encode())

    def explain(self, role: Optional[str] = None) -> Iterator[dict]:
        body = json.dumps({"role": role}).encode() if role else None
        yield from self._sse("/api/explain", body)

    def chat(self, message: str, role: Optional[str] = None) -> Iterator[dict]:
        payload = {"message": message}
        if role:
            payload["role"] = role
        yield from self._sse("/api/chat", json.dumps(payload).encode())

    def kb_search(self, query: str, k: int = 6) -> List[dict]:
        return self._json(f"/api/kb?q={urllib.parse.quote(query)}&k={k}", timeout=30)

    def sentence(self, i: int) -> Optional[str]:
        try:
            return self._json(f"/api/sentence/{i}")["text"]
        except Exception:
            return None
