"""app.py · offline web server: inference API + the web app (index) + the live architecture explainer.

    ./run_app.sh [--lan] [--port 7860] [--model builtin|<ollama name>|none] [--verifier llama|none]

  GET  /                  the app (upload a judgment, explanation, chat)
  GET  /architecture      project architecture explainer with live view
  POST /api/analyse       SSE   body = raw PDF/TXT bytes (X-Filename header) or JSON {"text": ...}
  POST /api/explain       SSE   explanation of the loaded judgment in the lawyer's four-part format
  POST /api/chat          SSE   {"message": ...}: Bayesian router -> retrieval -> chat model
  GET  /api/chats               chat history (every conversation, per judgment; see ipc/cases.py)
  POST /api/chats/new  POST /api/chats/{id}/open (SSE)  PATCH /api/chats/{id} {"title"}  DELETE /api/chats/{id}
  GET  /api/state  /api/models  /api/graph      POST /api/model {"name"}  /api/verifier {"mode"}  /api/bayes {"evidence"}
  GET  /api/events        SSE   every event produced for any client (this is what the live architecture view watches)
"""
import argparse, json, os, queue, tempfile, threading, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")

ap = argparse.ArgumentParser()
ap.add_argument("--lan", action="store_true", help="listen on the local network, not only this computer")
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--model", default=None, help="chat model: builtin | none | an `ollama list` name (default: last used)")
ap.add_argument("--verifier", default=None, choices=["llama", "none"], help="section verifier (none = Hyena encoder only, no Llama)")
args, _ = ap.parse_known_args()

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_load, daemon=True).start()          # models load in the background; /api/state reports progress
    yield


app = FastAPI(title="IPC Assistant", lifespan=lifespan)
ENGINE = None
LOAD = {"state": "loading", "error": None, "t0": time.time()}
SUBS: list = []                      # live-view subscribers (queues)
SUBS_LOCK = threading.Lock()


def publish(ev: dict):
    with SUBS_LOCK:
        for q in list(SUBS):
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass


OOM_HELP = ("Not enough free GPU memory to load the models. Another program is using the GPU, usually a second copy of this app "
            "(for example a terminal UI started with ./tui.sh before the server, which then loaded its own models). "
            "Close it, then press Retry. A terminal UI that loads the models also serves this page, so one copy is enough.")


def _load():
    global ENGINE
    if ENGINE is not None:                                            # models handed over by the process that embeds this server (the TUI)
        LOAD["state"] = "ready"
        return
    LOAD.update(state="loading", error=None, t0=time.time())
    e = None
    try:
        from ipc.engine import Engine
        e = Engine(model=args.model, verifier=args.verifier)
        publish({"type": "server", "status": "loading"})
        e.load(emit=lambda ev: publish(ev))
        ENGINE = e
        LOAD["state"] = "ready"
        publish({"type": "server", "status": "ready"})
    except Exception as ex:                                           # noqa: BLE001
        oom = "out of memory" in str(ex).lower()
        LOAD.update(state="error", error=OOM_HELP if oom else f"{type(ex).__name__}: {ex}")
        publish({"type": "server", "status": "error", "message": LOAD["error"]})
        try:                                                          # do not sit on a half-loaded model
            if e is not None:
                e.an.llm = e.an.encoders = e.an.emb = None
            del e
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:                                             # noqa: BLE001
            pass


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_load, daemon=True).start()          # models load in the background; /api/state reports progress
    yield


app = FastAPI(title="IPC Assistant", lifespan=lifespan)
ENGINE = None
LOAD = {"state": "loading", "error": None, "t0": time.time()}
SUBS: list = []                      # live-view subscribers (queues)
SUBS_LOCK = threading.Lock()


def publish(ev: dict):
    with SUBS_LOCK:
        for q in list(SUBS):
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass


def _load():
    global ENGINE
    try:
        from ipc.engine import Engine
        e = Engine(model=args.model, verifier=args.verifier)
        publish({"type": "server", "status": "loading"})
        e.load(emit=lambda ev: publish(ev))
        ENGINE = e
        LOAD["state"] = "ready"
        publish({"type": "server", "status": "ready"})
    except Exception as ex:                                           # noqa: BLE001
        LOAD.update(state="error", error=f"{type(ex).__name__}: {ex}")
        publish({"type": "server", "status": "error", "message": LOAD["error"]})


def engine():
    if ENGINE is None:
        raise HTTPException(503, LOAD["error"] or "models are still loading")
    return ENGINE


def sse(gen: Iterator[dict], kind: str) -> StreamingResponse:
    def body():
        publish({"type": "request", "kind": kind, "t": time.time()})
        try:
            for ev in gen:
                publish({**ev, "kind": kind})
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        except Exception as ex:                                       # noqa: BLE001
            err = {"type": "error", "message": f"{type(ex).__name__}: {ex}"}
            publish({**err, "kind": kind})
            yield f"data: {json.dumps(err)}\n\n"
    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ───────────────────────────── pages ─────────────────────────────
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/architecture")
def architecture():
    return FileResponse(os.path.join(WEB, "architecture.html"))


# ───────────────────────────── API ─────────────────────────────
STARTED = time.time()          # the launcher restarts a server that is older than the code on disk (after an update)


@app.get("/api/state")
def state():
    if ENGINE is None:
        return {"loaded": False, "load_state": LOAD["state"], "error": LOAD["error"], "elapsed": round(time.time() - LOAD["t0"], 1),
                "started": STARTED}
    return {**ENGINE.state(), "load_state": "ready", "started": STARTED}


@app.post("/api/reload")
def reload_models():
    """Try loading the models again after a failed start (e.g. once the other GPU program has been closed)."""
    if ENGINE is None and LOAD["state"] == "error":
        threading.Thread(target=_load, daemon=True).start()
    return {"load_state": LOAD["state"]}


@app.get("/api/models")
def models():
    return engine().models()


@app.post("/api/model")
async def set_model(request: Request):
    try:
        return engine().set_model((await request.json())["name"])
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/verifier")
async def set_verifier(request: Request):
    try:
        return engine().set_verifier((await request.json())["mode"])
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/role")
async def set_role(request: Request):
    try:
        return engine().set_role((await request.json())["role"])
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/analyse")
async def analyse(request: Request):
    e = engine()
    ctype = request.headers.get("content-type", "")
    hide = request.headers.get("x-hide-cited", "1") != "0"
    cache = request.headers.get("x-no-cache", "0") != "1"
    if "json" in ctype:
        j = await request.json()
        return sse(e.analyse(text=j.get("text", ""), name=j.get("name"), hide_cited=j.get("hide_cited", True),
                                 use_cache=not j.get("no_cache", False)), "analyse")
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty body")
    name = os.path.basename(request.headers.get("x-filename", "judgment.pdf"))
    suffix = ".pdf" if (name.lower().endswith(".pdf") or data[:4] == b"%PDF") else ".txt"
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="ipc-upload-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)

    def run():
        try:
            yield from e.analyse(path=tmp, name=name, hide_cited=hide, use_cache=cache)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return sse(run(), "analyse")


@app.post("/api/explain")
async def explain(request: Request):
    role = None
    try:
        body = await request.body()
        if body:
            role = (json.loads(body)).get("role")
    except Exception:
        pass
    return sse(engine().explain(role=role), "explain")


@app.post("/api/chat")
async def chat(request: Request):
    j = await request.json()
    msg = j.get("message", "")
    role = j.get("role")
    return sse(engine().chat(msg, role=role), "chat")


@app.post("/api/clear-chat")
def clear_chat():
    engine().clear_chat()
    return {"ok": True}


# ───────────────────────────── chat history (ipc/cases.py) ─────────────────────────────
@app.get("/api/chats")
def list_chats():
    return engine().chat_list()


@app.post("/api/chats/new")
def new_chat():
    try:
        return engine().new_chat()
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/chats/{chat_id}/open")
def open_chat(chat_id: str):
    """Switch to a saved conversation (re-opens its judgment if another one is open)."""
    return sse(engine().open_chat(chat_id), "analyse")


@app.patch("/api/chats/{chat_id}")
async def rename_chat(chat_id: str, request: Request):
    try:
        return engine().rename_chat(chat_id, (await request.json()).get("title", ""))
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    return engine().delete_chat(chat_id)


@app.post("/api/bayes")
async def bayes_route(request: Request):
    """Run the Bayesian network on hand-set evidence ({node: "Low"|"High"}); used by the architecture page's playground."""
    from ipc import bayes
    ev = (await request.json()).get("evidence", {})
    try:
        return bayes.route_query(ev)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/result")
def result():
    """The analysis of the judgment that is open right now (whoever opened it: this page, the terminal UI, the CLI)."""
    e = engine()
    if e.res is None:
        raise HTTPException(404, "no judgment is open")
    return e.public(e.res)


@app.get("/api/sentence/{i}")
def sentence(i: int):
    r = engine().res
    if r is None or not (0 <= i < len(r["sentences_list"])):
        raise HTTPException(404, "no such sentence")
    return {"id": f"S{i}", "text": r["sentences_list"][i]}


@app.get("/api/kb")
def kb(q: str, k: int = 6):
    return engine().kb_search(q, k)


@app.get("/api/graph")
def graph():
    from ipc import bayes
    return bayes.GRAPH


@app.get("/api/events")
async def events(request: Request):
    q: queue.Queue = queue.Queue(maxsize=2000)
    with SUBS_LOCK:
        SUBS.append(q)

    def body():
        try:
            yield f"data: {json.dumps({'type': 'hello', 'ready': ENGINE is not None})}\n\n"
            while True:
                try:
                    ev = q.get(timeout=15)
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with SUBS_LOCK:
                if q in SUBS:
                    SUBS.remove(q)
    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def already_running(port: int) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=1.5) as r:
            return "load_state" in json.load(r)
    except Exception:                                                 # noqa: BLE001
        return False


def serve_in_thread(engine, port: int = 7860, lan: bool = False) -> bool:
    """Serve the web app + API from a process that already holds the models (the terminal UI). False if the port is taken."""
    global ENGINE
    import socket, uvicorn
    with socket.socket() as sk:
        if sk.connect_ex(("127.0.0.1", port)) == 0:
            return False
    ENGINE = engine
    LOAD["state"] = "ready"
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0" if lan else "127.0.0.1", port=port, log_level="critical"))
    server.install_signal_handlers = lambda: None
    threading.Thread(target=server.run, daemon=True).start()
    return True


if __name__ == "__main__":
    import uvicorn
    a = ap.parse_args()
    if already_running(a.port):
        print(f"IPC Assistant is already running on http://127.0.0.1:{a.port} (a server or a terminal UI that loaded the models).\n"
              f"Open that address in the browser; ./tui.sh attaches to it. Nothing was started.")
        raise SystemExit(0)
    host = "0.0.0.0" if a.lan else "127.0.0.1"
    print(f"IPC Assistant → http://{host if a.lan else '127.0.0.1'}:{a.port}   (architecture explainer: /architecture)")
    uvicorn.run(app, host=host, port=a.port, log_level="warning")
