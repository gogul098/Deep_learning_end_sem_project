"""tui.py · terminal UI for the IPC Legal Assistant.

    ./tui.sh                       attach to the running server (http://127.0.0.1:7860) or, if none, load the models in-process
    ./tui.sh --local               always load in-process          ./tui.sh --url http://host:7860
    ./tui.sh --model gemma3:4b     chat model (in-process mode)    ./tui.sh --verifier none   Hyena encoder only
    ./tui.sh --embed "text"        search the offline library (CPU only) and exit

Type to ask about the loaded judgment. Type "/" and a menu opens right above the cursor: ↑/↓ choose, Tab completes, Enter runs.
Menus follow the command: "/model " lists your Ollama models, "/open " lists recent judgments and folders, "/evidence " lists sections.
"""
from __future__ import annotations

import argparse, os, re, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rich import box
from rich.console import Group
from rich.markdown import Markdown as RichMarkdown
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static, TextArea
from textual.widgets.option_list import Option

# ───────────────────────────── palette ─────────────────────────────
BLUE, VIOLET, GREEN, AMBER, RED, DIM, HI = "#58a6ff", "#a78bfa", "#3fb950", "#d29922", "#f85149", "#7d8590", "#e6edf3"
ROUTE_COL = {"Serve": GREEN, "Search": AMBER, "Clarify": BLUE}
STAGES = {"read": "read", "cache": "cache", "hide": "mask", "split": "split", "embed": "embed", "hyena": "hyena",
          "shortlist": "shortlist", "verify": "verify", "route": "route"}

# (command, argument hint, description, argument required?)
COMMANDS: List[Tuple[str, str, str, bool]] = [
    ("/open", "<path>", "open and analyse a judgment (PDF or TXT)", True),
    ("/paste", "", "paste judgment text in a box", False),
    ("/explain", "", "explain the judgment and the suggested sections", False),
    ("/sections", "", "suggested IPC sections with evidence", False),
    ("/candidates", "", "every section the verifier checked", False),
    ("/evidence", "<section>", "evidence sentences for one section", True),
    ("/passages", "", "key passages the verifier read", False),
    ("/route", "", "how the router decided on the last question", False),
    ("/model", "[name]", "list or switch the chat model", False),
    ("/verifier", "<llama|none>", "Llama + Hyena, or Hyena encoder only", True),
    ("/role", "<accused|complainant|neutral>", "switch advocate perspective (defence vs prosecution)", True),
    ("/kb", "<query>", "search the offline legal library", True),
    ("/proceedings", "<section>", "CrPC / BNSS facts for an IPC section", True),
    ("/gpu", "", "GPU memory and temperature", False),
    ("/clear", "", "clear the screen", False),
    ("/help", "", "keys and commands", False),
    ("/quit", "", "leave", False),
]
READ_ONLY = {"/sections", "/candidates", "/evidence", "/passages", "/route", "/model", "/gpu", "/clear", "/help", "/proceedings", "/quit"}

def welcome(url: str = "", served: bool = False) -> Group:
    g = Table.grid(padding=(0, 2))
    for _ in range(4):
        g.add_column()
    rows = [("/open <path>", "open a judgment (PDF or TXT)", "/model", "choose the chat model"),
            ("/explain", "explain it in four parts", "/route", "why a question was answered that way"),
            ("just type", "ask about the open judgment", "/", "every command, right above the cursor")]
    for c1, d1, c2, d2 in rows:
        g.add_row(Text(c1, style=BLUE), Text(d1, style=DIM), Text(c2, style=BLUE), Text(d2, style=DIM))
    parts = [Text("offline · Indian criminal law · Hyena encoder + Llama verifier + Bayesian retriever", style=DIM), Text(""), g]
    if url:
        parts += [Text(""), Text.assemble(("web app and live architecture view  ", DIM), (url, BLUE),
                                          ("   served by this window" if served else "   attached to this server", DIM))]
    return Group(*parts)


def short_name(name: str, n: int = 30) -> str:
    """Middle-ellipsis for long names and sha-style file names: e13d3a51…901352.pdf"""
    if len(name) <= n:
        return name
    stem, ext = os.path.splitext(name)
    ext = ext if len(ext) <= 6 else ""
    stem = name[: -len(ext)] if ext else name
    keep = max(4, (n - len(ext) - 1) // 2)
    return f"{stem[:keep]}…{stem[-keep:]}{ext}"


def bar(p: float, width: int = 10) -> Text:
    n = max(0, min(width, round(p * width)))
    col = GREEN if p >= 0.7 else (BLUE if p >= 0.4 else DIM)
    return Text.assemble(("━" * n, col), ("━" * (width - n), "#21262d"))


def item(tag: str, text, score: Optional[float] = None, tag_style: str = VIOLET, text_style: str = DIM, indent: int = 2) -> Table:
    """One tagged line whose wrapped text hangs under the text, not under the tag."""
    g = Table.grid(padding=(0, 1))
    g.add_column(width=indent); g.add_column(no_wrap=True)
    if score is not None:
        g.add_column(no_wrap=True)
    g.add_column(ratio=1)
    cells = [Text(""), Text(tag, style=tag_style)] + ([Text(f"{score:.2f}", style=DIM)] if score is not None else [])
    g.add_row(*cells, text if isinstance(text, Text) else Text(text, style=text_style))
    return g


def clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


# ───────────────────────────── renderers ─────────────────────────────
def render_sections(r: dict, evidence: bool = True) -> Group:
    weak = r.get("weak_prediction")
    parts: List = [Text("Suggested IPC sections" + ("  · LOW CONFIDENCE, likely not applicable" if weak else ""), style=f"bold {HI}")]
    if weak:
        parts.append(Text("  " + r.get("weak_reason", ""), style=DIM))
    if not r["sections"]:
        parts.append(Text("  no section reached the threshold (/candidates shows what was checked)", style=DIM))
    for s in r["sections"]:
        sc = r["scores"][s]
        line = Text.assemble(("  ", ""), (f"{s:<13}", f"bold {HI}"), bar(sc), (f" {sc:.2f}  ", HI), (r["statutes"].get(s, ""), DIM))
        parts.append(line)
        if evidence:
            for e in r["evidence"].get(s, [])[:2]:
                parts.append(item(f"[{e['id']}]", f"“{clip(e['text'], 220)}”", text_style=f"italic {DIM}", indent=4))
    return Group(*parts)


def render_candidates(r: dict) -> Table:
    t = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style=DIM)
    for c, j in (("section", "left"), ("encoder", "left"), ("llama", "left"), ("final", "left"), ("", "left")):
        t.add_column(c, justify=j)
    for c, v in r["candidates"].items():
        picked = c in r["sections"]
        llm = Text("–", style=DIM) if v["llm"] is None else Text.assemble(bar(v["llm"], 6), (f" {v['llm']:.2f}", DIM))
        t.add_row(Text(c, style=f"bold {HI}" if picked else DIM), Text.assemble(bar(v["encoder"], 6), (f" {v['encoder']:.2f}", DIM)), llm,
                  Text(f"{v['final']:.2f}", style=HI if picked else DIM), Text("✓ suggested" if picked else "", style=GREEN))
    return t


def render_stats(r: dict) -> Text:
    bits = [f"{r['doc_tokens']:,} tokens", f"{r['sentences']} sentences", f"{r['seconds']}s"]
    if r.get("peak_vram_gb"):
        bits.append(f"peak VRAM {r['peak_vram_gb']} GB")
    if r.get("cached"):
        bits.append("from cache")
    return Text("  " + " · ".join(bits), style=DIM)


# ───────────────────────────── modals ─────────────────────────────
class PasteModal(ModalScreen[str]):
    BINDINGS = [Binding("escape", "dismiss('')", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Label("Paste the judgment text · Ctrl+S analyses · Esc cancels", id="modal-title")
            yield TextArea(id="ta")
            with Horizontal(id="modal-btns"):
                yield Button("Analyse", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#ta", TextArea).focus()

    def on_key(self, e) -> None:
        if e.key == "ctrl+s":
            self.dismiss(self.query_one("#ta", TextArea).text.strip())

    @on(Button.Pressed, "#ok")
    def ok(self) -> None:
        self.dismiss(self.query_one("#ta", TextArea).text.strip())

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss("")


# ───────────────────────────── composer (input + menu share one box) ─────────────────────────────
class Composer(Input):
    BINDINGS = [Binding("up", "nav(-1)", priority=True, show=False), Binding("down", "nav(1)", priority=True, show=False),
                Binding("tab", "complete", priority=True, show=False), Binding("escape", "close_menu", priority=True, show=False)]

    def action_nav(self, d: int) -> None:
        self.app.menu_nav(d)

    def action_complete(self) -> None:
        self.app.menu_complete(run=False)

    def action_close_menu(self) -> None:
        self.app.menu_close()


class Suggestion:
    __slots__ = ("fill", "run", "prompt")

    def __init__(self, fill: str, run: bool, prompt: Text):
        self.fill, self.run, self.prompt = fill, run, prompt


CSS = f"""
Screen {{ background: #0b0f14; color: #c9d1d9; }}
#top {{ height: 1; padding: 0 2; background: #0b0f14; }}
#title {{ width: 1fr; color: {HI}; text-style: bold; }}
#gpu {{ width: auto; color: {GREEN}; }}
#chat {{ height: 1fr; padding: 0 2; scrollbar-size-vertical: 1; scrollbar-color: #30363d; scrollbar-background: #0b0f14; }}
.row {{ height: auto; margin: 1 0 0 0; }}
.dot {{ width: 2; color: {VIOLET}; }}
.body {{ width: 1fr; height: auto; }}
.user {{ height: auto; margin: 1 0 0 0; }}
.note {{ height: auto; color: {DIM}; margin: 0; }}
.err-row .dot {{ color: {RED}; }}
#welcome {{ height: auto; margin: 1 0; padding: 0 1; }}
#dock {{ dock: bottom; height: auto; padding: 0 2 0 2; background: #0b0f14; }}
#activity {{ height: auto; padding: 0 1; color: {DIM}; display: none; }}
#activity.on {{ display: block; }}
#box {{ height: auto; border: round #30363d; background: #0f151c; }}
#box:focus-within {{ border: round {BLUE}; }}
#menu {{ display: none; height: auto; max-height: 10; border: none; padding: 0 1; background: #0f151c; scrollbar-size-vertical: 1; }}
#menu.on {{ display: block; }}
#menu > .option-list--option-highlighted {{ background: #1b2430; color: {HI}; text-style: none; }}
#menu > .option-list--option-hover {{ background: #161e28; }}
#sep {{ display: none; height: 1; color: #30363d; padding: 0 1; }}
#sep.on {{ display: block; }}
#inputrow {{ height: 1; padding: 0 1; }}
#prompt {{ width: 2; color: {BLUE}; text-style: bold; }}
#input {{ width: 1fr; height: 1; border: none; padding: 0; background: transparent; color: {HI}; }}
#input:focus {{ border: none; background: transparent; }}
#hintrow {{ height: 1; padding: 0 1; }}
#status {{ width: 1fr; color: {DIM}; }}
#keys {{ width: auto; color: {DIM}; }}
PasteModal {{ align: center middle; }}
#modal {{ width: 96; height: 80%; background: #10161d; border: round {BLUE}; padding: 1 2; }}
#modal-title {{ color: {HI}; text-style: bold; }}
#ta {{ height: 1fr; margin: 1 0; }}
#modal-btns {{ height: 3; align: right middle; }}
#modal-btns Button {{ margin-left: 1; }}
"""


class LegalTUI(App):
    CSS = CSS
    TITLE = "IPC Legal Assistant"
    BINDINGS = [Binding("ctrl+o", "open", show=False), Binding("ctrl+p", "paste", show=False), Binding("ctrl+l", "clear", show=False),
                Binding("f1", "help", show=False), Binding("ctrl+q", "quit", show=False), Binding("ctrl+c", "quit", show=False)]

    def __init__(self, engine, **kw):
        super().__init__(**kw)
        self.engine = engine
        self.result: Optional[dict] = None
        self.last_route: Optional[dict] = None
        self.busy = False
        self.sugg: List[Suggestion] = []
        self.models_cache: List[dict] = []
        self.history: List[str] = []
        self.hist_i = 0
        self._dismissed = ""
        self._stages: Dict[str, Tuple[str, float, Optional[int]]] = {}   # name -> (state, t0, ms)
        self._job_label, self._job_t0, self._spin = "", 0.0, 0
        self._status = {"model": "…", "verifier": "…", "role": "…", "doc": None, "gpu": ""}

    # ───────────── layout ─────────────
    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            yield Static("⚖  IPC Legal Assistant", id="title")
            yield Static("", id="gpu")
        with VerticalScroll(id="chat"):
            served = getattr(self.engine, "web_url", None)
            yield Static(welcome(served or getattr(self.engine, "url", "") or "", bool(served)), id="welcome")
        with Vertical(id="dock"):
            yield Static("", id="activity")
            with Vertical(id="box"):
                yield OptionList(id="menu")
                yield Static("─" * 400, id="sep")
                with Horizontal(id="inputrow"):
                    yield Static("❯", id="prompt")
                    yield Composer(placeholder="Ask about the judgment…  or type / for commands", id="input")
            with Horizontal(id="hintrow"):
                yield Static("", id="status")
                yield Static("/ commands · ↑↓ history · ctrl+o open · ctrl+l clear · ctrl+q quit", id="keys")

    def on_mount(self) -> None:
        self.query_one("#menu", OptionList).can_focus = False
        self.query_one("#input", Composer).focus()
        self.set_interval(2.0, self._poll)
        self.set_interval(0.1, self._tick)
        self._poll()

    # ───────────── status (never on the UI thread) ─────────────
    @work(thread=True, exclusive=True, group="poll")
    def _poll(self) -> None:
        try:
            s = self.engine.state()
        except Exception as e:                                                  # noqa: BLE001
            self.call_from_thread(self._set_status, {"model": "server unreachable", "gpu": str(e)[:30]})
            return
        g = s.get("gpu", {})
        doc = s.get("document")
        self.call_from_thread(self._set_status, {
            "model": s.get("model", "?"), "verifier": s.get("verifier", "?"), "role": s.get("role", "accused"),
            "doc": f"{short_name(doc['name'])} · {doc['tokens'] / 1000:.1f}k tok" if doc else None,
            "gpu": f"{g['name'].replace('NVIDIA GeForce ', '').replace(' Laptop GPU', '')}  {g['used_gb']:.1f}/{g['total_gb']:.0f} GB  {g['temp']}°C" if g.get("ok") else "CPU"})

    def _set_status(self, d: dict) -> None:
        self._status.update(d)
        s = self._status
        r_label = "🛡️ defence" if s.get("role") == "accused" else ("⚖️ prosecution" if s.get("role") == "complainant" else "📜 neutral")
        t = Text.assemble(("model ", DIM), (str(s["model"]).replace("mannix/", ""), VIOLET), ("  ·  verifier ", DIM), (str(s["verifier"]), VIOLET),
                          ("  ·  role ", DIM), (r_label, BLUE),
                          ("  ·  ", DIM), (("📄 " + s["doc"]) if s["doc"] else "no judgment open", HI if s["doc"] else DIM))
        self.query_one("#status", Static).update(t)
        self.query_one("#gpu", Static).update(Text("● " + s["gpu"], style=GREEN) if s["gpu"] else "")

    def _tick(self) -> None:
        if not self.busy:
            return
        self._spin += 1
        sp = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[self._spin % 10]
        now = time.time()
        t = Text(" ")
        for name, (state, t0, ms) in self._stages.items():
            if state == "run":
                t.append(f"{sp} {STAGES.get(name, name)} {now - t0:.1f}s   ", style=BLUE)
            else:
                t.append(f"✓ {STAGES.get(name, name)}{f' {ms / 1000:.1f}s' if ms and ms >= 100 else ''}   ", style=GREEN if False else DIM)
        if not self._stages:
            t.append(f"{sp} {self._job_label}", style=BLUE)
        t.append(f"· {now - self._job_t0:.0f}s  (esc ignored while working)", style=DIM)
        self.query_one("#activity", Static).update(t)

    # ───────────── conversation ─────────────
    def _chat(self) -> VerticalScroll:
        return self.query_one("#chat", VerticalScroll)

    def _clear_welcome(self) -> None:
        for w in self.query("#welcome"):
            w.remove()

    def _mount(self, w) -> None:
        self._clear_welcome()
        c = self._chat()
        c.mount(w)
        c.scroll_end(animate=False)

    def add_user(self, text: str) -> None:
        self._mount(Static(Text.assemble(("❯ ", f"bold {BLUE}"), (text, HI)), classes="user"))

    def add_bot(self, renderable, kind: str = "") -> Static:
        body = Static(renderable, classes="body")
        row = Horizontal(Static("●", classes="dot"), body, classes=f"row {kind}")
        self._mount(row)
        return body

    def add_note(self, text, style: str = DIM) -> None:
        self._mount(Static(text if isinstance(text, Text) else Text("  " + text, style=style), classes="note"))

    def add_err(self, text: str) -> None:
        self.add_bot(Text(text, style=RED), "err-row")

    def md(self, text: str) -> RichMarkdown:
        return RichMarkdown(text, code_theme="ansi_dark", inline_code_lexer=None)

    def set_busy(self, on: bool, label: str = "") -> None:
        self.busy = on
        act = self.query_one("#activity", Static)
        act.set_class(on, "on")
        if on:
            self._stages.clear()
            self._job_label, self._job_t0 = label, time.time()
        else:
            act.update("")
            self.query_one("#input", Composer).focus()

    # ───────────── the menu attached to the input ─────────────
    def _suggest(self, text: str) -> List[Suggestion]:
        if not text.startswith("/"):
            return []
        cmd, sp, arg = text.partition(" ")
        cmd = cmd.lower()
        if not sp:                                                   # still typing the command name
            q = cmd[1:]
            hits = [c for c in COMMANDS if c[0][1:].startswith(q)] or [c for c in COMMANDS if q in c[0]]
            return [Suggestion(c[0] + (" " if c[3] else ""), not c[3],
                               Text.assemble((f"{c[0]:<13}", f"bold {BLUE}"), (f"{c[1]:<13}", DIM), (c[2], DIM))) for c in hits]
        a = arg.strip().lower()
        if cmd == "/model":
            return [Suggestion(f"/model {m['name']}", True,
                               Text.assemble(("● " if m["active"] else "  ", GREEN), (f"{m['label'][:48]:<50}", HI),
                                             (f"{m.get('size_gb', '')} GB " if m.get("size_gb") else "", DIM), (m.get("params", ""), DIM)))
                    for m in self.models_cache if a in m["name"].lower() or a in m["label"].lower()]
        if cmd == "/verifier":
            return [Suggestion(f"/verifier {v}", True, Text.assemble((f"{v:<8}", f"bold {BLUE}"), (d, DIM)))
                    for v, d in (("llama", "Llama verifies the Hyena shortlist (best accuracy)"), ("none", "Hyena encoder only: 0.5 s, no Llama in memory"))
                    if a in v]
        if cmd == "/role":
            return [Suggestion(f"/role {r}", True, Text.assemble((f"{r:<12}", f"bold {BLUE}"), (d, DIM)))
                    for r, d in (("accused", "Defence co-counsel: missing ingredients, bail, FIR flaws"),
                                 ("complainant", "Prosecution co-counsel: proof of charges, oppose bail, s.311/216"),
                                 ("neutral", "Neutral judicial overview: balanced 5-part summary"))
                    if a in r]
        if cmd in ("/evidence", "/proceedings"):
            if cmd == "/evidence" and self.result:
                names = list(self.result["evidence"])
            else:
                try:
                    from procedural_kb import IPC_PROCEDURAL_DATA
                    names = list(IPC_PROCEDURAL_DATA)
                except Exception:
                    names = []
            return [Suggestion(f"{cmd} {n.replace('Section ', '')}", True,
                               Text.assemble((f"{n:<14}", f"bold {BLUE}"), ((self.result or {}).get("statutes", {}).get(n, "") if cmd == "/evidence" else "", DIM)))
                    for n in names if a.replace("section", "").strip() in n.lower()]
        if cmd in ("/open", "/load"):
            return self._path_suggestions(arg.strip().strip("'\""))
        return []

    def _path_suggestions(self, arg: str) -> List[Suggestion]:
        out: List[Suggestion] = []
        try:
            if not arg:                                              # newest judgments in Downloads and here
                files: List[Path] = []
                for d in (Path.home() / "Downloads", Path.cwd()):
                    if d.is_dir():
                        ok = (".pdf", ".txt") if d != Path.cwd() else (".pdf",)
                        files += [p for p in d.iterdir() if p.suffix.lower() in ok and p.is_file() and p.stat().st_size > 0]
                files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                for p in files[:8]:
                    out.append(self._file_sugg(p))
                out.append(Suggestion("/open ~/Downloads/", False, Text.assemble(("  ~/Downloads/", f"bold {BLUE}"), ("  browse a folder", DIM))))
                return out
            p = Path(arg).expanduser()
            folder, prefix = (p, "") if arg.endswith("/") else (p.parent, p.name.lower())
            if not folder.is_dir():
                return []
            for q in sorted(folder.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if q.name.startswith(".") or not q.name.lower().startswith(prefix):
                    continue
                if q.is_dir():
                    out.append(Suggestion(f"/open {self._tilde(q)}/", False, Text.assemble(("  📁 ", ""), (q.name + "/", f"bold {BLUE}"))))
                elif q.suffix.lower() in (".pdf", ".txt"):
                    out.append(self._file_sugg(q))
                if len(out) >= 40:
                    break
        except OSError:
            return []
        return out

    @staticmethod
    def _tilde(p: Path) -> str:
        s = str(p)
        h = str(Path.home())
        return "~" + s[len(h):] if s.startswith(h) else s

    def _file_sugg(self, p: Path) -> Suggestion:
        kb = p.stat().st_size / 1024
        age = time.strftime("%d %b", time.localtime(p.stat().st_mtime))
        return Suggestion(f"/open {self._tilde(p)}", True, Text.assemble(("  📄 ", ""), (f"{short_name(p.name, 40):<42}", HI), (f"{kb:>7.0f} KB  {age}", DIM)))

    def _refresh_menu(self, text: str) -> None:
        menu, sep = self.query_one("#menu", OptionList), self.query_one("#sep", Static)
        if text == self._dismissed:
            self.sugg = []
        else:
            self.sugg = self._suggest(text)
        menu.clear_options()
        if not self.sugg:
            menu.set_class(False, "on"); sep.set_class(False, "on")
            return
        menu.add_options([Option(s.prompt) for s in self.sugg])
        menu.highlighted = 0
        menu.set_class(True, "on"); sep.set_class(True, "on")

    @on(Input.Changed, "#input")
    def _changed(self, ev: Input.Changed) -> None:
        if ev.value != self._dismissed:
            self._dismissed = ""
        if ev.value.lower().startswith("/model") and not self.models_cache:
            self._load_models()
        self._refresh_menu(ev.value)

    @work(thread=True, exclusive=True, group="models")
    def _load_models(self) -> None:
        try:
            self.models_cache = self.engine.models()
            self.call_from_thread(self._refresh_menu, self.query_one("#input", Composer).value)
        except Exception:                                                       # noqa: BLE001
            pass

    def menu_nav(self, d: int) -> None:
        menu = self.query_one("#menu", OptionList)
        if self.sugg:
            n = len(self.sugg)
            menu.highlighted = ((menu.highlighted or 0) + d) % n
            return
        if self.history:                                             # no menu: shell-style history
            self.hist_i = max(0, min(len(self.history), self.hist_i + d))
            inp = self.query_one("#input", Composer)
            inp.value = self.history[self.hist_i] if self.hist_i < len(self.history) else ""
            inp.cursor_position = len(inp.value)

    def menu_close(self) -> None:
        inp = self.query_one("#input", Composer)
        self._dismissed = inp.value
        self._refresh_menu(inp.value)

    def menu_complete(self, run: bool) -> bool:
        """Fill the input with the highlighted suggestion (and submit it if `run` and the suggestion is complete)."""
        if not self.sugg:
            return False
        s = self.sugg[self.query_one("#menu", OptionList).highlighted or 0]
        inp = self.query_one("#input", Composer)
        inp.value = s.fill
        inp.cursor_position = len(s.fill)
        if run and s.run:
            self._submit(s.fill.strip())
            return True
        return True

    @on(OptionList.OptionSelected, "#menu")
    def _clicked(self, ev: OptionList.OptionSelected) -> None:
        self.query_one("#menu", OptionList).highlighted = ev.option_index
        self.menu_complete(run=True)

    # ───────────── input ─────────────
    @on(Input.Submitted, "#input")
    def _enter(self, ev: Input.Submitted) -> None:
        text = ev.value.strip()
        if self.sugg:
            s = self.sugg[self.query_one("#menu", OptionList).highlighted or 0]
            if s.fill.strip() != text:                               # Enter on a different highlighted item completes it
                self.menu_complete(run=True)
                return
        if text:
            self._submit(text)

    def _submit(self, raw: str) -> None:
        inp = self.query_one("#input", Composer)
        inp.value = ""
        self._dismissed = ""
        self._refresh_menu("")
        if not raw:
            return
        self.history.append(raw)
        self.hist_i = len(self.history)
        cmd = raw.split()[0].lower() if raw.startswith("/") else ""
        if self.busy and cmd not in READ_ONLY:
            return self.add_note("still working: wait for the current task to finish (read-only commands like /route and /sections work now)", AMBER)
        cand = raw.strip("'\" ")
        if not raw.startswith("/") and len(cand) < 400 and re.search(r"\.(pdf|txt)$", cand, re.I) and Path(cand).expanduser().exists():
            return self.open_file(cand)                              # a dropped file path
        if raw.startswith("/"):
            return self.command(raw)
        if len(raw) > 1500:
            self.add_user(raw[:140] + " …")
            return self.analyse(text=raw, label=f"pasted text ({len(raw)} chars)")
        self.add_user(raw)
        self.chat(raw)

    # ───────────── actions ─────────────
    def action_open(self) -> None:
        inp = self.query_one("#input", Composer)
        inp.value = "/open "
        inp.cursor_position = 6
        inp.focus()

    def action_paste(self) -> None:
        self.push_screen(PasteModal(), lambda t: t and self.analyse(text=t, label=f"pasted text ({len(t)} chars)"))

    def action_clear(self) -> None:
        self._chat().remove_children()
        if hasattr(self.engine, "clear_chat"):
            self.engine.clear_chat()

    def action_help(self) -> None:
        t = Text()
        for c, a, d, _ in COMMANDS:
            t.append(f"{c:<13}", style=f"bold {BLUE}"); t.append(f"{a:<13}", style=DIM); t.append(d + "\n", style=HI)
        t.append("\nKeys  ", style=f"bold {HI}")
        t.append("↑↓ menu / history · Tab complete · Enter run · Esc close menu · Ctrl+O open · Ctrl+P paste · Ctrl+L clear · Ctrl+Q quit\n", style=DIM)
        t.append("Anything else you type is a question about the loaded judgment. A Bayesian router decides whether to Serve it from the judgment, "
                 "Search the wider library, or Clarify with you.", style=DIM)
        self.add_bot(t)

    def open_file(self, path: str) -> None:
        p = Path(path.strip("'\" ")).expanduser()
        if not p.is_file():
            return self.add_err(f"File not found: {p}")
        self.add_user(f"/open {self._tilde(p)}")
        self.analyse(path=str(p), label=p.name)

    def command(self, line: str) -> None:
        cmd, _, arg = line.partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        try:
            if cmd in ("/open", "/load"):
                return self.open_file(arg) if arg else self.action_open()
            self.add_user(line)
            if cmd in ("/paste", "/text"):
                return self.action_paste()
            if cmd == "/explain":
                return self.explain()
            if cmd == "/sections":
                return self.add_bot(render_sections(self.result)) if self.result else self.add_note("no judgment open: /open <path>")
            if cmd == "/candidates":
                return self.add_bot(render_candidates(self.result)) if self.result else self.add_note("no judgment open: /open <path>")
            if cmd == "/evidence":
                return self.add_bot(self._evidence(arg))
            if cmd == "/passages":
                return self.add_bot(Text((self.result or {}).get("passages", "No judgment open."), style=DIM))
            if cmd == "/route":
                return self.add_bot(self._route())
            if cmd == "/model":
                return self.model_cmd(arg)
            if cmd == "/verifier":
                r = self.engine.set_verifier(arg or "llama")
                return self.add_note(f"verifier → {r['verifier']} (applies to the next analysis)", GREEN)
            if cmd == "/role":
                r = self.engine.set_role(arg or "accused")
                return self.add_note(f"advocate perspective → {r['role']} ({'defence co-counsel' if r['role']=='accused' else 'prosecution co-counsel' if r['role']=='complainant' else 'neutral'})", GREEN)
            if cmd == "/kb":
                return self.kb(arg) if arg else self.add_note("usage: /kb <query>")
            if cmd == "/proceedings":
                return self.add_bot(self.md(self._proceedings(arg)))
            if cmd == "/gpu":
                return self.add_bot(self._gpu())
            if cmd in ("/clear", "/cls"):
                return self.action_clear()
            if cmd in ("/help", "/?"):
                return self.action_help()
            if cmd in ("/quit", "/exit"):
                return self.exit()
            self.add_note(f"unknown command {cmd}: type / to see them", AMBER)
        except Exception as e:                                                  # noqa: BLE001
            self.add_err(f"{cmd} failed: {type(e).__name__}: {e}")

    # ───────────── pure renderers ─────────────
    def _evidence(self, arg: str):
        r = self.result
        if not r:
            return Text("No judgment open.", style=DIM)
        key = arg if arg.lower().startswith("section") else f"Section {arg}"
        m = next((s for s in r["evidence"] if s.lower() == key.lower()), None)
        if not m:
            return Text(f"No evidence stored for {key}. Candidates: {', '.join(r['evidence'])}", style=DIM)
        parts = [Text(f"Evidence for {m}", style=f"bold {HI}")]
        parts += [item(f"[{e['id']}]", f"“{' '.join(e['text'].split())}”", text_style=f"italic {HI}") for e in r["evidence"][m]]
        return Group(*parts)

    def _route(self):
        r = self.last_route
        if not r:
            return Text("No question has been routed yet.", style=DIM)
        d, col = r["distribution"], ROUTE_COL.get(r["route"], HI)
        parts: List = [Text.assemble(("Router: ", f"bold {HI}"), (r["route"], f"bold {col}"), (f"   risk of a wrong answer {r['p_fail']:.2f}  ", HI), bar(r["p_fail"], 12)),
                       Text(f"  Serve {d['Serve']:.2f} · Search {d['Search']:.2f} · Clarify {d['Clarify']:.2f}   best passage relevance {r['best_relevance']}", style=DIM),
                       Text(f"  query used: {r['query']}", style=DIM),
                       Text("  signals in trouble: " + (", ".join(r["high_signals"]) or "none"), style=DIM),
                       Text("  blame if wrong: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(r["blame"].items(), key=lambda x: -x[1])), style=DIM)]
        for p in r["passages"]:
            tag = f"[{p['id']}]" if p["where"] == "judgment" else f"{p['id'][:26]}"
            parts.append(item(tag, clip(p["text"], 200), p["relevance"]))
        return Group(*parts)

    def _proceedings(self, arg: str) -> str:
        from procedural_kb import format_proceedings_text, get_procedural_summary
        if not arg:
            return "usage: `/proceedings 302`"
        name = arg if arg.lower().startswith("section") else f"Section {arg.upper()}"
        kb = get_procedural_summary(name)
        return format_proceedings_text(name, kb) if kb else f"No procedural entry for `{name}`."

    def _gpu(self):
        from ipc.engine import gpu_stats
        g = gpu_stats()
        if not g["ok"]:
            return Text("No NVIDIA GPU detected.", style=DIM)
        return Group(Text(g["name"], style=f"bold {HI}"), Text.assemble(("  VRAM  ", DIM), bar(g["pct"] / 100, 20), (f"  {g['used_gb']} / {g['total_gb']} GB", HI)),
                     Text(f"  {g['temp']}°C · utilisation {g['util']}%", style=DIM))

    # ───────────── workers ─────────────
    @work(thread=True, exclusive=True, group="job")
    def model_cmd(self, arg: str) -> None:
        try:
            self.models_cache = self.engine.models()
            if arg:
                self.engine.set_model(arg)
                self.models_cache = self.engine.models()
                self.call_from_thread(self.add_note, f"chat model → {arg}", GREEN)
            else:
                t = Text("Chat models\n", style=f"bold {HI}")
                for m in self.models_cache:
                    t.append("  ● " if m["active"] else "    ", style=GREEN)
                    t.append(f"{m['label'][:52]:<54}", style=HI if m["active"] else DIM)
                    t.append(f"{m.get('size_gb', '')} GB\n" if m.get("size_gb") else "\n", style=DIM)
                t.append("  switch with /model <name> · add more with `ollama pull <name>`", style=DIM)
                self.call_from_thread(self.add_bot, t)
            self._poll()
        except Exception as e:                                                  # noqa: BLE001
            self.call_from_thread(self.add_err, f"/model failed: {e}")

    @work(thread=True, exclusive=True, group="job")
    def kb(self, query: str) -> None:
        self.call_from_thread(self.set_busy, True, "searching the library")
        try:
            hits = self.engine.kb_search(query, 6)
            parts: List = [Text(f"Library · “{query}”", style=f"bold {HI}")]
            parts += [item(h["id"][:34], clip(h["text"], 220), h["relevance"]) for h in hits]
            self.call_from_thread(self.add_bot, Group(*parts) if hits else Text("Nothing relevant in the library.", style=DIM))
        except Exception as e:                                                  # noqa: BLE001
            self.call_from_thread(self.add_err, f"/kb failed: {e}")
        finally:
            self.call_from_thread(self.set_busy, False)

    def _on_stage(self, ev: dict) -> None:
        n = ev["name"]
        if ev["status"] == "start":
            self._stages[n] = ("run", time.time(), None)
        else:
            self._stages[n] = ("done", 0.0, ev.get("ms"))

    def _stream(self, gen, label: str) -> str:
        """Drive an event generator from a worker thread; stream tokens into one answer."""
        body, buf, last = None, [], 0.0
        for ev in gen:
            t = ev["type"]
            if t == "stage":
                self.call_from_thread(self._on_stage, ev)
            elif t == "result":
                self.result = ev["result"]
            elif t == "route":
                self.last_route = ev
                col = ROUTE_COL.get(ev["route"], HI)
                self.call_from_thread(self.add_note, Text.assemble(("  ↳ ", DIM), (ev["route"], f"bold {col}"), (f"  risk {ev['p_fail']:.2f} · {len(ev['passages'])} passages · best {ev['best_relevance']}", DIM)))
            elif t == "token":
                buf.append(ev["text"])
                if body is None:
                    body = self.call_from_thread(self.add_bot, self.md("".join(buf)))
                elif time.time() - last > 0.12:
                    last = time.time()
                    self.call_from_thread(self._update, body, self.md("".join(buf)))
            elif t == "revise":                   # grounding check: the corrected answer replaces the draft
                buf.clear()
                if body is not None:
                    self.call_from_thread(self._update, body, Text("  checking the answer against the judgment…", style=DIM))
            elif t == "verify":
                self.call_from_thread(self.add_note, "  ⚠ some statements could not be matched to this court's own paragraphs; check the cited passages")
            elif t == "error":
                self.call_from_thread(self.add_err, f"{label} failed: {ev['message']}")
            elif t == "done" and body is not None:
                st = ev.get("stats", {})
                self.call_from_thread(self._update, body, self.md("".join(buf)))
                if st.get("model") and st["model"] != "none":
                    bits = [str(st["model"]).replace("mannix/", ""), f"first token {st.get('first_token_s', '–')}s"] + ([f"{st['tokens_per_s']} tok/s"] if st.get("tokens_per_s") else [])
                    self.call_from_thread(self.add_note, "  " + " · ".join(bits))
        return "".join(buf)

    def _update(self, w: Static, r) -> None:
        w.update(r)
        self._chat().scroll_end(animate=False)

    @work(thread=True, exclusive=True, group="job")
    def analyse(self, path: Optional[str] = None, text: Optional[str] = None, label: str = "judgment") -> None:
        self.call_from_thread(self.set_busy, True, f"analysing {short_name(label)}")
        try:
            self._stream(self.engine.analyse(path=path, text=text, name=label), "analysis")
            r = self.result
            if r:
                self.call_from_thread(self.add_bot, Group(render_sections(r), Text(""), render_stats(r),
                                                          Text("  /explain for the full explanation · /candidates for every checked section · or ask a question", style=DIM)))
        except Exception as e:                                                  # noqa: BLE001
            self.call_from_thread(self.add_err, f"Analysis failed: {type(e).__name__}: {e}")
        finally:
            self.call_from_thread(self.set_busy, False)
            self._poll()

    @work(thread=True, exclusive=True, group="job")
    def explain(self) -> None:
        self.call_from_thread(self.set_busy, True, "writing the explanation")
        try:
            self._stream(self.engine.explain(), "explanation")
        except Exception as e:                                                  # noqa: BLE001
            self.call_from_thread(self.add_err, f"Explanation failed: {e}")
        finally:
            self.call_from_thread(self.set_busy, False)

    @work(thread=True, exclusive=True, group="job")
    def chat(self, q: str) -> None:
        self.call_from_thread(self.set_busy, True, "routing and answering")
        try:
            self._stream(self.engine.chat(q), "chat")
        except Exception as e:                                                  # noqa: BLE001
            self.call_from_thread(self.add_err, f"Chat failed: {e}")
        finally:
            self.call_from_thread(self.set_busy, False)


def main() -> None:
    ap = argparse.ArgumentParser(description="IPC Legal Assistant terminal UI")
    ap.add_argument("--local", action="store_true", help="load the models in this process instead of attaching to a server")
    ap.add_argument("--url", default="http://127.0.0.1:7860", help="server to attach to")
    ap.add_argument("--model", default=None, help="chat model: builtin | none | an `ollama list` name (in-process mode)")
    ap.add_argument("--verifier", default=None, choices=["llama", "none"], help="in-process mode: Hyena only (none) or Llama + Hyena")
    ap.add_argument("--embed", default=None, help="search the offline library for this text and exit (CPU only)")
    ap.add_argument("--no-serve", action="store_true", help="in-process mode: do not also serve the web app on the port")
    a = ap.parse_args()

    if a.embed:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from ipc.retriever import BayesianRetriever
        br = BayesianRetriever()
        qv = br.embed([a.embed])[0]
        for h in br.search_wide(a.embed, qv, 8):
            print(f"{h['relevance']:.2f}  {h['id'][:40]:40s} {h['text'][:110]!r}")
        return

    from ipc.client import HttpEngine
    if not a.local and HttpEngine.alive(a.url):
        engine = HttpEngine(a.url)
        print(f"attached to {a.url}")
    else:
        from ipc.engine import Engine
        print("loading the models in this process (run ./run_app.sh in another terminal to share one copy) …")
        engine = Engine(model=a.model, verifier=a.verifier).load()
        if not a.no_serve:                       # one copy of the models for everything: this process also serves the web app
            try:
                import app as server
                port = int(a.url.rsplit(":", 1)[-1].strip("/")) if a.url.rsplit(":", 1)[-1].strip("/").isdigit() else 7860
                engine.web_url = f"http://127.0.0.1:{port}" if server.serve_in_thread(engine, port) else None
            except Exception:                                                   # noqa: BLE001
                engine.web_url = None
    LegalTUI(engine).run()


if __name__ == "__main__":
    main()
