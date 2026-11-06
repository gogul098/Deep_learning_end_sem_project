"""
tui.py · Claude Code / OpenCode styled Chat TUI for IPC Legal Assistant.
Features:
  • Interactive conversational chat stream explaining IPC sections, clauses, and CrPC/BNSS court proceedings
  • Live GPU corner monitor showing VRAM usage, temperature, and utilization (NVIDIA RTX 3050)
  • Natural LLM text augmentation powered by Llama-3.2-3B
  • Text embedding via InLegalBERT
  • Claude Code slash commands (/open, /embed, /proceedings, /clauses, /clear, /help, /gpu)
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List, Any

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll, Container
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.message import Message
from textual.widgets import (
    Header, Footer, Static, Input, Button, Label,
    LoadingIndicator, Markdown, TextArea,
)
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown

HERE = Path(__file__).resolve().parent

# ═══════════════════════════════ Filename & Text Helpers ═══════════════════════════════
def truncate_filename(filename: str, max_len: int = 32) -> str:
    """Middle-truncate long filenames or SHA256 hashes (e.g. e13d3a51...1901352.pdf)."""
    if len(filename) <= max_len:
        return filename
    stem, ext = os.path.splitext(filename)
    if len(ext) > 8:
        ext = ""
        stem = filename
    keep = (max_len - len(ext) - 3) // 2
    if keep <= 3:
        return filename[:max_len-3] + "..."
    return f"{stem[:keep]}...{stem[-keep:]}{ext}"


# ═══════════════════════════════ GPU Monitor ═══════════════════════════════
def get_gpu_stats() -> Dict[str, Any]:
    """Query NVIDIA GPU stats via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.5
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 5:
            used_mb = float(parts[1])
            total_mb = float(parts[2])
            pct = (used_mb / total_mb) * 100 if total_mb else 0
            return {
                "name": parts[0],
                "used_gb": used_mb / 1024.0,
                "total_gb": total_mb / 1024.0,
                "pct": pct,
                "temp": int(parts[3]),
                "util": int(parts[4]),
                "ok": True
            }
    except Exception:
        pass
    return {"name": "NVIDIA GPU", "used_gb": 0, "total_gb": 0, "pct": 0, "temp": 0, "util": 0, "ok": False}


# ═══════════════════════════════ Backend Service ═══════════════════════════════
class BackendClient:
    """Connects to Gradio backend at http://127.0.0.1:7860 or runs locally."""

    def __init__(self, url: str = "http://127.0.0.1:7860", local: bool = False):
        self.url = url.rstrip("/")
        self.local = local
        self._local_predict = None
        self._local_info = None

    def _init_local(self):
        if self._local_predict is None:
            from local_model import load
            self._local_predict, self._local_info = load()

    def analyse(self, pdf_path: Optional[str], text: str, hide_cited: bool = True) -> Dict[str, Any]:
        """Analyze document and return structured result + conversational briefing."""
        if self.local:
            self._init_local()
            from local_model import read_document
            from augment import format_conversational_briefing
            doc = read_document(pdf_path) if pdf_path else text
            r = self._local_predict(doc, hide_cited=hide_cited)
            briefing = format_conversational_briefing(r, llm=self._local_info.get("llm"), tokenizer=self._local_info.get("tok"))
            r["chat_briefing"] = briefing
            return r
        else:
            from gradio_client import Client, handle_file
            client = Client(self.url, verbose=False)
            file_arg = handle_file(pdf_path) if pdf_path else None
            # Gradio API endpoint returns [briefing_md, table_html, passages, (optional)]
            res = client.predict(file_arg, text or "", hide_cited, api_name="/analyse")
            briefing_md = res[0]
            table_html = res[1] if len(res) > 1 else ""
            passages = res[2] if len(res) > 2 else ""
            raw_json_str = res[3] if len(res) > 3 else ""
            data = None
            if raw_json_str:
                try:
                    data = json.loads(raw_json_str)
                except Exception:
                    pass
            if not data:
                secs = list(dict.fromkeys(re.findall(r"Section\s+\d+[A-Za-z]?", table_html)))
                data = {
                    "sections": secs,
                    "evidence": {},
                    "passages": passages,
                    "table_html": table_html,
                }
            data["chat_briefing"] = briefing_md
            return data

    def chat_query(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Ask conversational question about loaded document or criminal procedure."""
        if self.local:
            self._init_local()
            from augment import conversational_qa
            return conversational_qa(message, context, llm=self._local_info.get("llm"), tokenizer=self._local_info.get("tok"))
        else:
            try:
                from gradio_client import Client
                client = Client(self.url, verbose=False)
                raw_json_str = json.dumps(context) if context else ""
                # Call /chat endpoint
                res = client.predict(message, [], raw_json_str, api_name="/chat")
                # res is [history, cleared_input]
                history = res[0]
                if history and len(history) > 0:
                    return history[-1][1]
            except Exception:
                pass
            # Fallback to local procedural module if client call fails
            from augment import conversational_qa
            return conversational_qa(message, context)


# ═══════════════════════════════ Embedding Engine ═══════════════════════════════
class EmbeddingEngine:
    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or str(HERE / "models")
        self._embedder = None
        self._statute_desc = None
        self._desc_embeddings = None

    def _load(self):
        if self._embedder is not None:
            return
        import torch
        paths = json.load(open(os.path.join(self.models_dir, "paths.json")))
        encoder_path = paths.get("legal_encoder", "inlegalbert")
        if not os.path.isabs(encoder_path):
            encoder_path = os.path.join(self.models_dir, encoder_path)

        sys.path.insert(0, os.path.join(self.models_dir, paths.get("final", "final")))
        from legal_hybrid import SentenceEmbedder
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and torch.cuda.get_device_properties(0).total_memory < 7 * 1024**3:
            device = "cpu"
        self._embedder = SentenceEmbedder.load(encoder_path, device)

        ck_path = os.path.join(self.models_dir, paths.get("final", "final"), "hybrid_model.pt")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        self._statute_desc = ck.get("statute_desc", {})
        self._label_names = ck.get("label_names", [])

    def find_similar_sections(self, text: str, top_k: int = 8) -> List[dict]:
        import torch
        self._load()
        if self._desc_embeddings is None:
            descs, keys = [], []
            for sec in self._label_names:
                d = self._statute_desc.get(sec, "")
                if d:
                    descs.append(f"{sec}: {d}")
                    keys.append(sec)
            if descs:
                with torch.no_grad():
                    self._desc_embeddings = (keys, self._embedder(descs))

        if self._desc_embeddings is None:
            return []

        keys, desc_vecs = self._desc_embeddings
        with torch.no_grad():
            query_vec = self._embedder([text])
            sims = torch.nn.functional.cosine_similarity(query_vec, desc_vecs)
            topk = sims.topk(min(top_k, len(keys)))

        results = []
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            sec = keys[idx]
            results.append({
                "section": sec,
                "description": self._statute_desc.get(sec, ""),
                "similarity": round(score, 4),
            })
        return results


# ═══════════════════════════════ File Open Modal ═══════════════════════════════
class FileOpenModal(ModalScreen[str]):
    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("📂 [bold cyan]Open Court Judgment[/bold cyan] (PDF / TXT)", id="modal-title")
            yield Static("Enter path to judgment file:", id="modal-desc")
            yield Input(placeholder="~/Downloads/judgment.pdf", id="modal-input")
            with Horizontal(id="modal-actions"):
                yield Button("Load & Analyze", variant="primary", id="modal-ok")
                yield Button("Cancel", variant="default", id="modal-cancel")

    @on(Button.Pressed, "#modal-ok")
    def confirm(self) -> None:
        val = self.query_one("#modal-input", Input).value.strip()
        self.dismiss(val)

    @on(Button.Pressed, "#modal-cancel")
    def cancel(self) -> None:
        self.dismiss("")

    @on(Input.Submitted, "#modal-input")
    def submit(self) -> None:
        self.confirm()


# ═══════════════════════════════ Multiline Paste Modal ═══════════════════════════════
class PasteModal(ModalScreen[str]):
    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box-large"):
            yield Label("📋 [bold cyan]Paste Judgment Text[/bold cyan]", id="modal-title")
            yield Static("Paste the judgment or facts text below (press Analyze when done):", id="modal-desc")
            from textual.widgets import TextArea
            yield TextArea(id="paste-textarea")
            with Horizontal(id="modal-actions"):
                yield Button("⚡ Analyze Text", variant="primary", id="paste-ok")
                yield Button("Cancel", variant="default", id="paste-cancel")

    @on(Button.Pressed, "#paste-ok")
    def confirm(self) -> None:
        from textual.widgets import TextArea
        text = self.query_one("#paste-textarea", TextArea).text.strip()
        self.dismiss(text)

    @on(Button.Pressed, "#paste-cancel")
    def cancel(self) -> None:
        self.dismiss("")


# ═══════════════════════════════ Expandable Chat Input Widget ═══════════════════════════════
class ExpandableChatInput(TextArea):
    """Multiline chat input that auto-expands from 3 to 7 lines and submits on Enter."""

    class Submitted(Message):
        def __init__(self, value: str):
            super().__init__()
            self.value = value

    BINDINGS = [
        Binding("enter", "submit_input", "Submit", show=False, priority=True),
        Binding("shift+enter", "insert_newline", "New Line", show=False, priority=True),
        Binding("ctrl+j", "insert_newline", "New Line", show=False, priority=True),
    ]

    def action_submit_input(self) -> None:
        val = self.text.strip()
        if val and not self.disabled:
            self.post_message(self.Submitted(val))
            self.text = ""
            self._update_height()

    def action_newline(self) -> None:
        if not self.disabled:
            self.insert("\n")
            self._update_height()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._update_height()

    def _update_height(self) -> None:
        lines = len(self.text.splitlines()) or 1
        target_h = max(3, min(lines + 1, 7))
        self.styles.height = target_h
        if self.parent:
            self.parent.styles.height = target_h + 1


# ═══════════════════════════════ CSS Styling (Claude Code Aesthetic) ═══════════════════════════════
CLAUDE_CODE_CSS = """
Screen {
    background: #0d1117;
    color: #c9d1d9;
}

FileOpenModal, PasteModal {
    align: center middle;
}

#top-header {
    dock: top;
    height: 3;
    background: #161b22;
    border-bottom: solid #30363d;
    padding: 0 1;
    align: left middle;
}

#brand-title {
    color: #38bdf8;
    text-style: bold;
    width: 1fr;
    content-align: left middle;
    height: 100%;
}

#gpu-badge {
    color: #3fb950;
    text-style: bold;
    width: auto;
    background: #0d1117;
    padding: 0 1;
    border: round #30363d;
    content-align: right middle;
}

#chat-scroll {
    height: 1fr;
    padding: 1 2;
    scrollbar-gutter: stable;
}

#collapsed-header {
    height: 1;
    color: #8b949e;
    padding: 0 1;
    margin: 0 0 1 0;
    border-bottom: dashed #30363d;
}

.message-card {
    margin: 1 0;
    padding: 1 2;
    border-left: thick #30363d;
}

.user-card {
    background: #161b22;
    border-left: thick #38bdf8;
    padding: 1 2;
    color: #c9d1d9;
}

.assistant-card {
    background: #0d1117;
    border-left: thick #a855f7;
    padding: 1 2;
    color: #f0f6fc;
}

.system-card {
    background: #161b22;
    border-left: thick #d29922;
    padding: 1 2;
    color: #8b949e;
}

.error-card {
    background: #1c1313;
    border-left: thick #f85149;
    padding: 1 2;
    color: #ff7b72;
}

#loading-bar {
    height: 1;
    padding: 0 2;
    background: #161b22;
    color: #38bdf8;
    display: none;
    border-top: solid #30363d;
}

#loading-bar.visible {
    display: block;
}

#prompt-container {
    dock: bottom;
    height: 4;
    min-height: 4;
    max-height: 8;
    background: #161b22;
    border-top: solid #30363d;
    padding: 0 1;
    align: left middle;
}

#prompt-prefix {
    width: 3;
    color: #38bdf8;
    text-style: bold;
    padding-top: 1;
}

#chat-input {
    width: 1fr;
    height: 3;
    min-height: 3;
    max-height: 7;
    background: #0d1117;
    border: solid #30363d;
    color: #f0f6fc;
}

#chat-input:focus {
    border: solid #38bdf8;
}

#modal-box {
    width: 70;
    height: auto;
    background: #161b22;
    border: thick #38bdf8;
    padding: 1 2;
    align: center middle;
}

#modal-box-large {
    width: 85;
    height: 80%;
    background: #161b22;
    border: thick #38bdf8;
    padding: 1 2;
}

#paste-textarea {
    height: 1fr;
    margin: 1 0;
}

#modal-actions {
    height: 3;
    align: right middle;
    margin-top: 1;
}

#modal-actions Button {
    margin-left: 1;
}
"""


WELCOME_BANNER = """\
# ⚖️ Claude Code · Legal IPC Assistant
**Indian Criminal Law Jurisprudence (IPC · CrPC · BNSS)**
Powered by **Hyena Long-Document Encoder + Llama-3.2-3B Verifier** running locally on your **NVIDIA RTX 3050**.

---
### 🚀 How to use this assistant:
- **Type or paste judgment facts** to extract applicable IPC sections, crucial clauses, and trial proceedings.
- `/open <path>` or `Ctrl+O` — Load a Supreme Court / High Court judgment PDF.
- `/paste` — Open a multiline paste window.
- `/proceedings <section>` — View the CrPC/BNSS court proceedings roadmap for any IPC section (e.g. `/proceedings 302`).
- `/embed <text>` — Find similar IPC sections by InLegalBERT text embeddings.
- `/clauses` — View key evidentiary clauses extracted from the loaded document.
- `/help` — View all commands.

*Ask me anything: "What are the bailable clauses?", "How to frame charges u/s 302/34?", or "Explain witness testimony."*
"""


class ClaudeCodeTUI(App):
    """Claude Code styled Terminal UI for IPC Legal Assistant."""

    TITLE = "Claude Code · IPC Legal Assistant"
    SUB_TITLE = "Hyena + Llama-3.2-3B · Offline Indian Criminal Law"
    CSS = CLAUDE_CODE_CSS

    BINDINGS = [
        Binding("ctrl+o", "open_file", "Open PDF", show=True),
        Binding("ctrl+p", "paste_text", "Paste Text", show=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    gpu_info_text: reactive[str] = reactive("Initializing GPU...")
    is_busy: reactive[bool] = reactive(False)

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, backend: BackendClient, embed_engine: Optional[EmbeddingEngine] = None, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.embed_engine = embed_engine or EmbeddingEngine()
        self.last_context: Optional[Dict[str, Any]] = None
        self.chat_history: List[Dict[str, str]] = []
        self._current_task_label: str = "Running model"
        self._busy_start_time: float = 0.0
        self._spinner_step: int = 0
        self._spinner_timer = None
        self._banner_collapsed: bool = False

    def compose(self) -> ComposeResult:
        # Top Header with live GPU corner monitor (high contrast styling)
        with Horizontal(id="top-header"):
            yield Static("● [bold #38bdf8]Claude Code[/] [#6e7681]|[/] [bold #f0f6fc]IPC Legal Assistant[/] [#8b949e](Llama-3.2-3B · 4-bit NF4)[/]", id="brand-title")
            yield Static(self.gpu_info_text, id="gpu-badge")

        # Main Chat Stream
        with VerticalScroll(id="chat-scroll"):
            yield Container(Markdown(WELCOME_BANNER), id="welcome-banner", classes="message-card assistant-card")

        # Loading / Busy indicator with animated spinner and timer
        yield Static("", id="loading-bar")

        # Bottom Input Prompt (expandable multiline)
        with Horizontal(id="prompt-container"):
            yield Static("❯", id="prompt-prefix")
            yield ExpandableChatInput(
                placeholder="Type a legal query, paste judgment facts, or type /help...",
                show_line_numbers=False,
                soft_wrap=True,
                id="chat-input"
            )

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-input", ExpandableChatInput).focus()
        # Set recurring GPU monitoring every 1.5 seconds
        self.set_interval(1.5, self.update_gpu_stats)
        # Recurring spinner animation on main loop every 100ms
        self.set_interval(0.1, self._tick_spinner)
        self.update_gpu_stats()

    def update_gpu_stats(self) -> None:
        stats = get_gpu_stats()
        if stats["ok"]:
            color = "#3fb950" if stats["pct"] < 70 else ("#d29922" if stats["pct"] < 90 else "#f85149")
            txt = f"[{color}]● {stats['name'][:16]} | VRAM: {stats['used_gb']:.2f}/{stats['total_gb']:.2f} GB ({stats['pct']:.0f}%) | {stats['temp']}°C[/{color}]"
        else:
            txt = "[dim]● GPU: Offline / CPU mode[/dim]"
        self.query_one("#gpu-badge", Static).update(txt)

    def watch_is_busy(self, busy: bool) -> None:
        def _update():
            try:
                bar = self.query_one("#loading-bar", Static)
                inp = self.query_one("#chat-input", ExpandableChatInput)
                if busy:
                    self._busy_start_time = time.time()
                    self._spinner_step = 0
                    bar.add_class("visible")
                    inp.disabled = True
                else:
                    bar.remove_class("visible")
                    inp.disabled = False
                    inp.focus()
            except Exception:
                pass

        if threading.current_thread() is not threading.main_thread():
            self.call_from_thread(_update)
        else:
            _update()

    def _tick_spinner(self) -> None:
        if not self.is_busy:
            return
        self._spinner_step += 1
        frame = self.SPINNER_FRAMES[self._spinner_step % len(self.SPINNER_FRAMES)]
        elapsed = time.time() - getattr(self, "_busy_start_time", time.time())
        label = getattr(self, "_current_task_label", "Running neural model on RTX 3050")
        try:
            bar = self.query_one("#loading-bar", Static)
            bar.update(f"{frame} [bold #38bdf8]{label}[/] [#8b949e]({elapsed:.1f}s · RTX 3050 active)[/]")
        except Exception:
            pass

    def collapse_welcome_banner(self) -> None:
        """Auto-collapse onboarding boilerplate banner on first interaction to maximize screen space."""
        if self._banner_collapsed:
            return
        self._banner_collapsed = True
        try:
            banner = self.query_one("#welcome-banner")
            banner.remove()
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            scroll.mount(Static(
                "⚖️ [bold #38bdf8]IPC Legal Assistant[/] [dim]· Hyena Encoder + Llama-3.2-3B Verifier · Ready (F1 for help)[/dim]",
                id="collapsed-header"
            ), before=0)
        except Exception:
            pass

    # ─────────────── Actions & Commands ───────────────
    def action_open_file(self) -> None:
        def callback(path: str):
            if path:
                self.handle_load_file(path)
        self.push_screen(FileOpenModal(), callback=callback)

    def action_paste_text(self) -> None:
        def callback(text: str):
            if text:
                preview = text[:150].strip() + ("…" if len(text) > 150 else "")
                self.add_system_message(f"📋 Pasted judgment facts ({len(text)} chars):\n> *\"{preview}\"*")
                self.trigger_analysis(pdf_path=None, text=text, task_label="Analyzing pasted facts")
        self.push_screen(PasteModal(), callback=callback)

    def action_clear_chat(self) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.remove_children()
        self._banner_collapsed = False
        scroll.mount(Container(Markdown(WELCOME_BANNER), id="welcome-banner", classes="message-card assistant-card"))
        self.chat_history.clear()

    def action_show_help(self) -> None:
        self.add_assistant_message(
            "### 📖 Claude Code Legal Assistant — Command Cheat Sheet\n"
            "- `/open <path>` or `Ctrl+O` — Load & analyze a judgment PDF or TXT\n"
            "- `/paste` or `Ctrl+P` — Open a multiline text paste box\n"
            "- `/proceedings <section>` — Show CrPC/BNSS court proceedings for an IPC section\n"
            "- `/embed <text>` — Find similar IPC sections via InLegalBERT embeddings\n"
            "- `/clauses` — View extracted clauses from the currently loaded judgment\n"
            "- `/gpu` — Display detailed GPU memory and thermal stats\n"
            "- `/clear` or `Ctrl+L` — Reset the chat screen\n"
            "- `/quit` or `Ctrl+Q` — Exit application\n"
            "\n*You can also paste any judgment text or ask follow-up questions directly.*"
        )

    # ─────────────── Chat Input Handler ───────────────
    @on(ExpandableChatInput.Submitted)
    def on_input_submitted(self, event: ExpandableChatInput.Submitted) -> None:
        raw = event.value.strip()
        if not raw or self.is_busy:
            return

        # Process Slash Commands
        if raw.startswith("/"):
            self.handle_slash_command(raw)
            return

        # Normal User Message
        if len(raw) > 220:
            preview = raw[:200].strip() + "\n\n*(…pasted passage continued)*"
            self.add_user_message(preview)
        else:
            self.add_user_message(raw)

        # If it looks like a long judgment or paste (> 150 chars or contains legal keywords), run analysis
        if len(raw) > 200 or any(kw in raw.lower() for kw in ["in the supreme court", "criminal appeal", "the appellant", "the deceased", "prosecution case", "sessions judge"]):
            self.trigger_analysis(pdf_path=None, text=raw, task_label="Analyzing judgment facts")
        else:
            # Interactive conversational query on loaded context
            self.trigger_chat_qa(raw)

    def handle_slash_command(self, cmd_line: str) -> None:
        parts = cmd_line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/open", "/load"):
            if not arg:
                self.action_open_file()
            else:
                self.handle_load_file(arg)
        elif cmd in ("/paste", "/text"):
            self.action_paste_text()
        elif cmd == "/proceedings":
            from procedural_kb import get_procedural_summary, format_proceedings_text
            if not arg:
                self.add_assistant_message("⚠️ Please specify a section, e.g.: `/proceedings 302` or `/proceedings 376`")
                return
            sec_name = f"Section {arg}" if not arg.lower().startswith("section") else arg
            kb = get_procedural_summary(sec_name)
            if kb:
                self.add_assistant_message(format_proceedings_text(sec_name, kb))
            else:
                self.add_assistant_message(f"No specific procedural entry found for `{sec_name}`.")
        elif cmd == "/embed":
            if not arg:
                self.add_assistant_message("⚠️ Please provide text to embed, e.g.: `/embed accused inflicted grievous hurt with sharp weapon`")
                return
            self.trigger_embedding(arg)
        elif cmd == "/clauses":
            if not self.last_context or not self.last_context.get("passages"):
                self.add_assistant_message("⚠️ No judgment has been loaded yet. Use `/open <path>` or paste text first.")
                return
            passages = self.last_context.get("passages", "")
            self.add_assistant_message(f"### 📜 Key Evidentiary Passages from Loaded Judgment\n```\n{passages}\n```")
        elif cmd == "/gpu":
            stats = get_gpu_stats()
            self.add_assistant_message(
                f"### 🖥️ GPU Hardware & VRAM Diagnostics\n"
                f"- **Model**: {stats['name']}\n"
                f"- **VRAM Allocated**: `{stats['used_gb']:.2f} GB` / `{stats['total_gb']:.2f} GB` ({stats['pct']:.1f}%)\n"
                f"- **Temperature**: `{stats['temp']}°C`\n"
                f"- **Utilization**: `{stats['util']}%`\n"
                f"- **Quantization**: 4-bit NormalFloat4 (NF4) with Double Quantization"
            )
        elif cmd in ("/clear", "/cls"):
            self.action_clear_chat()
        elif cmd in ("/help", "/?"):
            self.action_show_help()
        elif cmd in ("/quit", "/exit"):
            self.exit()
        else:
            self.add_assistant_message(f"Unknown command `{cmd}`. Type `/help` for available commands.")

    def handle_load_file(self, path_str: str) -> None:
        p = Path(path_str).expanduser().resolve()
        if not p.exists():
            self.add_error_message(f"### ❌ File Not Found\n`{path_str}` does not exist on disk. Please verify the file path.")
            return
        disp_name = truncate_filename(p.name, max_len=32)
        try:
            size_kb = p.stat().st_size / 1024.0
            size_str = f" ({size_kb:.1f} KB)"
        except Exception:
            size_str = ""
        self.add_system_message(f"📂 Open & analyze: `{disp_name}`{size_str}")
        self.trigger_analysis(pdf_path=str(p), text="", task_label=f"Analyzing {disp_name}")

    # ─────────────── Message Renderers ───────────────
    def add_user_message(self, text: str) -> None:
        self.collapse_welcome_banner()
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        content = f"**❯ User**\n\n{text}"
        scroll.mount(Container(Markdown(content), classes="message-card user-card"))
        scroll.scroll_end(animate=False)

    def add_system_message(self, text: str) -> None:
        self.collapse_welcome_banner()
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        content = f"**⚙️ System**\n\n{text}"
        scroll.mount(Container(Markdown(content), classes="message-card system-card"))
        scroll.scroll_end(animate=False)

    def add_assistant_message(self, markdown_text: str) -> None:
        self.collapse_welcome_banner()
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.mount(Container(Markdown(markdown_text), classes="message-card assistant-card"))
        scroll.scroll_end(animate=False)

    def add_error_message(self, markdown_text: str) -> None:
        self.collapse_welcome_banner()
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.mount(Container(Markdown(markdown_text), classes="message-card error-card"))
        scroll.scroll_end(animate=False)

    # ─────────────── Background Workers ───────────────
    @work(thread=True, exclusive=True, group="analysis")
    def trigger_analysis(self, pdf_path: Optional[str], text: str, task_label: str = "Analyzing judgment") -> None:
        self._current_task_label = task_label
        self.is_busy = True
        try:
            res = self.backend.analyse(pdf_path, text, hide_cited=True)
            self.last_context = res
            briefing = res.get("chat_briefing", "")
            if not briefing:
                from augment import format_conversational_briefing
                briefing = format_conversational_briefing(res)
            self.call_from_thread(self.add_assistant_message, briefing)
        except Exception as e:
            raw_err = str(e)
            if "name 'res' is not defined" in raw_err:
                err_title = "Internal Pipeline State Error"
                err_desc = "A response variable was uninitialized during response parsing."
            elif any(w in raw_err.lower() for w in ["connection refused", "failed to establish", "max retries"]):
                err_title = "Inference Service Unavailable"
                err_desc = "Could not reach the local neural server at `http://127.0.0.1:7860`."
            elif any(w in raw_err.lower() for w in ["cuda out of memory", "oom"]):
                err_title = "GPU Memory Limit Exceeded"
                err_desc = "The RTX 3050 ran out of VRAM while encoding the judgment document."
            elif "not found" in raw_err.lower() or "no such file" in raw_err.lower():
                err_title = "Judgment File Not Found"
                err_desc = "The specified document file could not be read from disk."
            else:
                err_title = "Document Analysis Failed"
                err_desc = "Failed to process the document. The neural backend encountered an unexpected error while reading or analyzing the file."

            friendly_err = (
                f"### ⚠️ {err_title}\n\n"
                f"{err_desc}\n\n"
                f"> **Actionable Troubleshooting:**\n"
                f"> • Verify backend service status: `systemctl --user status ipc-app`\n"
                f"> • Inspect live service logs: `tail -n 20 /tmp/opencode/app.log`\n"
                f"> • Or run in-process directly: `./tui.sh --local`"
            )
            self.call_from_thread(self.add_error_message, friendly_err)
        finally:
            self.is_busy = False

    @work(thread=True, exclusive=True, group="chat")
    def trigger_chat_qa(self, query: str) -> None:
        self._current_task_label = "Consulting Llama-3.2-3B legal advocate"
        self.is_busy = True
        try:
            reply = self.backend.chat_query(query, self.last_context)
            self.call_from_thread(self.add_assistant_message, reply)
        except Exception as e:
            raw_err = str(e)
            if any(w in raw_err.lower() for w in ["connection refused", "failed to establish"]):
                err_msg = "Could not reach the local model backend at `http://127.0.0.1:7860`."
            else:
                err_msg = "An error occurred while generating the advocate advice response."
            friendly_err = (
                f"### ⚠️ Chat Consultation Error\n\n"
                f"{err_msg}\n\n"
                f"> **Actionable Troubleshooting:**\n"
                f"> • Verify backend service status: `systemctl --user status ipc-app`\n"
                f"> • Or run in-process directly: `./tui.sh --local`"
            )
            self.call_from_thread(self.add_error_message, friendly_err)
        finally:
            self.is_busy = False

    @work(thread=True, exclusive=True, group="embed")
    def trigger_embedding(self, query: str) -> None:
        self._current_task_label = f"Embedding query ({len(query)} chars)"
        self.is_busy = True
        try:
            results = self.embed_engine.find_similar_sections(query, top_k=8)
            lines = [f"### 🔤 InLegalBERT Semantic Similarity for: \"{query}\"\n"]
            lines.append("| Rank | Section | Similarity | Description |")
            lines.append("| :---: | :--- | :---: | :--- |")
            for i, r in enumerate(results, 1):
                lines.append(f"| {i} | **{r['section']}** | `{r['similarity']:.4f}` | {r['description'][:75]}... |")
            self.call_from_thread(self.add_assistant_message, "\n".join(lines))
        except Exception as e:
            err = (
                f"### ⚠️ Embedding Computation Error\n\n"
                f"Failed to generate text embeddings for the requested query.\n\n"
                f"> **Actionable Troubleshooting:** Check model files in `models/inlegalbert`."
            )
            self.call_from_thread(self.add_error_message, err)
        finally:
            self.is_busy = False

    @work(thread=True, exclusive=True, group="chat")
    def trigger_chat_qa(self, query: str) -> None:
        self.is_busy = True
        try:
            reply = self.backend.chat_query(query, self.last_context)
            self.call_from_thread(self.add_assistant_message, reply)
        except Exception as e:
            err = f"❌ **Chat Error**: {str(e)}"
            self.call_from_thread(self.add_assistant_message, err)
        finally:
            self.is_busy = False

    @work(thread=True, exclusive=True, group="embed")
    def trigger_embedding(self, query: str) -> None:
        self.is_busy = True
        try:
            results = self.embed_engine.find_similar_sections(query, top_k=8)
            lines = [f"### 🔤 InLegalBERT Semantic Similarity for: \"{query}\"\n"]
            lines.append("| Rank | Section | Similarity | Description |")
            lines.append("| :---: | :--- | :---: | :--- |")
            for i, r in enumerate(results, 1):
                lines.append(f"| {i} | **{r['section']}** | `{r['similarity']:.4f}` | {r['description'][:75]}... |")
            self.call_from_thread(self.add_assistant_message, "\n".join(lines))
        except Exception as e:
            err = f"❌ **Embedding Error**: {str(e)}"
            self.call_from_thread(self.add_assistant_message, err)
        finally:
            self.is_busy = False


# ═══════════════════════════════ Entry Point ═══════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Claude Code styled TUI for IPC Legal Assistant")
    ap.add_argument("--local", action="store_true", help="run in-process using GPU directly without Gradio service")
    ap.add_argument("--url", default="http://127.0.0.1:7860", help="Gradio backend URL (default: http://127.0.0.1:7860)")
    ap.add_argument("--embed", type=str, default=None, help="quick CLI embedding search and exit")
    args = ap.parse_args()

    if args.embed:
        engine = EmbeddingEngine()
        results = engine.find_similar_sections(args.embed, top_k=10)
        c = Console()
        c.print(f"\n[bold cyan]🔤 InLegalBERT Embedding Search for:[/bold cyan] \"{args.embed}\"\n")
        tbl = Table(show_header=True, header_style="bold magenta")
        tbl.add_column("#", justify="center", width=3)
        tbl.add_column("Section", style="bold")
        tbl.add_column("Similarity", justify="right")
        tbl.add_column("Statute Description")
        for i, r in enumerate(results, 1):
            s_style = "green" if r["similarity"] >= 0.7 else ("yellow" if r["similarity"] >= 0.4 else "red")
            tbl.add_row(str(i), r["section"], Text(f"{r['similarity']:.4f}", style=s_style), r["description"][:80])
        c.print(tbl)
        return

    backend = BackendClient(url=args.url, local=args.local)
    app = ClaudeCodeTUI(backend=backend)
    app.run()


if __name__ == "__main__":
    main()
