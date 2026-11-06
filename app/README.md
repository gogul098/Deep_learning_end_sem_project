# IPC Section Assistant (v8) · offline legal assistant for Arch Linux

Open an Indian judgment (PDF) → the Hyena encoder reads the whole document and proposes IPC sections, Llama-3.2-3B verifies each one,
a chat model of your choice explains the judgment in a fixed four-part format, and you can chat with the document. A Bayesian
retriever decides, for every question, whether to answer from the judgment, search the wider offline library, or ask you to clarify.
After the one-time model download **nothing leaves the machine**.

## Requirements
| | |
|---|---|
| GPU | NVIDIA, 6 GB or more (developed on an RTX 3050 6 GB). Driver: `sudo pacman -S nvidia-open nvidia-utils`, reboot |
| RAM / disk | 16 GB RAM · ~15 GB free disk (models 7.5 GB, 4-bit Llama cache 2.1 GB, Python environment) |
| No NVIDIA GPU? | Works on CPU (≈14 GB RAM, minutes per judgment). The Bayesian retriever is CPU-only in any case |
| Chat models | optional: [Ollama](https://ollama.com) for any model other than the built-in Llama |

## Install (once)
```bash
./setup_arch.sh                         # uv + Python 3.12 venv + PyTorch (CUDA 12.6) + libraries (requirements.txt)
./download.sh                           # links ../model_files (trained model) + downloads Llama + InLegalBERT → ./models (internet once)
```

## Use
```bash
./run_app.sh                            # web app → http://127.0.0.1:7860 · architecture explainer → /architecture
./tui.sh                                # terminal UI (attaches to the running server, else loads the models itself)
./analyse.sh judgment.pdf               # one-shot terminal analysis
./analyse.sh judgment.pdf --explain --model gemma3:4b      # + the four-part explanation from an Ollama model
./analyse.sh ~/cases/ --json out.json   # every PDF in a folder → JSON
```
**One copy of the models, whichever you start first.** `./tui.sh` attaches to a running server; if there is none it loads the models itself
*and serves the web app and the architecture page from the same process* (`--no-serve` turns that off). `./run_app.sh` refuses to start a second
copy when one is already running. A judgment opened in the terminal shows up in the browser, and the live view on `/architecture` shows what
either of them is doing. If loading ever fails for lack of GPU memory, the web page says which kind of program is in the way and has a Retry button.

### Choosing the chat model
The chat model is only used for the explanation and the chat; the section prediction does not depend on it.
* **Any Ollama model**: `ollama pull <tag>`, then pick it in the web app's *Chat model* menu, with `/model <name>` in the TUI, or
  `./run_app.sh --model <name>`. The list is read live from `ollama list`; the choice is remembered (`settings.json`).
* **builtin**: the local Llama-3.2-3B with its verifier LoRA switched off for chat.
* **none**: extractive answers assembled directly from the retrieved passages.

**6 GB note.** The verifier (≈3 GB) and an 8B Ollama model (≈5 GB) do not fit together. The engine frees whichever is not needed
(Ollama unloads before an analysis, the verifier unloads before an Ollama answer), so there is a few seconds' pause when switching.
For the most room, run `--verifier none`: the Hyena encoder alone proposes the sections (0.5 s, 0.4 GB; micro-F1 0.563 against 0.583 for
the full hybrid on the project's test split), and the whole card is left to the chat model.

### The explanation
`/explain` (TUI) or *Explain this judgment* (web) sends the model your template: the system rules (use only the context, quote evidence
exactly as `[S12]`, say "not stated in the judgment", the section list is final) and a context built from the opening of the judgment,
the suggested sections with confidence, statute line and evidence, key passages with sentence ids, and the end of the judgment.
Click a `[S12]` in the web app to read that sentence. If a model stops before part 4, the engine asks it once to finish.

### The Bayesian retriever (`ipc/bayes.py`, `ipc/retriever.py`)
Ported from the `prob_reas_kaggle` project. For every chat question it measures 14 signals, a 20-node Bayesian network (14 inputs → 4
bottlenecks → risk → route) computes P(wrong answer) by exact inference, and the route is **Serve** (< 0.20), **Search** (0.20–0.60) or
**Clarify** (> 0.60). It runs **on the CPU only**: a small CPU copy of InLegalBERT embeds the question in ~60 ms.

What is the same as the original: the network, priors, weights, thresholds, the clarify questions. What had to change to run offline, and what
that means:
* MiniLM embedder + cross-encoder reranker → InLegalBERT (the model `legal_kb` was built with) + BM25; "relevance" is half rescaled cosine and half
  IDF-weighted term coverage. spaCy → regex rules. The distance cut-offs were **re-calibrated** on InLegalBERT cosines (`CUTS` in `ipc/retriever.py`).
* The Search route used live Wikipedia; here it searches the offline `legal_kb` (100 statute cards + 250 passages from 50 judgments) and the CrPC/BNSS cards.
* Passages from *other* cases are dropped when a question is about *this* case ("he", "the applicant"), because a small model otherwise mistakes them for its facts.
* `legal_kb` is small (350 passages). Rebuild a larger one with the original `build_legal_kb.py`.

## Terminal UI
The menu lives in the same box as the cursor. Type `/` and it opens right above what you are typing, filtering as you type;
`↑`/`↓` choose, `Tab` completes, `Enter` runs, `Esc` closes it. The menu follows the command:

| You type | The menu shows |
|---|---|
| `/` | every command with a one-line description |
| `/open ` | your newest judgments in `~/Downloads`, then folders to browse as you type a path |
| `/model ` | the models from `ollama list`, the built-in Llama, and "none" |
| `/evidence ` | the sections found in the open judgment |
| `/verifier ` · `/proceedings ` | `llama` / `none` · the sections with CrPC/BNSS cards |

Other commands: `/explain` `/sections` `/candidates` `/passages` `/route` `/kb <query>` `/paste` `/gpu` `/clear` `/help` `/quit`.
With no menu open, `↑`/`↓` walk through what you typed before. The line under the box shows the chat model, the verifier and the open
judgment; while something runs, one line above the box shows each stage and its time. You can drop a file path into the box to open it.

## Web app and architecture explainer
`http://127.0.0.1:7860` is the app: drop a PDF, watch the stages, read the sections and explanation, chat with route badges.
`/architecture` (button in the header) draws the whole system. **Explain** mode: click any block for what it does, where the code is and
whether it runs on the CPU or GPU, and play with the real 20-node network. **Live** mode: blocks light up as the server works, with real
timings, the shortlist, the router's decision and signals, and the answer speed. It shows runs started from the web app, the TUI, or the page itself.

API (all local): `POST /api/analyse` · `/api/explain` · `/api/chat` (server-sent events) · `GET /api/state` `/api/models` `/api/events` `/api/kb` `/api/graph` ·
`POST /api/model` `/api/verifier` `/api/bayes`.

## Performance (RTX 3050 6 GB, 12-page judgment, 5,622 tokens, measured)
| | v7 | v8 |
|---|---|---|
| InLegalBERT sentence vectors | 18.6 s (forced onto the CPU on 6 GB cards) | 0.25 s (GPU, fp16; same probabilities, cosine 0.9999993) |
| Llama verification (2 candidate orders) | 9.1 s | 5.4 s (shared 1,640-token prefix read once) |
| **Model pipeline** | **22.1 s** | **5.7 s** |
| Upload → useful result in the app | 42–59 s (blocked until the 250-token commentary finished) | 5.7 s for the sections, explanation streams (first token ≈ 4 s, 20 tok/s built-in) |
| Re-opening the same judgment | full re-run | instant (result cache in `.cache/results`) |
| Server start | ≈ 34 s (quantises 6.4 GB each start) | 10–14 s (pre-quantised 4-bit cache `models/llama-4bit`, built on first start) |
| Hyena-only mode | – | 0.56 s, 0.4 GB VRAM |
| Peak VRAM during analysis | 2.98 GB | 3.06 GB |

Predictions are unchanged: Section 376 0.91 → 0.908, Section 34 0.40 → 0.391 on this judgment. The shared-prefix cache moves individual Llama scores by at most
0.017 (bf16 arithmetic), and the section set is identical.

Other fixes: the old Embedding tab called two methods that did not exist; chat generation ran with the yes/no verifier LoRA still active; the checkpoint
and InLegalBERT were each loaded twice; the TUI defined two methods twice and polled `nvidia-smi` on the UI thread; the "proceedings roadmap" was the same
boilerplate for every case.

## Known limits (read these)
* **Chat quality depends on the chat model.** The built-in 3B model sometimes repeats or paraphrases loosely; an 8B Ollama model is better but stops early or
  mislabels the court on some runs. Always check quotes against the `[S..]` sentences. Answers say "not stated in the judgment" when nothing relevant is retrieved.
* One threshold (0.35) is used for all sections (it generalised best on the test split); per-section thresholds are no longer offered.
* `ipc/statutes.py` holds official section headings for the 47 sections the checkpoint has no description for; they were written from general knowledge, so verify them.
* The router's signal extractors are heuristics (see above); expect some questions to be routed more cautiously (Search) than a human would.
* Scanned PDFs need OCR first: `sudo pacman -S ocrmypdf && ocrmypdf in.pdf out.pdf`.

## Tests
```bash
python -m tests.test_bayes [--pgmpy /path/to/venv/bin/python]    # 6 original scenarios + exact-inference cross-check against pgmpy (max diff 5e-5)
CUDA_VISIBLE_DEVICES= python -m tests.test_retriever judgment.pdf # the retriever on CPU only; asserts CUDA is never initialised
python -m tests.test_pipeline judgment.pdf                        # score parity, timing, cache
python -m tests.test_engine judgment.pdf [ollama-model]           # analyse → explain → chat end to end
```

## Files
| | |
|---|---|
| `ipc/pipeline.py` | analysis pipeline (GPU embedder, shared-prefix verifier, 4-bit cache, result cache) |
| `ipc/engine.py` · `ipc/client.py` | the engine every front end uses · the same interface over HTTP |
| `ipc/llm.py` | Ollama / built-in / none chat models |
| `ipc/prompts.py` · `ipc/statutes.py` | the explanation and chat prompts · one-line statute text |
| `ipc/bayes.py` · `ipc/retriever.py` | Bayesian network · CPU retriever, router and clarify loop |
| `app.py` · `web/` | server · web app and architecture page (no external requests) |
| `tui.py` · `analyse.py` | terminal UI · one-shot CLI |
| `legal_kb/` · `procedural_kb.py` | offline library · CrPC/BNSS cards |
| `../model_files/` | the trained model + `legal_hybrid.py`, the exact code it was trained with (linked through `models/paths.json`, imported, not edited) |
| `legacy/` | the v7 `app.py`, `tui.py`, `augment.py` |
| `setup_arch.sh` `download.sh` `ipc-assistant.service` | install, model download, optional systemd user service |

Decision support only: a qualified lawyer must verify every section.
