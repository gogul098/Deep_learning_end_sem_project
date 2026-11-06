# IPC Section Assistant: testing guide

An offline assistant for Indian court judgments. You open a judgment (PDF), and it suggests the IPC sections, explains the case, and answers questions about it, citing the sentences it used.

## What you need

- A Windows or Linux PC with an **NVIDIA GPU, 6 GB VRAM or more** (developed on an RTX 3050 6 GB).
- **Python 3.12**.
- **Ollama**, from https://ollama.com.
- About **15 GB of free disk** for the models.
- **Git LFS**, so that the trained model in `../model_files` is downloaded with the repo (`git lfs install` before cloning, or `git lfs pull` after).
- No Hugging Face token: the trained model is in the repo, and the Llama base (`unsloth/Llama-3.2-3B-Instruct`) and InLegalBERT are public.

## Setup (once)

Run these from inside this folder. The commands below are for Windows PowerShell.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

ollama pull qwen2.5:7b                      # the chat model, ~4.7 GB

python download_model.py                    # links ../model_files + downloads Llama base and InLegalBERT into .\models
python download_reranker.py
```

On Arch Linux, `./setup_arch.sh` and then `./download.sh` do the same.

## Run

```powershell
.venv\Scripts\activate
python tui.py --local
```

The first start takes 30–60 s while the models load.

**In the TUI:**
1. `/open C:\path\to\judgment.pdf` to load a judgment. The analysis takes about 15–30 s.
2. Type questions in plain English, for example:
   - "Who is the complainant?"
   - "List the bail conditions."
   - "Which Supreme Court judgments were cited?"
3. `/explain` gives a summary of the case and the suggested sections.
4. `/role accused | complainant | neutral` changes whose side the answers argue from.
5. `/help` lists all commands.

**Web version instead:** run `python app.py`, then open http://127.0.0.1:7860 in a browser.

## What to expect

- About **85% of answers are correct** on our tests.
- Answers cite the sentences they used, like `[S12: fact]`. Tags mark whether a sentence is this court's own text (`fact`) or a quote from another judgment (`precedent`).
- For a vague question ("what about the other one?"), the assistant asks you to clarify instead of guessing.
- **Known weak spots:**
  - long "list all …" questions can miss items;
  - a fact buried in one sentence of a 300-sentence judgment is sometimes not found;
  - now and then it quotes the wrong passage.
- It's a **reading aid, not legal advice**. Check the cited sentences.

## What to report

For each wrong or strange answer, please send:
1. the judgment (file name, or the PDF itself);
2. your question, exactly as typed;
3. the answer it gave (copy-paste);
4. what the right answer is, and where it appears in the judgment if you know.

Also report crashes (copy the error text) and anything very slow (more than 30 s for one answer).
