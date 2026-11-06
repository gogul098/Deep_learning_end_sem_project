# v8.1 – explanation fixes (case facts by code)

Files changed: `ipc/prompts.py`, `ipc/llm.py`, `ipc/engine.py`, `ipc/pipeline.py`; new `ipc/case_facts.py`.

| Problem in v8 | Fix |
|---|---|
| Charges = every "Section N" in the text, labelled IPC (e.g. "IPC 29, 137, 183, 361, 483") | `case_facts.charges()` reads the "offences punishable under ..." sentence and groups by Act (BNS, IPC, POCSO, SC/ST, ...) |
| Case number / parties "not stated" | `case_facts.case_details()` reads the first-page header from the original PDF text |
| No hearings in the prompt | new PROCEEDINGS block (FIR and dated hearings, quoted precedents skipped) |
| Only the last 260 words of the order -> conditions missing | DISPOSITION = full operative part from the last "Hence / For the reasons / In view of" (up to 3,500 chars, e-signature stamp removed) |
| "Matches registered charge" was a number search in the wrong list | `check_against_charges()` compares with the real charges, including IPC -> BNS (e.g. IPC 376 -> BNS 64(1)(2)) |
| Model invented reasons ("Supported (no evidence against)") | prompt: support needs a quoted [S#]; otherwise "no evidence quoted"; copy every condition |
| Ollama context 4096 could silently cut the prompt | num_ctx 8192, temperature 0, explanation up to 1,100 tokens |

`pipeline.analyse` now keeps `raw_text` in the result (CACHE_VERSION 3, so old cached results are recomputed once).
Not changed: the encoder, verifier, fusion, thresholds and the label set (still IPC-only), and Hyena's evidence sentences.
