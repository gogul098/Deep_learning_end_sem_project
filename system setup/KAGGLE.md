# Learning the router on Kaggle (about 3,000 graded IPC questions)

This builds the IPC version of the prob_reas 6k set, in three steps:

1. **Questions** (`make_questions.py`, CPU notebook): picks about 300 judgments from the AWS Supreme Court corpus and writes about 10 questions per judgment. Every question has a known right answer:
   - Metadata questions: parties, judge, date and outcome.
   - Fact questions: FIR date, police station, sentence, compensation and charges. These are asked only when the judgment states one clear value, and the sentence that holds the answer is saved with the question.
   - Trap questions: ask about something that isn't in the judgment.
   - Chains: three-step conversations with pronoun follow-ups.
   - Vague questions.
2. **Labels** (`label_ipc.py`, GPU notebook): every question goes through the same assistant as your TUI, with Qwen2.5-7B, the reranker and the grounding check.
   - The router records its signals and the route it *would* have taken, but every question is answered.
   - Each answer is then graded automatically.
3. **Learning** (`learn_router.py`, your PC): learns the cut-offs, the network and the serve threshold for InLegalBERT + Qwen2.5-7B. It starts from the prob_reas profile and checks the result before using it.

The labels are valid for your PC because it's the same `qwen2.5:7b` model file. Kaggle is only faster.

---

## 0. One-time preparation (on your PC)

1. **Hugging Face tokens.** Revoke the old token that was in `download_model.py` (it has been removed from the file). Create one new token with **write** access; write access includes read. Your account needs access to `meta-llama/Llama-3.2-3B-Instruct`.
2. **Zip the code.** Zip the *contents* of `ipc-assistant(updated)`, leaving out `.venv`, `models`, `.cache` and `bench_out`. You need `ipc/`, `legal_kb/`, `router_calibration/` and all the `.py` files. In PowerShell, from inside the folder:
   ```powershell
   Get-ChildItem -Exclude .venv,models,.cache,bench_out | Compress-Archive -DestinationPath ..\ipc-code.zip -Force
   ```
3. **Upload the zip as a dataset.** Go to kaggle.com → **Datasets → New Dataset**, upload `ipc-code.zip`, name it `ipc-assistant-code` and keep it private. Kaggle unzips it automatically.
4. **Verify your phone number** on Kaggle (Settings). Without it, notebooks get no GPU and no internet.

## 1. Notebook 1: generate the questions (CPU, about 1 hour)

1. Go to **Code → New Notebook → File → Import Notebook** and choose `kaggle/1_make_questions.ipynb`.
2. In the right panel:
   - **Accelerator: None**
   - **Internet: On**
   - **Add Input:** your `ipc-assistant-code` dataset
3. Click **Save Version → Save & Run All (Commit)**. You can close the browser while it runs.
4. When it finishes, the version's **Output** tab shows `ipc_qa/questions.jsonl` and `ipc_qa/pdfs/`. The last printed cell shows the counts, roughly 3,000 questions from about 300 judgments.

For more questions, use `--n 400` or add years with `--aws-years 2022 2023 2024`.

## 2. Notebook 2: label them (GPU T4 x2, about 7–8 hours)

1. Import `kaggle/2_label_ipc.ipynb`.
2. In the right panel:
   - **Accelerator: GPU T4 x2**
   - **Internet: On**
   - **Add Input:** `ipc-assistant-code`, and **Notebook Output Files** from notebook 1
   - **Add-ons → Secrets:** add `HF_TOKEN` (your write token) and switch it on for this notebook
3. **Run the cells by hand down to and including TRIAL.** The trial labels 2 judgments in about 3 minutes. Check that:
   - the `reply` fields are real answers
   - about half to three-quarters are `correct: true`
   - `route_router` varies

   If something looks wrong, stop here and send me the trial output.
4. **Start the full run in the background:** **Save Version → Save & Run All**.
   - What it costs: about 300 judgments × (15 s to analyse + 11 questions × about 7 s) ≈ 7.5 h. Kaggle allows 12 h per run and 30 GPU hours per week.
   - If the session dies, nothing is lost. Every 50 questions the labels are copied to the private dataset `huggingface.co/datasets/<you>/ipc-router-labels`. Running again picks up where it stopped.
   - Optional, to halve the wall-clock time: set `SHARD = '0/2'` in the first cell and run one version, then `SHARD = '1/2'` and run another. Two notebooks can run side by side if your quota allows. Each shard writes its own file.
5. Expected GPU memory use: Qwen 7B, about 5.5 GB on GPU 1; the verifier, embedder and reranker, about 4 GB on GPU 0. `IPC_KEEP_RESIDENT=1` keeps both loaded, instead of swapping them as your 6 GB card has to.

## 3. Back on your PC: learn the router

1. Download the labels file(s), either from the notebook's Output tab or from your HF dataset, into `labels\`.
2. Look at the numbers first:
   ```powershell
   python learn_router.py labels\ipc_qwen2.5-7b_shard0of1.jsonl --dry
   ```
   With two shards, pass both files.
3. Read the table it prints:
   - **AUROC** of the cross-validated row: 0.5 is chance; above 0.65 means the router has learned something real.
   - **Served / fail among served:** served answers should fail ≤ 20%.
   The cross-validation keeps each judgment's questions within a single fold, so the score isn't inflated.
4. If those checks pass, run it again without `--dry`. That writes `router_calibration\inlegalbert__qwen2.5-7b.json`, which the TUI loads automatically the next time it starts.
5. **Spot-check about 100 labels by hand** before you trust them. Open the `.jsonl` file and compare `reply` with `answers`. The grader is strict: an answer in a different form from the accepted variants counts as wrong. If you find mistakes, put `{"<id>": true/false}` in a JSON file and pass it with `--labels fixes.json`. Your grades override the automatic ones.

## What each label row contains

| field | meaning |
|---|---|
| `raw`, `evidence` | the router's measured signals and their Low/Med/High levels, which are what the network learns from |
| `route_router`, `p_fail` | what the current router would have done |
| `reply`, `correct`, `abstained`, `hallucinated` | the answer and its grade. A "not stated" answer to a real question counts as a failure |
| `gold_retrieved` | whether the sentence holding the answer was among the passages given to the model, i.e. the reranker's recall. This is the number that settles whether a different reranker is worth trying |
| `grounding_revised` | whether the grounding check made the model rewrite its answer |
