# Second-opinion brief: IPC Section Assistant (offline legal QA over Indian court judgments)

Paste everything below the line into the other assistant, and attach the files listed at the end.

---

I'm building an **offline legal question-answering assistant for Indian court judgments**. It runs on a laptop with an RTX 3050 (6–8 GB VRAM). I'd like an independent review of the retrieval and answering pipeline: what is likely to give the next real accuracy gain, and what we may be doing wrong. Please be critical. The scores have plateaued, and I suspect we are tuning to a small test set.

## What the system does

A user opens one judgment (PDF) and asks questions about it in a chat ("Who is the complainant?", "List the bail conditions", "Which Supreme Court cases were cited on speedy trial?").

**Pipeline per question:**
1. **Document analysis** (once per judgment): split into sentences (a rule-based splitter), embed each sentence with InLegalBERT (mean-pooled, CPU). A separate trained model predicts the IPC sections, and case facts are extracted by regex: parties, FIR, charges, disposition.
2. **Retrieval** (`ipc/retriever.py`):
   - Hybrid search: cosine on InLegalBERT plus 0.25 × normalised BM25, over the judgment's sentences.
   - The top 30 candidates are re-ranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`), and the best 5 are kept.
   - Neighbour expansion is then added:
     - the sentence after a colon or "as under:-";
     - the sentence that names a quoted case;
     - halves of sentences the splitter cut at "Vs." / "Ors.";
     - whole "(i)…(viii)" lists;
     - "Firstly/Secondly/Thirdly" reasons.
   - List questions use a wider search (up to 20 passages).
3. **Router** (`ipc/learned_bn.py`, `ipc/bayes.py`): a Bayesian network over about 14 signals decides between Serve (answer from the retrieved passages), Search (wider retrieval, still answered) and Clarify (ask the user). The signals include rerank score, vector distance, question length, entity count, unresolved pronouns and domain shift.
   - It was learned from about 2,900 automatically graded answers to generated questions on 300 Supreme Court judgments: information-gain cut-offs, MAP-EM from a prior learned on a 6,000-question general-domain set, and a serve threshold chosen so that served answers fail ≤ 7%.
   - A rule overrides Clarify for clearly specific questions.
4. **Prompting** (`ipc/prompts.py`): every passage is tagged **[fact]** (this court / this case), **[precedent]** (quoted from another judgment) or **[impugned-order]**. The prompt also carries a static case summary and an IPC→BNS mapping table.
5. **Generation**: Qwen2.5-7B-Instruct (Q4, via Ollama, temperature 0, 8k context).
6. **Grounding check** (`ipc/grounding.py`): after generation, the code checks that figures and units appear in the context, that BNS section numbers exist, and that claims about the present case are not backed only by [precedent] passages. If a check fails, the model rewrites the answer once with feedback.

## Measurements so far

- **Stress test:** 48 hand-written questions + 9 chained follow-ups (57 in all) on 3 judgments (`stress_cases.json`). Each answer is checked with regexes for required and forbidden phrases.
  - Score: **47–48 / 57 (82–84%)**, deterministic, the same over 3 repeats.
  - The last ~8 iterations of retrieval fixes have not moved it: each fix rescues one question and breaks another.
- **Generated set:** 3,000 questions on 300 judgments, with automatic grading (case details, facts like FIR date / police station / charges, trap questions, vague questions).
  - Qwen 7B answers about **87%** correctly when forced to answer.
- **Retrieval recall** on 230 generated fact questions where the answer sentence is known (`bench_out/reranker_compare.json`):

  | setting | answer sentence in the 5 passages the model gets |
  |---|---|
  | no reranker | 41% |
  | current cross-encoder, 12 candidates | 50% |
  | current cross-encoder, 30 candidates | 57% |
  | bge-reranker-v2-m3, 30 candidates | 60% |

  The answer sentence is among the 12 / 30 first-stage candidates only **55% / 73%** of the time. So **first-stage retrieval loses the answer before any reranker sees it.**

## The remaining failures (latest run, `bench_out/stress_results.json` has answers + the passages given)

| id | question | what happens |
|---|---|---|
| B2, B12 | list the grounds / the court's reasons | the right passages are given, but the model leaves points out of a long answer |
| B3 | counsel's arguments + cases relied on | names the cases now, misses the witness/CCTV argument |
| B14 | Bhajan Lal category (1) | the quoted text is in the judgment; it is never retrieved |
| C1 | under which provision was bail sought | "Section 483 BNSS" sits in paragraph 2; never retrieved |
| C3 | list all bail conditions | the list runs over a page break with footers inside it; the bond amount is lost |
| C4 | grounds the prosecution could raise (reasoning) | the model answers about bail conditions instead |
| C12 | list the SC judgments cited on speedy trial | case names sit in "In the case of X Vs." intro sentences, not in the quotes; it gets some, not all |
| C14 | what principle did Antil v. CBI reiterate | the sentence stating it was in the passages; the model quoted a different one |
| A12 | (new this run) "speaking order" | regression after the last retrieval change |

## What I'd like from you

1. **First-stage retrieval** is the measured bottleneck (55–73% candidate recall). For long Indian judgments (100–300 sentences, heavy quotation of precedents, page footers and footnotes inside the text, sentences cut at "Vs."/"Ors."), what would you change? For example:
   - chunking by paragraph instead of by sentence;
   - a different embedder (InLegalBERT is not trained for retrieval);
   - BM25 weighting;
   - query rewriting;
   - HyDE;
   - cleaning footers before splitting.

   Which would you try first, and why?
2. **List / multi-part questions** (B2, B12, C3, C12): is it better to give the model more passages, to retrieve per sub-question, to do a map-reduce extraction, or something else, given a 7B model and an 8k context?
3. **The model ignores the right passage it was given** (C14, B2, B12). Prompting ideas for a 7B model? Would a two-step "extract the relevant sentences, then answer" help?
4. **Evaluation:** is a 57-question regex-graded stress test a sound basis for decisions? How would you set up evaluation so we stop overfitting? We have the 3,000 generated questions with gold answers and gold evidence sentences.
5. **Anything in the architecture** (the Bayesian router, the [fact]/[precedent] tagging, the grounding check plus one rewrite) you think is over-engineered or hurting more than helping.

Please point to specific functions in the attached files where relevant.

---

## Files to attach

Code (the parts that matter):
- `ipc/retriever.py`: retrieval, neighbour expansion, the router hook (`ChatRouter.handle`)
- `ipc/prompts.py`: prompt building, [fact]/[precedent] tagging (`_quote_tags`, `classify_passage_type`, `build_chat`)
- `ipc/grounding.py`: the post-answer grounding check
- `ipc/engine.py`: how the pieces are called (`Engine.chat`)
- `ipc/learned_bn.py` and `ipc/bayes.py`: the router (optional, for question 5)
- `models/final/legal_hybrid.py`: only for `split_sentences` (the sentence splitter)

Tests and results:
- `stress_cases.json`: the 57 test questions and their regex checks
- `bench_out/stress_results.json`: the latest run, with answers, routes and the passages each question got
- `bench_out/reranker_compare.json`: the retrieval recall numbers
- `test-case-files/`: the 3 judgment PDFs (0037…, 657868…, ddc8…; 51e377… is a duplicate, skip it)

Optional: `FEATURES.md` (overview) and `compare_rerankers.py` (how recall was measured).

**Do not attach** `.venv`, `models/` (weights), or `labels/` (large).
