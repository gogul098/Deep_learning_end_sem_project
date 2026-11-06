"""tests/test_retriever.py · CPU-only check of the Bayesian retriever.
   CUDA_VISIBLE_DEVICES="" python -m tests.test_retriever [judgment.pdf]"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from ipc.retriever import BayesianRetriever, ChatRouter, ROOT

sys.path.insert(0, os.path.join(ROOT, "models", "final"))


def show(r):
    raw = r["raw"]
    print(f"  -> {r['route'].upper():8s} P(fail)={r['p_hallucination_high']:.2f} "
          f"top1_dist={raw['top1_faiss_distance']:.2f} rel={raw['top1_rerank_score']:.2f} relevant={raw['relevant_chunk_count']} "
          f"dom={raw['kb_domain_similarity']:.2f} words={raw['word_count']} ents={raw['named_entity_count']} "
          f"high={[k for k, v in r['evidence'].items() if v == 'High']}")


def main():
    t = time.time(); br = BayesianRetriever(); print(f"retriever built in {time.time() - t:.1f}s (KB {len(br.kb.docs)} passages)")
    t = time.time(); br.embed(["warm up"]); print(f"CPU embedder loaded in {time.time() - t:.1f}s")
    t = time.time(); br.embed(["Was the applicant granted bail under section 376?"]); print(f"one query embeds in {(time.time() - t) * 1000:.0f} ms (CPU)")

    print("\n== legal_kb retrieval (the three queries from test_legal_retriever.py) ==")
    for q in ["Punishment for murder and common intention", "Causing disappearance of evidence and conspiracy",
              "Rash and negligent driving on public way"]:
        qv = br.embed([q])[0]
        print(q, "| domain", round(br.kb.domain_similarity(qv), 3))
        for h in br.kb.search(q, qv, 3):
            print(f"    {h['id'][:34]:34s} cos={h['cos']:.3f} bm25={h['bm25']:.1f} rel={h['relevance']:.2f}")

    print("\n== router, no document loaded ==")
    for q in ["Which section punishes murder and what is common intention?", "What is the capital of France?", "why is it slow", "Explain"]:
        print(q); show(br.route(q))

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        import legal_hybrid as LH
        from local_model_text import read_document
        sents, _ = LH.view_sentences(LH.split_sentences(read_document(path)), "full")
        t = time.time(); vecs = br.embed(sents); print(f"\nembedded {len(sents)} judgment sentences on CPU in {time.time() - t:.1f}s")
        br.set_document(sents, vecs); print("lead party:", br.doc.party)
        cr = ChatRouter(br)
        for q in ["Was the applicant granted bail?", "What was the age of the prosecutrix and why does it matter?",
                  "What does section 376 punish?", "When was he arrested?", "Is the new BNS punishment different?",
                  "what about it", "the FIR delay", "What is the capital of France?"]:
            print(f"\nQ: {q}")
            r = cr.handle(q); show(r)
            if r["route"] == "Clarify":
                print("  ask:", r["ask_user"])
            for p in r["passages"][:2]:
                print(f"    [{p['id']}] ({p['where']}, rel {p['relevance']:.2f}) {p['text'][:110]}")
    import torch
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized(), "retriever must not touch CUDA"
    print("\nretriever ran with CUDA hidden: OK")


if __name__ == "__main__":
    main()
