"""hyena_extract.py · Run inference through Hyena encoder ONLY (before LLM) and extract text.

Usage:
    python hyena_extract.py path/to/judgment.pdf
"""

import argparse
import os
import sys

# Ensure offline execution and paths
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from local_model_text import read_document
from ipc.pipeline import Analyzer


def run_hyena_inference(pdf_path: str, hide_cited: bool = False, use_cache: bool = False) -> dict:
    """Extracts text from PDF and runs Hyena encoder inference without invoking Llama.

    Args:
        pdf_path: Path to the court judgment PDF file.
        hide_cited: If True, masks explicit section numbers (e.g. 'Section 302')
                    to test whether Hyena predicts sections from facts alone.
        use_cache: If True, uses cached results from .cache/results/ if available.

    Returns:
        dict containing:
            - passages: The selected text passages constructed by Hyena for the LLM.
            - candidates: Shortlisted IPC candidate sections with Hyena encoder probabilities.
            - evidence: Key evidence sentences per section identified by Hyena attention.
            - top_sentences: Top ranked narrative sentences by Hyena salience scores.
            - raw_text: Full raw text extracted from the PDF.
            - doc_tokens: Number of tokens processed.
            - sentences_count: Number of sentences in the document.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # 1. Read document text from PDF
    print(f"[*] Reading PDF: {pdf_path}")
    raw_text = read_document(pdf_path)
    if not raw_text.strip():
        raise ValueError("The PDF document is empty or requires OCR.")

    # 2. Load Analyzer with verifier='none' (Hyena only, 0 LLM loaded into memory)
    print("[*] Loading Hyena encoder & InLegalBERT (verifier='none')...")
    analyzer = Analyzer(verifier="none").load()

    # 3. Run Hyena encoder inference
    print("[*] Running Hyena inference...")
    res = analyzer.analyse(raw_text, hide_cited=hide_cited, use_cache=use_cache)

    # 4. Extract top salient sentences scored by Hyena
    all_sents = res["sentences_list"]
    scores = res["sentence_scores"]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    top_sentences = [{"index": f"S{i}", "score": scores[i], "text": all_sents[i]} for i in top_indices]

    return {
        "passages": res["passages"],
        "candidates": res["candidates"],
        "evidence": res["evidence"],
        "top_sentences": top_sentences,
        "raw_text": raw_text,
        "doc_tokens": res["doc_tokens"],
        "sentences_count": res["sentences"],
        "seconds": res["seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract Hyena encoder inference and text from PDF before LLM.")
    parser.add_argument("pdf_path", help="Path to judgment PDF file")
    parser.add_argument("--hide-cited", action="store_true", help="Mask section citations to infer from facts only")
    parser.add_argument("--show-raw", action="store_true", help="Print the full extracted raw text from PDF")
    args = parser.parse_args()

    results = run_hyena_inference(args.pdf_path, hide_cited=args.hide_cited)

    print("\n" + "=" * 70)
    print("HYENA ENCODER OUTPUT (BEFORE LLM)")
    print("=" * 70)
    print(f"Total Sentences : {results['sentences_count']}")
    print(f"Total Tokens    : {results['doc_tokens']:,}")
    print(f"Inference Time  : {results['seconds']}s")

    print("\n[1] SHORTLISTED IPC SECTIONS & HYENA ENCODER PROBABILITIES:")
    for section, data in results["candidates"].items():
        print(f"  • {section:<15} Encoder Score: {data['encoder']:.3f}")

    print("\n[2] PASSAGES SELECTED BY HYENA FOR THE LLM:")
    print("-" * 50)
    print(results["passages"])
    print("-" * 50)

    print("\n[3] HYENA EVIDENCE EXTRACTED PER SECTION (VIA ATTENTION):")
    for section, evidence_list in results["evidence"].items():
        print(f"\nSection: {section}")
        for ev in evidence_list:
            print(f"   ↳ {ev[:180]}...")

    print("\n[4] TOP 5 SALIENT SENTENCES (HIGHEST HYENA CONVOLUTION SCORES):")
    for s in results["top_sentences"][:5]:
        print(f"  [{s['index']}] (score: {s['score']:.4f}): {s['text'][:150]}...")

    if args.show_raw:
        print("\n[5] FULL EXTRACTED PDF TEXT:")
        print(results["raw_text"][:2000] + ("\n... [truncated]" if len(results["raw_text"]) > 2000 else ""))


if __name__ == "__main__":
    main()
