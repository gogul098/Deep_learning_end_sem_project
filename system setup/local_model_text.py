"""local_model_text.py · Read PDF/TXT judgment with automatic in-process OCR fallback (no torch import)."""
from __future__ import annotations

import os
from typing import Callable, Optional


def read_document(path: str, on_progress: Optional[Callable[[str], None]] = None) -> str:
    """Reads judgment text from a PDF or plain text file.

    Features:
      1. Instant digital extraction via PyMuPDF (fitz) for vector PDFs.
      2. Automatic in-process OCR fallback using PyMuPDF's built-in Tesseract integration
         if the PDF is a scanned bitmap (< 50 extractable words).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document file not found: {path}")

    if path.lower().endswith(".pdf"):
        import pymupdf

        with pymupdf.open(path) as doc:
            # 1. Fast digital text pass
            pages_text = [p.get_text() for p in doc]
            full_text = "\n".join(pages_text)
            words = len(full_text.split())

            # If digital text exists, return immediately
            if words >= 50:
                return full_text

            # 2. Scanned PDF detected: In-process OCR pass
            if on_progress:
                on_progress(f"Scanned PDF detected ({words} words found). Running in-process OCR...")

            ocr_pages = []
            for i, p in enumerate(doc):
                page_text = p.get_text().strip()
                # If page already had good digital text, keep it
                if len(page_text.split()) >= 30:
                    ocr_pages.append(page_text)
                    continue

                try:
                    # In-process OCR using PyMuPDF TextPage OCR
                    tp = p.get_textpage_ocr(language="eng", dpi=150, full=True)
                    text = tp.extractText().strip()
                    ocr_pages.append(text if text else page_text)
                except Exception:
                    ocr_pages.append(page_text)

            return "\n".join(ocr_pages)

    return open(path, encoding="utf-8", errors="ignore").read()
