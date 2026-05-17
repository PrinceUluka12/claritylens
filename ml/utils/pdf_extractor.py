# ml/utils/pdf_extractor.py
# ============================================================
# Extracts plain text from PDF files.
# Entry point for the entire ML pipeline — every document
# starts here before chunking, embedding, and classification.
# ============================================================

import time
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

import pypdf

from ml.utils.text_utils import clean_contract_text, word_count
from ml.utils.chunker import chunks_from_contract, Chunk


@dataclass
class ExtractedDocument:
    """
    Everything we know about a document after extraction.
    Passed through the pipeline as a single object.
    """
    filename:    str         # original uploaded filename
    filepath:    str         # path on disk
    raw_text:    str         # text before cleaning
    clean_text:  str         # text after cleaning
    page_count:  int         # number of PDF pages
    word_count:  int         # words in clean text
    chunks:      list[Chunk] # overlapping clause windows
    extract_ms:  float       # extraction time in milliseconds


def extract_pdf(
    filepath:   str | Path,
    chunk_size: int = 200,
    overlap:    int = 50,
) -> ExtractedDocument:
    """
    Full pipeline: PDF → clean text → chunks.

    Args:
        filepath:   Path to the uploaded PDF file
        chunk_size: Words per chunk (default 200)
        overlap:    Overlap between chunks (default 50)

    Returns:
        ExtractedDocument with text and chunks ready for ML

    Raises:
        FileNotFoundError: if the PDF doesn't exist
        ValueError: if the PDF has no extractable text
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    if filepath.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {filepath.suffix}")

    start_time = time.time()

    logger.info(f"Extracting text from: {filepath.name}")

    # ── Step 1: Open PDF and extract text page by page ───────
    raw_pages = []

    with open(filepath, "rb") as f:
        reader = pypdf.PdfReader(f)
        page_count = len(reader.pages)

        logger.info(f"  Pages: {page_count}")

        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    raw_pages.append(page_text)
            except Exception as e:
                # Some pages fail extraction (images, encrypted content)
                # Log and continue — don't fail the whole document
                logger.warning(
                    f"  Page {page_num + 1} extraction failed: {e}"
                )
                continue

    # ── Step 2: Join all pages ────────────────────────────────
    # Two newlines between pages preserves paragraph boundaries
    raw_text = "\n\n".join(raw_pages)

    if not raw_text.strip():
        raise ValueError(
            f"No text could be extracted from {filepath.name}. "
            f"The PDF may be scanned or image-based."
        )

    # ── Step 3: Clean the text ────────────────────────────────
    clean_text = clean_contract_text(raw_text)
    wc         = word_count(clean_text)

    logger.info(f"  Extracted: {wc} words after cleaning")

    # ── Step 4: Chunk the clean text ─────────────────────────
    chunks = chunks_from_contract(clean_text, chunk_size, overlap)

    logger.info(f"  Chunks: {len(chunks)}")

    extract_ms = (time.time() - start_time) * 1000
    logger.info(f"  Extraction time: {extract_ms:.1f}ms")

    return ExtractedDocument(
        filename   = filepath.name,
        filepath   = str(filepath),
        raw_text   = raw_text,
        clean_text = clean_text,
        page_count = page_count,
        word_count = wc,
        chunks     = chunks,
        extract_ms = extract_ms,
    )


def extract_text_only(filepath: str | Path) -> str:
    """
    Lightweight version — returns just the clean text string.
    Used by the training pipeline when we don't need chunks.
    """
    doc = extract_pdf(filepath)
    return doc.clean_text