# ml/utils/text_utils.py
# ============================================================
# Text cleaning utilities.
# Called by the chunker before splitting — garbage in means
# garbage embeddings out, so we clean aggressively here.
# ============================================================

import re
from loguru import logger


def clean_contract_text(text: str) -> str:
    """
    Cleans raw text extracted from a PDF contract.

    Order matters here — each step assumes the previous
    step has already run.
    """

    if not text or not text.strip():
        return ""

    # ── Step 1: Normalize line endings ───────────────────────
    # Windows PDFs often have \r\n; normalize everything to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ── Step 2: Remove page numbers ──────────────────────────
    # Matches patterns like: "Page 1", "PAGE 1", "- 1 -", "1."
    # at the start or end of a line
    text = re.sub(r"(?m)^\s*-?\s*[Pp]age\s+\d+\s*-?\s*$", "", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # ── Step 3: Remove repeated headers/footers ──────────────
    # Lines that appear 3+ times are likely headers or footers
    lines = text.split("\n")
    line_counts: dict = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # Remove lines that repeat 3 or more times
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or line_counts.get(stripped, 0) < 3:
            filtered_lines.append(line)
    text = "\n".join(filtered_lines)

    # ── Step 4: Fix broken hyphenation ───────────────────────
    # PDFs often break words across lines: "indem-\nnity" → "indemnity"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # ── Step 5: Collapse excessive whitespace ────────────────
    # Replace 3+ consecutive newlines with 2 (preserve paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # ── Step 6: Remove non-printable characters ───────────────
    # Keeps standard ASCII + common unicode punctuation
    text = re.sub(r"[^\x20-\x7E\n\u2019\u2018\u201C\u201D\u2013\u2014]", " ", text)

    # ── Step 7: Normalize quotes and dashes ──────────────────
    # Curly quotes → straight quotes (tokenizer handles these better)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201C", '"').replace("\u201D", '"')
    # Em/en dashes → hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    return text.strip()


def word_count(text: str) -> int:
    """Returns number of whitespace-separated tokens in text."""
    return len(text.split())


def sentence_split(text: str) -> list[str]:
    """
    Splits text into sentences using simple punctuation rules.
    We avoid NLTK here to keep this function dependency-free
    and fast — it's called thousands of times during chunking.

    Not perfect, but good enough for legal contract text which
    uses consistent punctuation.
    """
    # Split on . ! ? followed by whitespace and a capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    # Filter empty strings
    return [s.strip() for s in sentences if s.strip()]