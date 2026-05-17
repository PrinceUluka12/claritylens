
# ml/utils/chunker.py
# ============================================================
# Splits cleaned contract text into overlapping word windows.
# Each chunk is sized to fit within DistilBERT's 256 token limit.
#
# CPU NOTE: chunk size 200 words ≈ 250-280 tokens after
# DistilBERT tokenization. Staying under 256 tokens means
# attention computation stays fast on CPU.
# ============================================================

from dataclasses import dataclass
from loguru import logger
from ml.utils.text_utils import clean_contract_text, word_count


@dataclass
class Chunk:
    """
    Represents one clause-sized window of contract text.
    The dataclass gives us a clean object with typed fields
    instead of passing raw dicts everywhere.
    """
    chunk_index:  int    # position in document — 0, 1, 2...
    text:         str    # the chunk text itself
    word_start:   int    # index of first word in original document
    word_end:     int    # index of last word in original document
    word_count:   int    # number of words in this chunk


def chunk_document(
    text:         str,
    chunk_size:   int = 200,   # words per chunk
    overlap:      int = 50,    # words shared between adjacent chunks
    min_chunk:    int = 30,    # discard chunks shorter than this
) -> list[Chunk]:
    """
    Splits a contract into overlapping word-window chunks.

    Args:
        text:       Cleaned contract text (run clean_contract_text first)
        chunk_size: Target words per chunk
        overlap:    Words shared between adjacent chunks
        min_chunk:  Minimum words for a chunk to be kept

    Returns:
        List of Chunk objects in document order
    """

    if not text or not text.strip():
        logger.warning("chunk_document received empty text")
        return []

    # Split into individual words — our atomic unit
    words = text.split()

    if len(words) < min_chunk:
        logger.warning(
            f"Document too short to chunk ({len(words)} words) — "
            f"returning as single chunk"
        )
        return [Chunk(
            chunk_index = 0,
            text        = text.strip(),
            word_start  = 0,
            word_end    = len(words) - 1,
            word_count  = len(words),
        )]

    chunks  = []
    start   = 0
    idx     = 0

    # Step size = chunk_size - overlap
    # e.g. 200 - 50 = 150 words advance per chunk
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk_words = words[start:end]
        chunk_text  = " ".join(chunk_words)

        # Discard very short trailing chunks — they're usually
        # just a signature block or exhibit label, not a clause
        if len(chunk_words) >= min_chunk:
            chunks.append(Chunk(
                chunk_index = idx,
                text        = chunk_text,
                word_start  = start,
                word_end    = end - 1,
                word_count  = len(chunk_words),
            ))
            idx += 1

        # If we've reached the end of the document, stop
        if end == len(words):
            break

        start += step

    logger.debug(
        f"Chunked document: {len(words)} words → "
        f"{len(chunks)} chunks "
        f"(size={chunk_size}, overlap={overlap})"
    )

    return chunks


def chunks_from_contract(
    raw_text:   str,
    chunk_size: int = 200,
    overlap:    int = 50,
) -> list[Chunk]:
    """
    Convenience function: clean then chunk in one call.
    This is what the FastAPI service will call in Phase 12.
    """
    cleaned = clean_contract_text(raw_text)
    return chunk_document(cleaned, chunk_size, overlap)