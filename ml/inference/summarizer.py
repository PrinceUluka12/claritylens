# ml/inference/summarizer.py
# ============================================================
# Extractive summarization using TF-IDF + TextRank.
# No model weights needed — pure algorithm, runs instantly.
#
# Called by backend/services/ at request time.
# CPU NOTE: TF-IDF + sparse cosine similarity is extremely
# fast on CPU — a 50-page contract summarizes in under 1s.
# ============================================================

import time
import re
from dataclasses import dataclass

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


@dataclass
class SummaryResult:
    """Result from extractive summarization."""
    filename:       str
    summary:        str           # concatenated top sentences
    top_sentences:  list[dict]    # ranked sentences with scores
    sentence_count: int           # total sentences in document
    summary_ratio:  float         # compression ratio
    inference_ms:   float


class ExtractiveSummarizer:
    """
    TF-IDF + TextRank extractive summarizer.

    Loaded once at server startup — stateless so a single
    instance handles all requests safely.
    """

    def __init__(
        self,
        max_features: int = 5000,   # TF-IDF vocabulary size
        min_df:       int = 1,      # minimum document frequency
    ) -> None:

        # CPU NOTE: TfidfVectorizer with sparse matrices is
        # memory efficient — a 500-sentence document uses
        # ~50KB of RAM vs ~50MB for dense embeddings
        self.vectorizer = TfidfVectorizer(
            max_features = max_features,
            min_df       = min_df,
            stop_words   = "english",
            ngram_range  = (1, 2),   # unigrams + bigrams
        )
        logger.info("ExtractiveSummarizer initialized")

    def _split_sentences(self, text: str) -> list[str]:
        """
        Splits contract text into sentences.
        Legal contracts use consistent punctuation so
        a simple rule-based splitter works well.
        """
        # Split on sentence-ending punctuation followed by
        # whitespace and a capital letter or number
        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z0-9])',
            text
        )

        # Clean each sentence
        cleaned = []
        for s in sentences:
            s = s.strip()
            # Filter out very short sentences —
            # they're usually headings or artifacts
            if len(s.split()) >= 6:
                cleaned.append(s)

        return cleaned

    def _build_similarity_matrix(
        self,
        sentences: list[str],
    ) -> np.ndarray:
        """
        Builds a sentence similarity matrix using TF-IDF
        cosine similarity.

        Returns an NxN matrix where entry [i,j] is the
        cosine similarity between sentence i and sentence j.
        """
        # Fit TF-IDF on all sentences
        tfidf_matrix = self.vectorizer.fit_transform(sentences)

        # Compute pairwise cosine similarity
        # CPU NOTE: cosine_similarity on sparse matrices is
        # fast because it skips zero entries
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Zero out self-similarity (diagonal)
        np.fill_diagonal(sim_matrix, 0)

        return sim_matrix

    def _textrank(
        self,
        sim_matrix: np.ndarray,
        damping:    float = 0.85,
    ) -> np.ndarray:
        """
        Runs PageRank on the sentence similarity graph.

        damping=0.85 is the standard PageRank damping factor —
        probability that a random walker follows a graph edge
        vs teleports to a random node.

        Returns a score array where higher = more important.
        """
        # Build directed graph from similarity matrix
        graph = nx.from_numpy_array(sim_matrix)

        # Run PageRank
        scores = nx.pagerank(
            graph,
            alpha   = damping,
            max_iter = 100,
            tol     = 1e-6,
        )

        # Convert to numpy array indexed by sentence position
        score_array = np.array([scores[i] for i in range(len(scores))])
        return score_array

    def summarize(
        self,
        text:         str,
        filename:     str = "",
        num_sentences: int = 10,
        min_length:   int = 20,
    ) -> SummaryResult:
        """
        Generates an extractive summary of contract text.

        Args:
            text:          Clean contract text
            filename:      Document name for result object
            num_sentences: Number of sentences to include
            min_length:    Minimum words per sentence

        Returns:
            SummaryResult with ranked sentences and summary text
        """
        start_time = time.time()

        if not text or not text.strip():
            logger.warning("summarize() called with empty text")
            return SummaryResult(
                filename       = filename,
                summary        = "",
                top_sentences  = [],
                sentence_count = 0,
                summary_ratio  = 0.0,
                inference_ms   = 0.0,
            )

        # ── Step 1: Split into sentences ──────────────────────
        sentences = self._split_sentences(text)

        if len(sentences) < 3:
            # Document too short to summarize meaningfully
            logger.info(
                f"Document too short ({len(sentences)} sentences) "
                f"— returning full text"
            )
            return SummaryResult(
                filename       = filename,
                summary        = text,
                top_sentences  = [
                    {"sentence": s, "score": 1.0, "position": i}
                    for i, s in enumerate(sentences)
                ],
                sentence_count = len(sentences),
                summary_ratio  = 1.0,
                inference_ms   = (time.time() - start_time) * 1000,
            )

        # Cap at 500 sentences for performance
        # (contracts rarely exceed this in meaningful content)
        if len(sentences) > 500:
            sentences = sentences[:500]

        # ── Step 2: Build similarity matrix ───────────────────
        sim_matrix = self._build_similarity_matrix(sentences)

        # ── Step 3: Run TextRank ───────────────────────────────
        scores = self._textrank(sim_matrix)

        # ── Step 4: Select top sentences ──────────────────────
        # Get indices of top-scoring sentences
        num_to_select = min(num_sentences, len(sentences))
        top_indices   = np.argsort(scores)[::-1][:num_to_select]

        # Sort by original position to maintain reading order
        top_indices_ordered = sorted(top_indices.tolist())

        # Build result objects
        top_sentences = []
        for idx in top_indices_ordered:
            sentence = sentences[idx]
            # Filter minimum length
            if len(sentence.split()) >= min_length:
                top_sentences.append({
                    "sentence": sentence,
                    "score":    round(float(scores[idx]), 4),
                    "position": idx,
                })

        # ── Step 5: Build summary text ─────────────────────────
        summary = " ".join(s["sentence"] for s in top_sentences)

        summary_ratio = len(top_sentences) / len(sentences)
        inference_ms  = (time.time() - start_time) * 1000

        logger.info(
            f"Summarized {len(sentences)} sentences → "
            f"{len(top_sentences)} sentences "
            f"({summary_ratio:.1%} compression) "
            f"in {inference_ms:.1f}ms"
        )

        return SummaryResult(
            filename       = filename,
            summary        = summary,
            top_sentences  = top_sentences,
            sentence_count = len(sentences),
            summary_ratio  = summary_ratio,
            inference_ms   = inference_ms,
        )

    def summarize_by_risk(
        self,
        chunks:       list,
        risk_results: list,
        filename:     str = "",
        top_per_label: int = 2,
    ) -> dict:
        """
        Generates per-risk-label summaries.
        Groups chunks by their predicted risk label and
        summarizes each group separately.

        This is what the frontend displays in the risk panel —
        not one big summary but one focused summary per risk type.

        Args:
            chunks:        List of Chunk objects
            risk_results:  List of ClauseResult from classifier
            filename:      Document name
            top_per_label: Sentences per risk category

        Returns:
            Dict mapping risk_label → summary string
        """
        from collections import defaultdict

        # Group chunk texts by risk label
        label_texts = defaultdict(list)
        for result in risk_results:
            if result.confidence > 0.3:  # confidence threshold
                label_texts[result.risk_label].append(result.text)

        risk_summaries = {}

        for label, texts in label_texts.items():
            if not texts:
                continue

            # Combine all chunks for this risk type
            combined_text = " ".join(texts)

            summary_result = self.summarize(
                text          = combined_text,
                filename      = filename,
                num_sentences = top_per_label,
                min_length    = 10,
            )

            risk_summaries[label] = {
                "summary":    summary_result.summary,
                "sentences":  summary_result.top_sentences,
                "chunk_count": len(texts),
            }

        return risk_summaries


# Singleton instance
_summarizer_instance = None


def get_summarizer() -> ExtractiveSummarizer:
    """Returns the singleton summarizer instance."""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = ExtractiveSummarizer()
    return _summarizer_instance