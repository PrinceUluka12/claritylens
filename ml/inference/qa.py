# ml/inference/qa.py
# ============================================================
# RAG-based extractive QA inference.
# Combines Word2Vec retrieval with DistilBERT answer extraction.
#
# Flow:
#   question → embed → pgvector search → top-k chunks
#   → DistilBERT QA on each chunk → best answer span
#
# CPU NOTE: torch.no_grad() on all inference calls.
# Batch size 1 for QA — each chunk is a separate forward pass.
# ============================================================

import json
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
)
from loguru import logger

from ml.inference.embedder import get_embedder


@dataclass
class QAResult:
    """Result for a single Q&A query."""
    question:        str
    answer:          str           # extracted answer span
    source_chunk:    str           # chunk the answer came from
    chunk_index:     int           # position in document
    confidence:      float         # answer confidence score
    inference_ms:    float


class QAInference:
    """
    RAG-based QA inference wrapper.
    Loaded once at server startup.
    """

    def __init__(
        self,
        model_dir:  str | Path = "./ml/models/qa",
        max_length: int = 384,
        top_k:      int = 5,       # retrieve top-k chunks before QA
    ) -> None:

        model_dir  = Path(model_dir)
        self.max_length = max_length
        self.top_k      = top_k

        # Load tokenizer
        logger.info("Loading DistilBERT QA tokenizer...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )

        # Load model
        quantized_path = model_dir / "qa_quantized.pt"
        full_path      = model_dir / "best_qa_model.pt"

        if quantized_path.exists():
            model_path = quantized_path
            logger.info("Loading quantized QA model...")
        elif full_path.exists():
            model_path = full_path
            logger.info("Loading full precision QA model...")
        else:
            raise FileNotFoundError(
                f"No QA model found in {model_dir}. "
                f"Run train_qa.py first."
            )

        model = DistilBertForQuestionAnswering.from_pretrained(
            "distilbert-base-uncased"
        )
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        if model_path == full_path:
            logger.info("Applying dynamic quantization...")
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )

        # CPU NOTE: eval() disables dropout for deterministic inference
        model.eval()
        self.model = model

        # Load embedder for retrieval
        self.embedder = get_embedder()

        logger.info(
            f"QAInference ready — "
            f"top_k={top_k}, max_length={max_length}"
        )

    def _retrieve_chunks(
        self,
        question:       str,
        chunks:         list,
        top_k:          int,
    ) -> list:
        """
        Retrieves top-k most relevant chunks using
        Word2Vec cosine similarity.

        Args:
            question: user's question string
            chunks:   list of Chunk objects or dicts with 'text'
            top_k:    number of chunks to retrieve

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not chunks:
            return []

        # Embed the question
        q_embedding = self.embedder.embed_text(question)

        # Embed all chunks
        chunk_texts = [
            c.text if hasattr(c, "text") else c.get("text", "")
            for c in chunks
        ]
        chunk_embeddings = self.embedder.embed_batch(chunk_texts)

        # Cosine similarity — since both are unit normalized,
        # this is just a dot product
        # CPU NOTE: numpy dot product is faster than torch on CPU
        # for small matrices like this
        similarities = chunk_embeddings @ q_embedding

        # Get top-k indices
        top_k_actual = min(top_k, len(chunks))
        top_indices  = np.argsort(similarities)[::-1][:top_k_actual]

        return [
            (chunks[i], float(similarities[i]))
            for i in top_indices
        ]

    def _extract_answer(
        self,
        question: str,
        context:  str,
    ) -> tuple[str, float]:
        """
        Runs DistilBERT QA to extract answer span from context.

        Returns (answer_text, confidence_score).
        """
        encoding = self.tokenizer(
            question,
            context,
            max_length  = self.max_length,
            truncation  = "only_second",
            padding     = "max_length",
            return_tensors = "pt",
        )

        # CPU NOTE: torch.no_grad() skips gradient computation
        with torch.no_grad():
            outputs = self.model(
                input_ids      = encoding["input_ids"],
                attention_mask = encoding["attention_mask"],
            )

        # Get start and end positions
        start_logits = outputs.start_logits.squeeze(0)
        end_logits   = outputs.end_logits.squeeze(0)

        # Find best valid span (end >= start, max combined score)
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs   = torch.softmax(end_logits,   dim=-1)

        # Limit answer length to 50 tokens
        max_answer_len = 50
        best_score     = -float("inf")
        best_start     = 0
        best_end       = 0

        for start in range(len(start_logits)):
            for end in range(start, min(start + max_answer_len, len(end_logits))):
                score = start_probs[start].item() + end_probs[end].item()
                if score > best_score:
                    best_score = score
                    best_start = start
                    best_end   = end

        # Convert token positions back to text
        input_ids = encoding["input_ids"].squeeze(0)
        answer_tokens = input_ids[best_start : best_end + 1]
        answer = self.tokenizer.decode(
            answer_tokens,
            skip_special_tokens = True,
        ).strip()

        # Confidence = geometric mean of start and end probabilities
        confidence = (
            start_probs[best_start].item() *
            end_probs[best_end].item()
        ) ** 0.5

        return answer, confidence

    def answer(
        self,
        question: str,
        chunks:   list,
    ) -> QAResult:
        """
        Answers a question about a document using RAG.

        Args:
            question: natural language question
            chunks:   document chunks (Chunk objects or dicts)

        Returns:
            QAResult with extracted answer and source chunk
        """
        start_time = time.time()

        if not question or not chunks:
            return QAResult(
                question     = question,
                answer       = "No document loaded.",
                source_chunk = "",
                chunk_index  = -1,
                confidence   = 0.0,
                inference_ms = 0.0,
            )

        # Step 1: Retrieve relevant chunks
        retrieved = self._retrieve_chunks(question, chunks, self.top_k)

        if not retrieved:
            return QAResult(
                question     = question,
                answer       = "Could not find relevant sections.",
                source_chunk = "",
                chunk_index  = -1,
                confidence   = 0.0,
                inference_ms = 0.0,
            )

        # Step 2: Run QA on each retrieved chunk, pick best answer
        best_answer     = ""
        best_confidence = 0.0
        best_chunk      = retrieved[0][0]
        best_chunk_idx  = 0

        for chunk, sim_score in retrieved:
            chunk_text = (
                chunk.text if hasattr(chunk, "text")
                else chunk.get("text", "")
            )
            chunk_idx = (
                chunk.chunk_index if hasattr(chunk, "chunk_index")
                else chunk.get("chunk_index", 0)
            )

            answer, confidence = self._extract_answer(question, chunk_text)

            # Weight confidence by retrieval similarity
            weighted_confidence = confidence * (0.5 + 0.5 * sim_score)

            if weighted_confidence > best_confidence and answer:
                best_answer     = answer
                best_confidence = weighted_confidence
                best_chunk      = chunk
                best_chunk_idx  = chunk_idx

        inference_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Q&A: '{question[:50]}...' → "
            f"'{best_answer[:50]}' "
            f"(conf={best_confidence:.3f}, {inference_ms:.0f}ms)"
        )

        return QAResult(
            question     = question,
            answer       = best_answer,
            source_chunk = (
                best_chunk.text if hasattr(best_chunk, "text")
                else best_chunk.get("text", "")
            ),
            chunk_index  = best_chunk_idx,
            confidence   = round(best_confidence, 4),
            inference_ms = inference_ms,
        )


# Singleton
_qa_instance = None


def get_qa(
    model_dir: str = "./ml/models/qa",
) -> QAInference:
    global _qa_instance
    if _qa_instance is None:
        _qa_instance = QAInference(model_dir)
    return _qa_instance