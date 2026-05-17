# ml/inference/embedder.py
# ============================================================
# Word2Vec embedder inference wrapper.
# Converts text to a 128-dim vector for pgvector similarity search.
# Called by the RAG pipeline to embed both documents and questions.
#
# CPU NOTE: inference is a pure numpy lookup — no neural network
# forward pass needed. Embedding a chunk takes under 1ms.
# ============================================================

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from loguru import logger


@dataclass
class EmbedderConfig:
    embedding_dim: int
    vocab_size:    int
    window_size:   int
    min_count:     int


class EmbedderInference:
    """
    Loads trained Word2Vec weights and embeds text at inference time.
    Document-level embedding = mean of word embeddings.
    """

    def __init__(
        self,
        model_dir: str | Path = "./ml/models/embedder",
    ) -> None:

        model_dir = Path(model_dir)

        # Load config
        config_path = model_dir / "word2vec_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"word2vec_config.json not found in {model_dir}. "
                f"Run train_word2vec.py first."
            )
        with open(config_path) as f:
            config_data = json.load(f)

        self.config = EmbedderConfig(
            embedding_dim = config_data["embedding_dim"],
            vocab_size    = config_data["vocab_size"],
            window_size   = config_data["window_size"],
            min_count     = config_data["min_count"],
        )

        # Load vocabulary
        vocab_path = model_dir / "word2vec_vocab.json"
        with open(vocab_path) as f:
            self.word_to_idx = json.load(f)

        # Load embedding matrix
        # Shape: [vocab_size, embedding_dim]
        embeddings_path = model_dir / "word2vec_embeddings.npy"
        self.embeddings = np.load(str(embeddings_path))

        # L2-normalize all embeddings once at load time
        # This makes cosine similarity a simple dot product at query time
        # CPU NOTE: pre-normalizing saves repeated computation per query
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        self.embeddings_normed = self.embeddings / norms

        logger.info(
            f"EmbedderInference loaded — "
            f"vocab={self.config.vocab_size:,}, "
            f"dim={self.config.embedding_dim}"
        )

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer matching training preprocessing."""
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return [w.strip("-") for w in text.split() if w.strip("-")]

    def embed_text(self, text: str) -> np.ndarray:
        """
        Converts text to a 128-dim embedding vector.
        Uses mean pooling over word embeddings.

        Returns a unit-norm vector of shape [embedding_dim].
        Returns zero vector if no known words found.
        """
        tokens = self._tokenize(text)

        # Collect embeddings for known words
        word_embeddings = []
        for token in tokens:
            if token in self.word_to_idx:
                idx = self.word_to_idx[token]
                word_embeddings.append(self.embeddings[idx])

        if not word_embeddings:
            # Return zero vector for unknown text
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        # Mean pooling
        mean_embedding = np.mean(word_embeddings, axis=0)

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        return mean_embedding.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embeds a list of texts.
        Returns array of shape [len(texts), embedding_dim].
        """
        return np.stack([self.embed_text(t) for t in texts])


# Singleton
_embedder_instance = None


def get_embedder(
    model_dir: str = "./ml/models/embedder",
) -> EmbedderInference:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = EmbedderInference(model_dir)
    return _embedder_instance