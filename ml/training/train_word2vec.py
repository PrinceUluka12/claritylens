# ml/training/train_word2vec.py
# ============================================================
# Trains a Word2Vec Skip-gram model on CUAD contract text.
# Saves embedding matrix and word index to disk.
#
# Usage:
#   python ml/training/train_word2vec.py
#
# Expected runtime: 3-8 minutes on CPU
# ============================================================

import json
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter
from loguru import logger


# ── Hyperparameters ──────────────────────────────────────────
EMBEDDING_DIM  = 128    # vector size per word
                        # CPU NOTE: 128 dims is the sweet spot —
                        # small enough for fast similarity search,
                        # large enough to capture legal semantics.
                        # 300 dims (standard Word2Vec) is 2.3x slower
                        # for cosine search with no meaningful gain
                        # on a domain-specific corpus this size.

WINDOW_SIZE    = 5      # context window — words on each side of target
MIN_COUNT      = 3      # ignore words appearing fewer than 3 times
NEGATIVE_SAMPLES = 5   # negative samples per positive pair (noise contrastive)
LEARNING_RATE  = 0.025  # initial learning rate
MIN_LR         = 0.0001 # learning rate floor
EPOCHS         = 5      # passes over the corpus
SEED           = 42


def tokenize_text(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases and strips punctuation attached to words.
    We don't use the ContractTokenizer here because Word2Vec
    works on whole words, not subword pieces.
    """
    import re
    text = text.lower()
    # Remove punctuation except hyphens inside words
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return [w.strip("-") for w in text.split() if w.strip("-")]


def load_corpus(
    processed_dir: str = "./ml/data/processed",
    max_docs:      int = 20_000,
) -> list[list[str]]:
    """
    Loads contract texts and tokenizes them into sentences.
    Returns a list of token lists — one per document.
    """
    processed_path = Path(processed_dir)
    sentences = []

    for split in ["train.json"]:
        fpath = processed_path / split
        if not fpath.exists():
            logger.warning(f"{split} not found")
            continue

        logger.info(f"Loading corpus from {split}...")
        with open(fpath, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records[:max_docs]:
            # Use full_context for richer vocabulary coverage
            text = record.get("full_context", "")
            if not text:
                text = record.get("clause_text", "")
            if text:
                tokens = tokenize_text(text[:2000])  # cap at 2000 chars
                if len(tokens) > 5:
                    sentences.append(tokens)

    logger.info(f"Loaded {len(sentences)} documents")
    return sentences


def build_vocabulary(
    sentences:  list[list[str]],
    min_count:  int = MIN_COUNT,
) -> tuple[dict, dict, list]:
    """
    Builds word-to-index and index-to-word mappings.
    Filters out words below min_count frequency.

    Returns:
        word_to_idx: dict mapping word → integer ID
        idx_to_word: dict mapping integer ID → word
        vocab_list:  ordered list of vocabulary words
    """
    logger.info("Building vocabulary...")

    # Count all words across all sentences
    counter = Counter(
        word for sentence in sentences for word in sentence
    )

    # Filter by minimum frequency
    vocab_words = [
        word for word, count in counter.most_common()
        if count >= min_count
    ]

    logger.info(
        f"Vocabulary: {len(counter)} unique words → "
        f"{len(vocab_words)} after min_count={min_count} filter"
    )

    # Add special tokens at the front
    special = ["<PAD>", "<UNK>"]
    vocab_list  = special + vocab_words
    word_to_idx = {word: idx for idx, word in enumerate(vocab_list)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab_list)}

    return word_to_idx, idx_to_word, vocab_list


class Word2VecSkipGram:
    """
    Skip-gram Word2Vec implemented in pure NumPy.

    Why NumPy instead of PyTorch?
    For a model this small (vocab × 128 dims) NumPy is faster
    to implement, easier to inspect, and produces weights that
    are just a plain matrix — no model loading overhead at
    inference time, just a dictionary lookup + dot product.

    Architecture:
      Input:  one-hot word ID
      Hidden: embedding matrix W1 [vocab_size × embed_dim]
      Output: context matrix W2 [embed_dim × vocab_size]
      Loss:   negative sampling (approximates full softmax)
    """

    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int = EMBEDDING_DIM,
        seed:         int = SEED,
    ) -> None:

        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

        # W1: input embeddings — this is what we save and use
        # Initialize small random values — Xavier initialization
        scale = np.sqrt(1.0 / embed_dim)
        self.W1 = np.random.uniform(
            -scale, scale, (vocab_size, embed_dim)
        ).astype(np.float32)

        # W2: output/context embeddings
        self.W2 = np.zeros(
            (vocab_size, embed_dim), dtype=np.float32
        )

    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Look up embedding vector for a word ID."""
        return self.W1[word_idx]

    def train_pair(
        self,
        center_idx:   int,
        context_idx:  int,
        neg_indices:  list[int],
        lr:           float,
    ) -> float:
        """
        One gradient update for a (center, context) pair
        using negative sampling.

        Negative sampling: instead of computing softmax over
        the entire vocabulary (expensive), we only update
        weights for the true context word + k random "negative"
        words. This approximates the full objective while being
        much faster on CPU.
        """

        # Forward pass
        h = self.W1[center_idx]         # hidden layer: [embed_dim]
        u_pos = self.W2[context_idx]    # positive context vector

        # Sigmoid activation for positive pair
        score_pos = np.dot(h, u_pos)
        sig_pos   = 1.0 / (1.0 + np.exp(-np.clip(score_pos, -10, 10)))

        # Loss contribution from positive pair
        loss = -np.log(sig_pos + 1e-7)

        # Gradient for positive pair
        grad_pos = (sig_pos - 1.0)  # d_loss/d_score for positive

        # Process negative samples
        grad_h = grad_pos * u_pos   # accumulate gradient for center word

        # Update positive context vector
        self.W2[context_idx] -= lr * grad_pos * h

        for neg_idx in neg_indices:
            u_neg     = self.W2[neg_idx]
            score_neg = np.dot(h, u_neg)
            sig_neg   = 1.0 / (1.0 + np.exp(-np.clip(score_neg, -10, 10)))

            loss     += -np.log(1.0 - sig_neg + 1e-7)
            grad_neg  = sig_neg  # d_loss/d_score for negative

            grad_h           += grad_neg * u_neg
            self.W2[neg_idx] -= lr * grad_neg * h

        # Update center word embedding
        self.W1[center_idx] -= lr * grad_h

        return float(loss)


def get_noise_distribution(
    word_to_idx: dict,
    sentences:   list[list[str]],
) -> np.ndarray:
    """
    Builds the unigram noise distribution for negative sampling.
    Words are sampled proportional to frequency^0.75 — this
    smoothing gives rare words a slightly higher chance of being
    selected as negatives, which improves embedding quality for
    infrequent legal terms.
    """
    counts = Counter(
        word for sentence in sentences
        for word in sentence
        if word in word_to_idx
    )

    vocab_size = len(word_to_idx)
    dist = np.zeros(vocab_size, dtype=np.float32)

    for word, idx in word_to_idx.items():
        dist[idx] = counts.get(word, 0) ** 0.75

    # Normalize to probability distribution
    total = dist.sum()
    if total > 0:
        dist /= total

    return dist


def train(
    output_dir:    str = "./ml/models/embedder",
    processed_dir: str = "./ml/data/processed",
) -> None:

    random.seed(SEED)
    np.random.seed(SEED)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load corpus ──────────────────────────────────
    sentences = load_corpus(processed_dir)
    if not sentences:
        raise ValueError("No corpus loaded. Run build_dataset.py first.")

    # ── Step 2: Build vocabulary ─────────────────────────────
    word_to_idx, idx_to_word, vocab_list = build_vocabulary(sentences)
    vocab_size = len(vocab_list)

    # ── Step 3: Build noise distribution ─────────────────────
    noise_dist = get_noise_distribution(word_to_idx, sentences)

    # ── Step 4: Initialize model ─────────────────────────────
    model = Word2VecSkipGram(vocab_size=vocab_size, embed_dim=EMBEDDING_DIM)
    logger.info(
        f"Model initialized — vocab={vocab_size:,}, "
        f"embed_dim={EMBEDDING_DIM}, epochs={EPOCHS}"
    )

    # ── Step 5: Training loop ─────────────────────────────────
    total_pairs   = 0
    total_loss    = 0.0
    start_time    = time.time()

    # Precompute total pairs for learning rate decay
    pairs_per_epoch = sum(
        max(0, len(s) - 2 * WINDOW_SIZE) * WINDOW_SIZE * 2
        for s in sentences
    )
    total_pairs_all = pairs_per_epoch * EPOCHS

    pairs_seen = 0

    for epoch in range(EPOCHS):
        epoch_loss  = 0.0
        epoch_pairs = 0

        # Shuffle sentences each epoch
        random.shuffle(sentences)

        for sent_idx, sentence in enumerate(sentences):

            # Convert words to indices, skip unknown words
            indices = [
                word_to_idx[w] for w in sentence
                if w in word_to_idx
            ]

            if len(indices) < 2:
                continue

            for center_pos, center_idx in enumerate(indices):

                # Dynamic window: randomly reduce window size
                # This is the original Word2Vec trick — closer
                # words get weighted more heavily
                dynamic_window = random.randint(1, WINDOW_SIZE)

                context_range = range(
                    max(0, center_pos - dynamic_window),
                    min(len(indices), center_pos + dynamic_window + 1)
                )

                for ctx_pos in context_range:
                    if ctx_pos == center_pos:
                        continue

                    context_idx = indices[ctx_pos]

                    # Sample negative examples from noise distribution
                    neg_indices = np.random.choice(
                        vocab_size,
                        size=NEGATIVE_SAMPLES,
                        p=noise_dist,
                    ).tolist()

                    # Learning rate decay — linear warmdown
                    lr = max(
                        MIN_LR,
                        LEARNING_RATE * (1 - pairs_seen / total_pairs_all)
                    )

                    # One gradient step
                    loss = model.train_pair(
                        center_idx, context_idx, neg_indices, lr
                    )

                    epoch_loss  += loss
                    epoch_pairs += 1
                    pairs_seen  += 1

            # Log progress every 2000 sentences
            if (sent_idx + 1) % 2000 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Sentence {sent_idx+1}/{len(sentences)} | "
                    f"Loss {epoch_loss/max(epoch_pairs,1):.4f} | "
                    f"Elapsed {elapsed:.0f}s"
                )

        avg_loss = epoch_loss / max(epoch_pairs, 1)
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} complete — "
            f"avg_loss={avg_loss:.4f}, pairs={epoch_pairs:,}"
        )

    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time:.0f}s")

    # ── Step 6: Save artifacts ───────────────────────────────

    # Save embedding matrix as numpy array
    embeddings_path = output_path / "word2vec_embeddings.npy"
    np.save(str(embeddings_path), model.W1)
    logger.info(f"Embeddings saved — shape: {model.W1.shape}")

    # Save word-to-index mapping
    w2i_path = output_path / "word2vec_vocab.json"
    with open(w2i_path, "w", encoding="utf-8") as f:
        json.dump(word_to_idx, f, ensure_ascii=False)
    logger.info(f"Vocabulary saved — {len(word_to_idx):,} words")

    # Save config for inference loader
    config = {
        "embedding_dim":     EMBEDDING_DIM,
        "vocab_size":        vocab_size,
        "window_size":       WINDOW_SIZE,
        "min_count":         MIN_COUNT,
        "negative_samples":  NEGATIVE_SAMPLES,
        "epochs":            EPOCHS,
    }
    with open(output_path / "word2vec_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Step 7: Quick similarity test ────────────────────────
    logger.info("\n=== SIMILARITY TEST ===")
    test_words = ["indemnify", "terminate", "liability", "license"]

    for word in test_words:
        if word not in word_to_idx:
            logger.info(f"  '{word}' not in vocabulary")
            continue

        word_vec = model.W1[word_to_idx[word]]

        # Cosine similarity against all embeddings
        norms     = np.linalg.norm(model.W1, axis=1, keepdims=True)
        norms     = np.where(norms == 0, 1e-8, norms)
        normed_W1 = model.W1 / norms
        word_norm = word_vec / (np.linalg.norm(word_vec) + 1e-8)
        sims      = normed_W1 @ word_norm

        # Top 5 most similar words
        top_idx = np.argsort(sims)[::-1][1:6]
        similar = [idx_to_word[i] for i in top_idx]
        logger.info(f"  '{word}' → most similar: {similar}")


if __name__ == "__main__":
    train()