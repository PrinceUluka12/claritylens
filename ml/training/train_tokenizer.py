# ml/training/train_tokenizer.py
# ============================================================
# Trains a BPE tokenizer on CUAD contract text from scratch.
# Saves to ml/models/embedder/ for use in all later phases.
#
# Usage:
#   python ml/training/train_tokenizer.py
#
# Expected runtime: 2-5 minutes on CPU
# ============================================================

import json
from pathlib import Path
from loguru import logger
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    processors,
    normalizers,
    decoders,
)


# ── Special tokens ───────────────────────────────────────────
# These must match DistilBERT's expected special tokens exactly.
# [PAD] fills sequences to equal length in a batch.
# [UNK] represents any token not in vocabulary.
# [CLS] is prepended to every sequence — DistilBERT reads it
#       as a summary of the whole sequence for classification.
# [SEP] marks the end of a sequence.
# [MASK] replaces tokens during masked language model training.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Vocabulary size — number of unique tokens the tokenizer knows.
# CPU NOTE: 30,522 matches DistilBERT's default vocabulary size.
# Using the same size means we can later initialize DistilBERT
# with our tokenizer without resizing its embedding matrix,
# which would require retraining the entire embedding layer.
VOCAB_SIZE = 30_522


def get_training_texts(
    processed_dir: str = "./ml/data/processed",
    max_texts:     int = 50_000,
) -> list[str]:
    """
    Loads clause texts from the processed dataset as training
    corpus for the tokenizer.

    We use clause_text rather than full_context because:
    1. Clause text is what the tokenizer will see at inference
    2. It's shorter and faster to iterate over
    3. 50k examples is more than enough for a good vocabulary
    """

    processed_path = Path(processed_dir)
    texts = []

    for split in ["train.json", "val.json", "test.json"]:
        fpath = processed_path / split
        if not fpath.exists():
            logger.warning(f"{split} not found — skipping")
            continue

        logger.info(f"Loading texts from {split}...")
        with open(fpath, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            text = record.get("clause_text", "").strip()
            if text:
                texts.append(text)

            # Also include full_context samples for richer vocabulary
            # Take first 1000 chars only — we just need the vocabulary
            context = record.get("full_context", "").strip()
            if context:
                texts.append(context[:1000])

            if len(texts) >= max_texts:
                break

        if len(texts) >= max_texts:
            break

    logger.info(f"Total training texts: {len(texts)}")
    return texts


def train_tokenizer(
    output_dir:    str = "./ml/models/embedder",
    processed_dir: str = "./ml/data/processed",
) -> None:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load training texts ──────────────────────────
    texts = get_training_texts(processed_dir)

    if not texts:
        raise ValueError("No training texts found. Run build_dataset.py first.")

    # ── Step 2: Build the tokenizer ──────────────────────────
    # We use the WordPiece model — same algorithm as BERT/DistilBERT.
    # This ensures our tokenizer is compatible with DistilBERT's
    # architecture without any adapter layers.
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    # Normalizer: lowercase + strip accents + clean text
    # This matches DistilBERT's bert-base-uncased preprocessing
    tokenizer.normalizer = normalizers.BertNormalizer(
        clean_text=True,
        handle_chinese_chars=False,  # contracts don't have Chinese
        strip_accents=True,
        lowercase=True,
    )

    # Pre-tokenizer: split on whitespace and punctuation
    # "indemnify." → ["indemnify", "."]
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # ── Step 3: Train ────────────────────────────────────────
    trainer = trainers.WordPieceTrainer(
        vocab_size        = VOCAB_SIZE,
        special_tokens    = SPECIAL_TOKENS,
        min_frequency     = 2,      # token must appear 2+ times to be kept
        continuing_subword_prefix = "##",  # DistilBERT convention for subwords
    )

    logger.info(f"Training WordPiece tokenizer on {len(texts)} texts...")
    logger.info(f"Target vocabulary size: {VOCAB_SIZE:,}")

    # Train directly from the list of strings
    tokenizer.train_from_iterator(texts, trainer=trainer)

    actual_vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Actual vocabulary size: {actual_vocab_size:,}")

    # ── Step 4: Add post-processor ───────────────────────────
    # Automatically adds [CLS] at start and [SEP] at end of
    # every encoded sequence — DistilBERT requires this format
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # Decoder: converts subword tokens back to readable text
    # "indem ##nifi ##cation" → "indemnification"
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    # ── Step 5: Save tokenizer ───────────────────────────────
    tokenizer_path = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    logger.info(f"Tokenizer saved to {tokenizer_path}")

    # Save vocab separately for easy inspection
    vocab = tokenizer.get_vocab()
    vocab_sorted = dict(sorted(vocab.items(), key=lambda x: x[1]))
    vocab_path = output_path / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_sorted, f, indent=2, ensure_ascii=False)
    logger.info(f"Vocabulary saved to {vocab_path}")

    # ── Step 6: Smoke test ───────────────────────────────────
    logger.info("\n=== SMOKE TEST ===")
    test_clauses = [
        "The indemnifying party shall hold harmless the indemnitee.",
        "Either party may terminate this agreement with 30 days notice.",
        "All intellectual property shall be assigned to the licensee.",
    ]

    for clause in test_clauses:
        encoded = tokenizer.encode(clause)
        decoded = tokenizer.decode(encoded.ids)
        logger.info(f"  Input  : {clause}")
        logger.info(f"  Tokens : {encoded.tokens}")
        logger.info(f"  IDs    : {encoded.ids[:10]}...")
        logger.info(f"  Decoded: {decoded}")
        logger.info(f"  Length : {len(encoded.tokens)} tokens")
        logger.info("")


if __name__ == "__main__":
    train_tokenizer()