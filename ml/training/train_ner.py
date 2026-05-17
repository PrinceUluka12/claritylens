# ml/training/train_ner.py
# ============================================================
# Fine-tunes DistilBERT for contract NER.
# Uses synthetically generated training data from CUAD text
# since CUAD doesn't have token-level NER annotations.
#
# Usage:
#   python ml/training/train_ner.py
#
# Expected runtime: 10-15 minutes on A10 GPU
# ============================================================

import json
import re
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    get_cosine_schedule_with_warmup,
)
from loguru import logger


# ── Config ────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_SEQ_LEN  = 256
BATCH_SIZE   = 32
EPOCHS       = 4
LEARNING_RATE = 3e-5
SEED          = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── NER label scheme (BIO format) ────────────────────────────
NER_LABELS = [
    "O",              # Outside — not an entity
    "B-PARTY",        # Beginning of party name
    "I-PARTY",        # Inside party name
    "B-DATE",         # Beginning of date
    "I-DATE",
    "B-AMOUNT",       # Beginning of dollar amount
    "I-AMOUNT",
    "B-JURISDICTION", # Beginning of governing law
    "I-JURISDICTION",
    "B-DURATION",     # Beginning of time period
    "I-DURATION",
    "B-TERMINATION_TRIGGER",
    "I-TERMINATION_TRIGGER",
    "B-IP_ASSET",
    "I-IP_ASSET",
]

LABEL2ID = {label: idx for idx, label in enumerate(NER_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(NER_LABELS)}
NUM_LABELS = len(NER_LABELS)

# Special token label — we ignore these in loss computation
IGNORE_INDEX = -100


# ── Synthetic data generation ─────────────────────────────────
# CUAD has no token-level NER annotations, so we generate
# synthetic training examples using pattern matching on
# real contract text from the processed dataset.

# Patterns for each entity type
PATTERNS = {
    "PARTY": [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|LLC|Corp|Ltd|LP|LLP|Co)\.?))\b',
        r'\bthe\s+(Licensor|Licensee|Company|Distributor|Vendor|Client|Contractor|Employee|Employer|Buyer|Seller|Franchisor|Franchisee)\b',
        r'\b(Party\s+[AB]|First\s+Party|Second\s+Party)\b',
    ],
    "DATE": [
        r'\b(\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Z][a-z]+,?\s+\d{4})\b',
        r'\b([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b',
        r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
        r'\b(\d{4}-\d{2}-\d{2})\b',
    ],
    "AMOUNT": [
        r'(\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand))?)',
        r'\b([\d,]+(?:\.\d{2})?\s+(?:dollars|USD))\b',
        r'\b(one|two|three|five|ten)\s+million\s+dollars\b',
    ],
    "JURISDICTION": [
        r'\blaws?\s+of\s+(?:the\s+)?(?:State\s+of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\bState\s+of\s+([A-Z][a-z]+)\b',
        r'\b([A-Z][a-z]+)\s+law\b',
    ],
    "DURATION": [
        r'\b(\d+[\-\s](?:year|month|day|week)s?(?:\s+period)?)\b',
        r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:year|month|day|week)s?\b',
        r'\b(\d+\s+(?:calendar|business)\s+days?)\b',
    ],
    "TERMINATION_TRIGGER": [
        r'\b(material\s+breach)\b',
        r'\b(insolvency|bankruptcy|dissolution)\b',
        r'\b(change\s+of\s+control)\b',
        r'\b(wilful\s+misconduct|gross\s+negligence)\b',
        r'\b(force\s+majeure)\b',
    ],
    "IP_ASSET": [
        r'\bthe\s+(Software|Platform|Technology|System|Product|Service|Application|API|SDK)\b',
        r'\b(Licensed\s+(?:Technology|Software|IP|Patents?|Marks?))\b',
        r'\b(Intellectual\s+Property|IP\s+Rights?|Patent\s+Rights?|Trade\s+Secrets?)\b',
    ],
}


def annotate_text(text: str) -> tuple:
    """
    Applies regex patterns to find entities in text.
    Returns (tokens, labels) where labels follow BIO scheme.

    This is synthetic annotation — not perfect, but good enough
    to teach the model what legal entities look like.
    """
    tokens = text.split()
    labels = ["O"] * len(tokens)

    # Rebuild text with token positions for span matching
    token_spans = []
    pos = 0
    for token in tokens:
        start = text.find(token, pos)
        end   = start + len(token)
        token_spans.append((start, end))
        pos = end

    # Apply each entity pattern
    for entity_type, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                match_start = match.start()
                match_end   = match.end()

                # Find which tokens overlap with this match
                matching_token_indices = []
                for i, (ts, te) in enumerate(token_spans):
                    if ts < match_end and te > match_start:
                        matching_token_indices.append(i)

                # Apply BIO labels
                for j, token_idx in enumerate(matching_token_indices):
                    if labels[token_idx] == "O":  # don't overwrite existing labels
                        if j == 0:
                            labels[token_idx] = f"B-{entity_type}"
                        else:
                            labels[token_idx] = f"I-{entity_type}"

    return tokens, labels


def generate_training_data(
    processed_dir: str = "./ml/data/processed",
    max_examples:  int = 8000,
) -> list:
    """
    Generates NER training examples from CUAD contract text.
    Each example is a dict with tokens and BIO labels.
    """
    processed_path = Path(processed_dir)
    examples = []

    fpath = processed_path / "train.json"
    with open(fpath) as f:
        records = json.load(f)

    random.shuffle(records)

    for record in records:
        text = record.get("clause_text", "")
        if not text or len(text.split()) < 10:
            continue

        # Take first 200 words to stay within token limit
        words = text.split()[:200]
        text  = " ".join(words)

        tokens, labels = annotate_text(text)

        # Only keep examples that have at least one entity
        if all(l == "O" for l in labels):
            continue

        examples.append({
            "tokens": tokens,
            "labels": labels,
        })

        if len(examples) >= max_examples:
            break

    logger.info(f"Generated {len(examples)} NER training examples")

    # Log entity distribution
    from collections import Counter
    entity_counts = Counter(
        l.replace("B-", "").replace("I-", "")
        for ex in examples
        for l in ex["labels"]
        if l != "O"
    )
    logger.info(f"Entity distribution: {dict(entity_counts.most_common())}")

    return examples


# ── Dataset ───────────────────────────────────────────────────
class NERDataset(Dataset):
    """
    Tokenizes pre-annotated examples for DistilBERT NER.

    Key challenge: DistilBERT's WordPiece tokenizer splits
    words into subwords. "Corporation" might become
    ["corporation"] or ["corp", "##oration"].
    We need to align our word-level labels to subword tokens.

    Strategy: assign the word's label to the FIRST subword,
    use IGNORE_INDEX (-100) for subsequent subwords.
    This is the standard approach for token classification.
    """

    def __init__(
        self,
        examples:  list,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = MAX_SEQ_LEN,
    ):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        tokens  = example["tokens"]
        labels  = example["labels"]

        # Tokenize word by word, tracking subword alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words = True,   # input is already tokenized
            max_length          = self.max_length,
            padding             = "max_length",
            truncation          = True,
            return_tensors      = "pt",
        )

        # Align labels to subword tokens
        word_ids      = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id  = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens [CLS], [SEP], [PAD]
                aligned_labels.append(IGNORE_INDEX)
            elif word_id != prev_word_id:
                # First subword of a word — use actual label
                label_str = labels[word_id] if word_id < len(labels) else "O"
                aligned_labels.append(LABEL2ID.get(label_str, 0))
            else:
                # Subsequent subwords — ignore in loss
                aligned_labels.append(IGNORE_INDEX)
            prev_word_id = word_id

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(aligned_labels, dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────
class NERModel(nn.Module):
    """
    DistilBERT with a token classification head.
    Outputs one label per token instead of one per sequence.
    """

    def __init__(self, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(MODEL_NAME)
        hidden_size     = self.distilbert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        outputs = self.distilbert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

        # Use ALL token outputs — not just [CLS]
        # Shape: [batch, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state

        # Classify each token
        # Shape: [batch, seq_len, num_labels]
        logits = self.classifier(sequence_output)

        return logits


# ── Evaluation ────────────────────────────────────────────────
def evaluate_ner(
    model:   NERModel,
    loader:  DataLoader,
    loss_fn: nn.CrossEntropyLoss,
) -> dict:
    """Computes loss and entity-level F1 on a dataloader."""
    model.eval()
    total_loss = 0.0

    # Track per-entity TP/FP/FN (ignoring O label)
    tp = {e: 0 for e in PATTERNS.keys()}
    fp = {e: 0 for e in PATTERNS.keys()}
    fn = {e: 0 for e in PATTERNS.keys()}

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)

            # Reshape for loss: [batch * seq_len, num_labels]
            loss = loss_fn(
                logits.view(-1, NUM_LABELS),
                labels.view(-1),
            )
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)  # [batch, seq_len]

            # Compare predictions to labels token by token
            for pred_seq, label_seq in zip(
                preds.tolist(), labels.tolist()
            ):
                for pred_id, label_id in zip(pred_seq, label_seq):
                    if label_id == IGNORE_INDEX:
                        continue

                    pred_label  = ID2LABEL.get(pred_id, "O")
                    true_label  = ID2LABEL.get(label_id, "O")

                    # Extract entity type from BIO label
                    pred_entity = pred_label.replace("B-","").replace("I-","")
                    true_entity = true_label.replace("B-","").replace("I-","")

                    if true_label != "O" and true_entity in tp:
                        if pred_entity == true_entity:
                            tp[true_entity] += 1
                        else:
                            fn[true_entity] += 1
                            if pred_label != "O" and pred_entity in fp:
                                fp[pred_entity] += 1

    avg_loss = total_loss / max(len(loader), 1)

    # Per-entity F1
    f1_scores = {}
    for entity in PATTERNS.keys():
        precision = tp[entity] / max(tp[entity] + fp[entity], 1)
        recall    = tp[entity] / max(tp[entity] + fn[entity], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_scores[entity] = round(f1, 3)

    macro_f1 = sum(f1_scores.values()) / len(f1_scores)

    return {
        "loss":      round(avg_loss, 4),
        "macro_f1":  round(macro_f1, 4),
        "f1_scores": f1_scores,
    }


# ── Main ──────────────────────────────────────────────────────
def train(
    processed_dir: str = "./ml/data/processed",
    output_dir:    str = "./ml/models/ner",
) -> None:

    torch.manual_seed(SEED)
    random.seed(SEED)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Generate training data ────────────────────────────────
    logger.info("Generating NER training data from CUAD...")
    all_examples = generate_training_data(processed_dir)

    if len(all_examples) < 100:
        raise ValueError("Too few training examples generated.")

    # 80/10/10 split
    random.shuffle(all_examples)
    n       = len(all_examples)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)

    train_examples = all_examples[:n_train]
    val_examples   = all_examples[n_train : n_train + n_val]
    test_examples  = all_examples[n_train + n_val:]

    logger.info(
        f"Split — train={len(train_examples)}, "
        f"val={len(val_examples)}, test={len(test_examples)}"
    )

    # ── Tokenizer and datasets ────────────────────────────────
    tokenizer     = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = NERDataset(train_examples, tokenizer)
    val_dataset   = NERDataset(val_examples,   tokenizer)
    test_dataset  = NERDataset(test_examples,  tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(DEVICE.type=="cuda"),
        num_workers=4 if DEVICE.type=="cuda" else 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE*2,
        pin_memory=(DEVICE.type=="cuda"),
        num_workers=4 if DEVICE.type=="cuda" else 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE*2,
        pin_memory=(DEVICE.type=="cuda"),
        num_workers=4 if DEVICE.type=="cuda" else 0,
    )

    # ── Model ─────────────────────────────────────────────────
    logger.info("Loading DistilBERT for token classification...")
    model   = NERModel(num_labels=NUM_LABELS).to(DEVICE)

    # Ignore padding and subword continuation tokens in loss
    loss_fn   = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    logger.info(
        f"Training — epochs={EPOCHS}, "
        f"steps/epoch={len(train_loader)}, "
        f"total={total_steps}"
    )

    # ── Training loop ─────────────────────────────────────────
    best_f1        = 0.0
    best_path      = output_path / "best_ner_model.pt"
    start_time     = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)

            loss = loss_fn(
                logits.view(-1, NUM_LABELS),
                labels.view(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            epoch_steps += 1

            if (step + 1) % 30 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss {epoch_loss/epoch_steps:.4f} | "
                    f"Elapsed {elapsed:.0f}s"
                )

        # Validation
        val_metrics = evaluate_ner(model, val_loader, loss_fn)
        logger.info(
            f"Epoch {epoch+1} val — "
            f"loss={val_metrics['loss']} | "
            f"macro_f1={val_metrics['macro_f1']}"
        )
        logger.info(f"  Per-entity F1: {val_metrics['f1_scores']}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)
            logger.info(f"  Best model saved — macro_f1={best_f1:.4f}")

    # ── Test evaluation ───────────────────────────────────────
    logger.info("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(best_path))
    test_metrics = evaluate_ner(model, test_loader, loss_fn)

    logger.info("=== NER TEST RESULTS ===")
    logger.info(f"  Macro F1 : {test_metrics['macro_f1']}")
    for entity, f1 in test_metrics["f1_scores"].items():
        logger.info(f"    {entity:<25} {f1:.3f}")

    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time:.0f}s")

    # ── Quantize and save ─────────────────────────────────────
    logger.info("Applying int8 quantization...")
    model.cpu()
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    quant_path = output_path / "ner_quantized.pt"
    torch.save(quantized.state_dict(), quant_path)

    # Save label config for inference
    label_config = {
        "ner_labels": NER_LABELS,
        "label2id":   LABEL2ID,
        "id2label":   {str(k): v for k, v in ID2LABEL.items()},
        "num_labels": NUM_LABELS,
    }
    with open(output_path / "ner_label_config.json", "w") as f:
        json.dump(label_config, f, indent=2)

    orig_mb  = best_path.stat().st_size / 1e6
    quant_mb = quant_path.stat().st_size / 1e6
    logger.info(
        f"Model size — original: {orig_mb:.1f}MB | "
        f"quantized: {quant_mb:.1f}MB"
    )
    logger.info("NER training complete.")


if __name__ == "__main__":
    train()