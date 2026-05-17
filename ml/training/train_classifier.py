# ml/training/train_classifier.py
# ============================================================
# Fine-tunes DistilBERT for clause risk classification.
# Runs on GPU (Lambda) — download weights after training.
#
# Usage:
#   python ml/training/train_classifier.py
#
# Expected runtime: 15-25 minutes on A10 GPU
# ============================================================

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from transformers import DistilBertTokenizerFast, DistilBertModel
from loguru import logger


# ── Config ───────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_SEQ_LEN  = 256       # CPU NOTE: hard limit, do not raise
BATCH_SIZE   = 32        # GPU NOTE: 32 fits comfortably on A10
EPOCHS       = 3
LEARNING_RATE = 2e-5     # Standard fine-tuning LR for BERT-family
WARMUP_RATIO  = 0.1      # 10% of steps used for LR warmup
SEED          = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────
class ClauseDataset(Dataset):
    """
    Loads processed CUAD records and tokenizes clause text.
    Each record becomes one training example:
      input:  clause_text tokenized to 256 tokens
      label:  integer risk label ID (0-5)
      weight: used for loss weighting (handles class imbalance)
    """

    def __init__(
        self,
        records:    list,
        tokenizer:  DistilBertTokenizerFast,
        label_map:  dict,
        max_length: int = MAX_SEQ_LEN,
    ):
        self.records    = records
        self.tokenizer  = tokenizer
        self.label_map  = label_map
        self.max_length = max_length

        # Filter out 'none' labels — metadata rows
        # are not useful for risk classification
        self.records = [
            r for r in records
            if r.get("risk_label") in label_map
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Use clause_text — this is the actual clause span
        # or the first 500 chars for negative examples
        text = record.get("clause_text", "")
        if not text:
            text = record.get("full_context", "")[:500]

        label_str = record.get("risk_label", "")
        label_id  = self.label_map.get(label_str, 0)

        # Tokenize with DistilBERT's own tokenizer
        # (NOT our custom tokenizer — see Phase 4 explanation)
        encoding = self.tokenizer(
            text,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label_id, dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────
class ClauseClassifier(nn.Module):
    """
    DistilBERT + classification head.

    Architecture:
      DistilBERT [CLS] token → Dropout → Linear → 6 logits

    Why use [CLS] token output?
    DistilBERT is trained so the [CLS] token aggregates
    information from the entire sequence. It's the standard
    approach for sentence-level classification tasks.
    """

    def __init__(self, num_labels: int, dropout: float = 0.1):
        super().__init__()

        self.distilbert = DistilBertModel.from_pretrained(MODEL_NAME)

        # Classification head
        hidden_size = self.distilbert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),  # 768 → 384
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),   # 384 → 6
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        # CPU NOTE: torch.no_grad() is used at inference time
        # not during training — gradients are needed for backprop
        outputs = self.distilbert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

        # [CLS] token is always at position 0
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        logits     = self.classifier(cls_output)          # [batch, 6]

        return logits


# ── Training utilities ────────────────────────────────────────
def compute_class_weights(
    records:   list,
    label_map: dict,
) -> torch.Tensor:
    """
    Computes inverse frequency weights for each class.
    Rare classes get higher weights so the model gets
    penalized more for missing them.

    Formula: weight = total_samples / (num_classes * class_count)
    """
    from collections import Counter

    counts = Counter(
        r["risk_label"] for r in records
        if r.get("risk_label") in label_map
    )

    num_classes  = len(label_map)
    total        = sum(counts.values())
    weights      = torch.zeros(num_classes)

    for label, idx in label_map.items():
        count       = counts.get(label, 1)
        weights[idx] = total / (num_classes * count)

    logger.info(f"Class weights: { {l: f'{weights[i]:.3f}' for l, i in label_map.items()} }")
    return weights


def evaluate(
    model:      ClauseClassifier,
    loader:     DataLoader,
    loss_fn:    nn.CrossEntropyLoss,
    label_map:  dict,
) -> dict:
    """
    Runs evaluation on a dataloader.
    Returns loss, accuracy, and per-class F1.
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    # Per-class tracking for F1
    id_to_label = {v: k for k, v in label_map.items()}
    tp = {i: 0 for i in range(len(label_map))}
    fp = {i: 0 for i in range(len(label_map))}
    fn = {i: 0 for i in range(len(label_map))}

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss   = loss_fn(logits, labels)

            preds = logits.argmax(dim=-1)

            total_loss += loss.item()
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            # Accumulate per-class TP/FP/FN
            for pred, label in zip(preds.tolist(), labels.tolist()):
                if pred == label:
                    tp[label] += 1
                else:
                    fp[pred]  += 1
                    fn[label] += 1

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)

    # Per-class F1
    f1_scores = {}
    for i in range(len(label_map)):
        precision = tp[i] / max(tp[i] + fp[i], 1)
        recall    = tp[i] / max(tp[i] + fn[i], 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_scores[id_to_label[i]] = round(f1, 3)

    macro_f1 = sum(f1_scores.values()) / len(f1_scores)

    return {
        "loss":      round(avg_loss, 4),
        "accuracy":  round(accuracy, 4),
        "macro_f1":  round(macro_f1, 4),
        "f1_scores": f1_scores,
    }


# ── Main training function ────────────────────────────────────
def train(
    processed_dir: str = "./ml/data/processed",
    output_dir:    str = "./ml/models/classifier",
) -> None:

    torch.manual_seed(SEED)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load label registry ───────────────────────────────────
    registry_path = Path(processed_dir) / "label_registry.json"
    with open(registry_path) as f:
        registry = json.load(f)

    label_map   = registry["label_to_id"]   # e.g. {"indemnity": 1, ...}
    num_labels  = len(label_map)
    logger.info(f"Labels: {label_map}")

    # ── Load tokenizer ────────────────────────────────────────
    # Using DistilBERT's own tokenizer — NOT our custom one
    logger.info("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # ── Load datasets ─────────────────────────────────────────
    def load_split(split_name: str) -> list:
        path = Path(processed_dir) / f"{split_name}.json"
        with open(path) as f:
            return json.load(f)

    logger.info("Loading datasets...")
    train_records = load_split("train")
    val_records   = load_split("val")
    test_records  = load_split("test")

    train_dataset = ClauseDataset(train_records, tokenizer, label_map)
    val_dataset   = ClauseDataset(val_records,   tokenizer, label_map)
    test_dataset  = ClauseDataset(test_records,  tokenizer, label_map)

    logger.info(
        f"Dataset sizes — "
        f"train={len(train_dataset)}, "
        f"val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        pin_memory  = (DEVICE.type == "cuda"),
        num_workers = 4 if DEVICE.type == "cuda" else 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE * 2,
        pin_memory  = (DEVICE.type == "cuda"),
        num_workers = 4 if DEVICE.type == "cuda" else 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE * 2,
        pin_memory  = (DEVICE.type == "cuda"),
        num_workers = 4 if DEVICE.type == "cuda" else 0,
    )

    # ── Class weights for imbalanced data ────────────────────
    class_weights = compute_class_weights(
        train_records, label_map
    ).to(DEVICE)

    # ── Model ─────────────────────────────────────────────────
    logger.info("Loading DistilBERT model...")
    model   = ClauseClassifier(num_labels=num_labels).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer and scheduler ───────────────────────────────
    # AdamW is the standard optimizer for transformer fine-tuning.
    # It adds weight decay to Adam which prevents overfitting
    # on small datasets like CUAD.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)

    # Linear warmup then linear decay
    scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = warmup_steps,
    num_training_steps = total_steps,
)

    logger.info(
        f"Training — epochs={EPOCHS}, "
        f"steps/epoch={len(train_loader)}, "
        f"total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}"
    )

    # ── Training loop ─────────────────────────────────────────
    best_val_f1   = 0.0
    best_model_path = output_path / "best_model.pt"
    start_time    = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss   = 0.0
        epoch_steps  = 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = loss_fn(logits, labels)
            loss.backward()

            # Gradient clipping prevents exploding gradients
            # which can destabilize transformer fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            epoch_steps += 1

            if (step + 1) % 50 == 0:
                avg = epoch_loss / epoch_steps
                elapsed = time.time() - start_time
                logger.info(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss {avg:.4f} | "
                    f"Elapsed {elapsed:.0f}s"
                )

        # ── Validation ────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, loss_fn, label_map)
        logger.info(
            f"Epoch {epoch+1} val — "
            f"loss={val_metrics['loss']} | "
            f"acc={val_metrics['accuracy']} | "
            f"macro_f1={val_metrics['macro_f1']}"
        )
        logger.info(f"  Per-class F1: {val_metrics['f1_scores']}")

        # Save best model by macro F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"  New best model saved — macro_f1={best_val_f1:.4f}"
            )

    # ── Test evaluation ───────────────────────────────────────
    logger.info("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(model, test_loader, loss_fn, label_map)

    logger.info("=== TEST RESULTS ===")
    logger.info(f"  Loss     : {test_metrics['loss']}")
    logger.info(f"  Accuracy : {test_metrics['accuracy']}")
    logger.info(f"  Macro F1 : {test_metrics['macro_f1']}")
    logger.info(f"  Per-class F1:")
    for label, f1 in test_metrics["f1_scores"].items():
        logger.info(f"    {label:<20} {f1:.3f}")

    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time:.0f}s")

    # ── Apply int8 quantization ───────────────────────────────
    # CPU NOTE: quantize_dynamic converts float32 → int8 for
    # Linear layers. This runs on CPU so we move the model first.
    # Result: 4x smaller model, 2-3x faster CPU inference.
    logger.info("Applying int8 quantization for CPU inference...")
    model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},    # quantize all Linear layers
        dtype=torch.qint8,
    )

    quantized_path = output_path / "classifier_quantized.pt"
    torch.save(quantized_model.state_dict(), quantized_path)
    logger.info(f"Quantized model saved → {quantized_path}")

    # Save label registry alongside model for inference
    import shutil
    shutil.copy(registry_path, output_path / "label_registry.json")

    # ── Size comparison ───────────────────────────────────────
    orig_mb  = best_model_path.stat().st_size / 1e6
    quant_mb = quantized_path.stat().st_size / 1e6
    logger.info(f"Model size — original: {orig_mb:.1f}MB | quantized: {quant_mb:.1f}MB")


if __name__ == "__main__":
    train()