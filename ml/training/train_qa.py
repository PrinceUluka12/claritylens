# ml/training/train_qa.py
# ============================================================
# Fine-tunes DistilBERT for extractive QA on CUAD contracts.
# Uses the SQuAD-format span annotations already in CUAD.
#
# Usage:
#   python ml/training/train_qa.py
#
# Expected runtime: 20-30 minutes on A10 GPU
# ============================================================

import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    get_cosine_schedule_with_warmup,
)
from loguru import logger


# ── Config ────────────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-uncased"
MAX_SEQ_LEN   = 384      # QA needs longer context than classification
MAX_QUERY_LEN = 64       # question tokens
DOC_STRIDE    = 128      # overlap between context windows
BATCH_SIZE    = 16       # smaller batch — QA sequences are longer
EPOCHS        = 3
LEARNING_RATE = 3e-5
SEED          = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────
class QADataset(Dataset):
    """
    Converts CUAD records into SQuAD-format QA examples.

    Each record has:
      context:  full contract text (may be very long)
      question: clause category question
      answers:  dict with 'text' and 'answer_start' lists

    DistilBERT QA expects:
      input_ids:      [CLS] question [SEP] context [SEP]
      attention_mask: 1 for real tokens, 0 for padding
      start_positions: token index of answer start
      end_positions:   token index of answer end
    """

    def __init__(
        self,
        records:    list,
        tokenizer:  DistilBertTokenizerFast,
        max_length: int = MAX_SEQ_LEN,
        doc_stride: int = DOC_STRIDE,
    ):
        self.examples  = []
        self.tokenizer = tokenizer

        for record in records:
            question = record.get("question", "").strip()
            context  = record.get("context",  "").strip()
            answers  = record.get("answers",  {})

            if not question or not context:
                continue

            # Parse answers — handle both string and dict formats
            if isinstance(answers, str):
                try:
                    answers = json.loads(answers.replace("'", '"'))
                except Exception:
                    answers = {}

            answer_texts  = answers.get("text", [])
            answer_starts = answers.get("answer_start", [])

            # Skip records with no answer
            if not answer_texts or not answer_starts:
                continue

            answer_text  = answer_texts[0]
            answer_start = int(answer_starts[0])
            answer_end   = answer_start + len(answer_text)

            # Tokenize question + context together
            # DistilBERT QA format: [CLS] Q [SEP] Context [SEP]
            encoding = tokenizer(
                question,
                context,
                max_length         = max_length,
                stride             = doc_stride,
                truncation         = "only_second",  # truncate context, not question
                padding            = "max_length",
                return_offsets_mapping = True,
                return_tensors     = "pt",
            )

            offset_mapping  = encoding["offset_mapping"].squeeze(0)
            input_ids       = encoding["input_ids"].squeeze(0)
            attention_mask  = encoding["attention_mask"].squeeze(0)

            # Find token positions of answer span using offset mapping
            # offset_mapping[i] = (char_start, char_end) for token i
            start_position = 0
            end_position   = 0

            # Token sequence: [CLS] Q tokens [SEP] Context tokens [SEP]
            # We need to find which context tokens cover the answer
            sequence_ids = encoding.sequence_ids(0)

            # Find where context tokens start (sequence_id == 1)
            context_start_token = 0
            context_end_token   = 0
            for i, sid in enumerate(sequence_ids):
                if sid == 1:
                    context_start_token = i
                    break
            for i in range(len(sequence_ids) - 1, -1, -1):
                if sequence_ids[i] == 1:
                    context_end_token = i
                    break

            # Find answer token positions within context
            found = False
            for i in range(context_start_token, context_end_token + 1):
                token_start, token_end = offset_mapping[i].tolist()
                if token_start <= answer_start < token_end:
                    start_position = i
                    found = True
                if token_start < answer_end <= token_end:
                    end_position = i
                    break

            # If answer span not found in this window, use CLS token
            # (signals unanswerable — model learns to return 0,0)
            if not found:
                start_position = 0
                end_position   = 0

            self.examples.append({
                "input_ids":       input_ids,
                "attention_mask":  attention_mask,
                "start_positions": torch.tensor(start_position, dtype=torch.long),
                "end_positions":   torch.tensor(end_position,   dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ── Training utilities ────────────────────────────────────────
def evaluate_qa(
    model:   DistilBertForQuestionAnswering,
    loader:  DataLoader,
) -> dict:
    """
    Evaluates QA model on a dataloader.
    Returns exact match and average loss.
    """
    model.eval()
    total_loss = 0.0
    exact_match = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids       = batch["input_ids"].to(DEVICE)
            attention_mask  = batch["attention_mask"].to(DEVICE)
            start_positions = batch["start_positions"].to(DEVICE)
            end_positions   = batch["end_positions"].to(DEVICE)

            outputs = model(
                input_ids       = input_ids,
                attention_mask  = attention_mask,
                start_positions = start_positions,
                end_positions   = end_positions,
            )

            total_loss += outputs.loss.item()

            # Check exact match on start and end positions
            pred_starts = outputs.start_logits.argmax(dim=-1)
            pred_ends   = outputs.end_logits.argmax(dim=-1)

            for ps, pe, gs, ge in zip(
                pred_starts.tolist(), pred_ends.tolist(),
                start_positions.tolist(), end_positions.tolist()
            ):
                if ps == gs and pe == ge:
                    exact_match += 1
                total += 1

    return {
        "loss":  round(total_loss / max(len(loader), 1), 4),
        "exact_match": round(exact_match / max(total, 1), 4),
    }


# ── Main ──────────────────────────────────────────────────────
def train(
    cuad_dir:   str = "./ml/data/cuad",
    output_dir: str = "./ml/models/qa",
) -> None:

    torch.manual_seed(SEED)
    random.seed(SEED)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load CUAD data ────────────────────────────────────────
    # Use raw CUAD train/test splits — they have answer spans
    logger.info("Loading CUAD QA data...")
    records = []
    for split in ["train.json", "test.json"]:
        fpath = Path(cuad_dir) / split
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            # Only keep records with answers
            records.extend([
                r for r in data
                if r.get("answers") and
                isinstance(r.get("answers"), (str, dict))
            ])

    random.shuffle(records)
    logger.info(f"Total QA records with answers: {len(records)}")

    # Limit to 8000 for manageable training time
    records = records[:8000]

    # 80/10/10 split
    n       = len(records)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)

    train_records = records[:n_train]
    val_records   = records[n_train : n_train + n_val]
    test_records  = records[n_train + n_val:]

    # ── Tokenizer and datasets ────────────────────────────────
    logger.info("Loading tokenizer and building datasets...")
    tokenizer     = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = QADataset(train_records, tokenizer)
    val_dataset   = QADataset(val_records,   tokenizer)
    test_dataset  = QADataset(test_records,  tokenizer)

    logger.info(
        f"Dataset sizes — "
        f"train={len(train_dataset)}, "
        f"val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )

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
    # DistilBertForQuestionAnswering has a QA head built in —
    # two linear layers for start and end logits
    logger.info("Loading DistilBERT for QA...")
    model     = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME)
    model     = model.to(DEVICE)

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
    best_val_loss  = float("inf")
    best_path      = output_path / "best_qa_model.pt"
    start_time     = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_loader):
            input_ids       = batch["input_ids"].to(DEVICE)
            attention_mask  = batch["attention_mask"].to(DEVICE)
            start_positions = batch["start_positions"].to(DEVICE)
            end_positions   = batch["end_positions"].to(DEVICE)

            optimizer.zero_grad()

            # DistilBertForQuestionAnswering computes loss internally
            # when start_positions and end_positions are provided
            outputs = model(
                input_ids       = input_ids,
                attention_mask  = attention_mask,
                start_positions = start_positions,
                end_positions   = end_positions,
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            epoch_steps += 1

            if (step + 1) % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"  Epoch {epoch+1}/{EPOCHS} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss {epoch_loss/epoch_steps:.4f} | "
                    f"Elapsed {elapsed:.0f}s"
                )

        # Validation
        val_metrics = evaluate_qa(model, val_loader)
        logger.info(
            f"Epoch {epoch+1} val — "
            f"loss={val_metrics['loss']} | "
            f"exact_match={val_metrics['exact_match']}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)
            logger.info(f"  Best model saved — val_loss={best_val_loss:.4f}")

    # ── Test evaluation ───────────────────────────────────────
    logger.info("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(best_path))
    test_metrics = evaluate_qa(model, test_loader)

    logger.info("=== QA TEST RESULTS ===")
    logger.info(f"  Loss        : {test_metrics['loss']}")
    logger.info(f"  Exact Match : {test_metrics['exact_match']}")

    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time:.0f}s")

    # ── Quantize ──────────────────────────────────────────────
    logger.info("Applying int8 quantization...")
    model.cpu()
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    quant_path = output_path / "qa_quantized.pt"
    torch.save(quantized.state_dict(), quant_path)

    orig_mb  = best_path.stat().st_size / 1e6
    quant_mb = quant_path.stat().st_size / 1e6
    logger.info(
        f"Model size — original: {orig_mb:.1f}MB | "
        f"quantized: {quant_mb:.1f}MB"
    )
    logger.info("QA training complete.")


if __name__ == "__main__":
    train()