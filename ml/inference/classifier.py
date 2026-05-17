# ml/inference/classifier.py
# ============================================================
# Loads the quantized DistilBERT classifier and runs inference
# on clause chunks. Called by backend/services/ at request time.
#
# CPU NOTE: model is quantized int8 and loaded once at startup.
# All inference runs inside torch.no_grad() to skip gradient
# computation — this alone saves ~30% of inference time on CPU.
# ============================================================

import json
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel
from loguru import logger


# ── Data classes ──────────────────────────────────────────────
@dataclass
class ClauseResult:
    """Result for a single clause chunk."""
    chunk_index:  int
    text:         str
    risk_label:   str
    confidence:   float        # 0.0 – 1.0
    all_scores:   dict         # {label: score} for all 6 classes


@dataclass
class DocumentResult:
    """Aggregated result for a full document."""
    filename:      str
    chunk_results: list[ClauseResult]
    risk_summary:  dict        # {label: max_confidence} across all chunks
    top_risk:      str         # highest confidence risk label
    inference_ms:  float


# ── Model definition (must match train_classifier.py exactly) ─
class ClauseClassifier(nn.Module):
    """
    Identical architecture to training — required to load weights.
    If you change the architecture in training, change it here too.
    """

    def __init__(self, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased",
            # CPU NOTE: don't load full precision weights if
            # we're about to quantize anyway
        )
        hidden_size = self.distilbert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs    = self.distilbert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


# ── Inference class ───────────────────────────────────────────
class ClassifierInference:
    """
    Singleton inference wrapper.
    Loaded once at server startup via backend/main.py lifespan.
    Reused for every request — never reloaded mid-request.
    """

    def __init__(
        self,
        model_dir:  str | Path = "./ml/models/classifier",
        max_length: int = 256,
        batch_size: int = 8,
    ) -> None:

        model_dir  = Path(model_dir)
        self.max_length = max_length
        self.batch_size = batch_size

        # ── Load label registry ───────────────────────────────
        registry_path = model_dir / "label_registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(
                f"label_registry.json not found in {model_dir}. "
                f"Run train_classifier.py first."
            )

        with open(registry_path) as f:
            registry = json.load(f)

        self.label_map    = registry["label_to_id"]   # label → int
        self.id_to_label  = {
            int(k): v for k, v in registry["id_to_label"].items()
        }
        self.num_labels   = len(self.label_map)

        # ── Load tokenizer ────────────────────────────────────
        logger.info("Loading DistilBERT tokenizer...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )

        # ── Load quantized model ──────────────────────────────
        quantized_path = model_dir / "classifier_quantized.pt"
        full_path      = model_dir / "best_model.pt"

        if quantized_path.exists():
            model_path = quantized_path
            logger.info("Loading quantized int8 model...")
        elif full_path.exists():
            model_path = full_path
            logger.info("Loading full precision model (quantized not found)...")
        else:
            raise FileNotFoundError(
                f"No model weights found in {model_dir}. "
                f"Run train_classifier.py first."
            )

        # Build model architecture then load weights
        model = ClauseClassifier(num_labels=self.num_labels)

        # CPU NOTE: map_location='cpu' ensures weights load on CPU
        # even if they were saved from a GPU training run
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        # CPU NOTE: apply dynamic quantization if loading full model
        if model_path == full_path:
            logger.info("Applying dynamic quantization...")
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )

        # CPU NOTE: eval() disables dropout layers which are only
        # needed during training — inference is deterministic
        model.eval()
        self.model = model

        logger.info(
            f"ClassifierInference ready — "
            f"{self.num_labels} labels, "
            f"max_length={max_length}, "
            f"batch_size={batch_size}"
        )

    def _tokenize_batch(self, texts: list[str]) -> dict:
        """Tokenizes a list of texts into padded tensors."""
        return self.tokenizer(
            texts,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )

    def predict(
        self,
        chunks: list,           # list of Chunk objects or plain strings
        filename: str = "",
    ) -> DocumentResult:
        """
        Runs risk classification on all chunks of a document.

        Args:
            chunks:   List of Chunk objects (from chunker.py) or strings
            filename: Document name for the result object

        Returns:
            DocumentResult with per-chunk labels and document summary
        """

        start_time = time.time()

        # Normalize input — accept both Chunk objects and plain strings
        texts = []
        for c in chunks:
            if hasattr(c, "text"):
                texts.append(c.text)
            else:
                texts.append(str(c))

        if not texts:
            logger.warning("predict() called with empty chunk list")
            return DocumentResult(
                filename      = filename,
                chunk_results = [],
                risk_summary  = {},
                top_risk      = "unknown",
                inference_ms  = 0.0,
            )

        # ── Batch inference ───────────────────────────────────
        # CPU NOTE: process in batches of batch_size (default 8)
        # to avoid loading all chunks into memory at once.
        # torch.no_grad() skips gradient tape — critical for
        # CPU inference speed.
        all_results = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start : batch_start + self.batch_size]

            encoding = self._tokenize_batch(batch_texts)
            input_ids      = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # CPU NOTE: torch.no_grad() is the single most
            # important CPU optimization for inference —
            # never run inference without it
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)  # [batch, num_labels]

            for i, (text, prob_vec) in enumerate(
                zip(batch_texts, probs.tolist())
            ):
                chunk_idx     = batch_start + i
                pred_id       = prob_vec.index(max(prob_vec))
                pred_label    = self.id_to_label[pred_id]
                confidence    = max(prob_vec)

                all_scores = {
                    self.id_to_label[j]: round(p, 4)
                    for j, p in enumerate(prob_vec)
                }

                all_results.append(ClauseResult(
                    chunk_index = chunk_idx,
                    text        = text,
                    risk_label  = pred_label,
                    confidence  = round(confidence, 4),
                    all_scores  = all_scores,
                ))

        # ── Aggregate to document level ───────────────────────
        # For each risk label, take the maximum confidence
        # across all chunks — a clause only needs to appear
        # once to flag the document
        risk_summary = {label: 0.0 for label in self.label_map}

        for result in all_results:
            for label, score in result.all_scores.items():
                if label in risk_summary:
                    risk_summary[label] = max(risk_summary[label], score)

        top_risk = max(risk_summary, key=risk_summary.get)

        inference_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Classified {len(texts)} chunks in {inference_ms:.1f}ms | "
            f"top_risk={top_risk} ({risk_summary[top_risk]:.3f})"
        )

        return DocumentResult(
            filename      = filename,
            chunk_results = all_results,
            risk_summary  = risk_summary,
            top_risk      = top_risk,
            inference_ms  = inference_ms,
        )