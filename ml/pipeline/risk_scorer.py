# ml/pipeline/risk_scorer.py
# ============================================================
# Computes document-level risk scores from classifier output.
# Pure Python — no models, no torch, instant execution.
#
# Risk score = weighted combination of:
#   - Clause presence flags (is this risk type present?)
#   - Classifier confidence (how certain is the model?)
#   - Entity signals (are there red-flag entities?)
#   - Absence penalties (missing clauses that SHOULD be present)
# ============================================================

from dataclasses import dataclass, field
from loguru import logger


# ── Risk weights ──────────────────────────────────────────────
# How much each category contributes to the overall score.
# Based on standard contract risk frameworks —
# indemnity and liability_cap are the highest-stakes clauses.
RISK_WEIGHTS = {
    "indemnity":     0.25,
    "liability_cap": 0.20,
    "ip_assignment": 0.20,
    "termination":   0.15,
    "non_compete":   0.12,
    "data_privacy":  0.08,
}

# Clauses that SHOULD be present in a well-drafted contract.
# Their absence is itself a risk signal.
EXPECTED_CLAUSES = {
    "liability_cap",   # missing cap = uncapped liability risk
    "termination",     # missing termination = locked-in forever
    "data_privacy",    # missing privacy clause = compliance risk
}

# Confidence threshold — below this we treat a prediction
# as uncertain and reduce its contribution to the score.
# This directly addresses the indemnity/liability_cap overlap.
CONFIDENCE_THRESHOLD = 0.35

# Entity red-flag signals — presence of these bumps risk score
ENTITY_RISK_SIGNALS = {
    "TERMINATION_TRIGGER": 0.05,   # material breach, insolvency etc.
    "AMOUNT":              0.02,   # dollar amounts signal financial risk
}


@dataclass
class CategoryRisk:
    """Risk assessment for a single clause category."""
    label:          str
    present:        bool          # was this clause type found?
    max_confidence: float         # highest classifier confidence
    chunk_count:    int           # how many chunks flagged this
    risk_score:     float         # 0.0 – 1.0 contribution
    top_clause:     str           # highest-confidence clause text
    entities:       list = field(default_factory=list)


@dataclass
class DocumentRisk:
    """Complete risk assessment for a document."""
    filename:         str
    overall_score:    float              # 0–100
    risk_level:       str               # LOW / MEDIUM / HIGH / CRITICAL
    category_risks:   list[CategoryRisk]
    missing_clauses:  list[str]         # expected but not found
    entity_summary:   dict              # entity type → list of values
    top_risks:        list[str]         # top 3 risk labels by score
    inference_ms:     float


class RiskScorer:
    """
    Computes document risk scores from ML pipeline outputs.
    Stateless — one instance handles all documents.
    """

    def __init__(self) -> None:
        logger.info("RiskScorer initialized")

    def score(
        self,
        filename:        str,
        chunk_results:   list,    # list of ClauseResult from classifier
        ner_results:     list,    # list of dicts from NER inference
        inference_ms:    float = 0.0,
    ) -> DocumentRisk:
        """
        Computes full risk assessment for a document.

        Args:
            filename:      document name
            chunk_results: ClauseResult objects from ClassifierInference
            ner_results:   NER entity dicts from NERInference
            inference_ms:  total pipeline time so far

        Returns:
            DocumentRisk with overall score and per-category breakdown
        """
        import time
        start = time.time()

        # ── Step 1: Aggregate classifier results by category ──
        # For each risk label, collect all chunks that were
        # classified as that label with their confidence scores
        from collections import defaultdict
        category_chunks = defaultdict(list)

        for result in chunk_results:
            label      = result.risk_label
            confidence = result.confidence
            text       = result.text

            # Apply dual-label logic for indemnity/liability_cap overlap
            # If both scores are above threshold, flag both categories
            if hasattr(result, "all_scores"):
                indem_score   = result.all_scores.get("indemnity", 0)
                liab_score    = result.all_scores.get("liability_cap", 0)
                if (indem_score > CONFIDENCE_THRESHOLD and
                    liab_score  > CONFIDENCE_THRESHOLD):
                    category_chunks["indemnity"].append(
                        (indem_score, text)
                    )
                    category_chunks["liability_cap"].append(
                        (liab_score, text)
                    )
                    continue

            if confidence >= CONFIDENCE_THRESHOLD and label in RISK_WEIGHTS:
                category_chunks[label].append((confidence, text))

        # ── Step 2: Build per-category risk objects ───────────
        category_risks = []
        weighted_total = 0.0

        for label, weight in RISK_WEIGHTS.items():
            chunks = category_chunks.get(label, [])

            if chunks:
                max_conf   = max(c[0] for c in chunks)
                top_clause = max(chunks, key=lambda x: x[0])[1]
                present    = True
            else:
                max_conf   = 0.0
                top_clause = ""
                present    = False

            # Category risk score formula:
            # base = max confidence if present, penalty if absent
            # Scale: present clause → confidence * weight
            #        absent expected clause → 0.8 * weight (high risk)
            #        absent optional clause → 0.0
            if present:
                category_score = max_conf
            elif label in EXPECTED_CLAUSES:
                # Missing a clause that should be there is HIGH risk
                category_score = 0.8
            else:
                category_score = 0.0

            weighted_contribution = category_score * weight
            weighted_total       += weighted_contribution

            category_risks.append(CategoryRisk(
                label          = label,
                present        = present,
                max_confidence = round(max_conf, 4),
                chunk_count    = len(chunks),
                risk_score     = round(category_score, 4),
                top_clause     = top_clause[:500] if top_clause else "",
            ))

        # ── Step 3: Entity signals ────────────────────────────
        # Parse NER results and check for red-flag entities
        entity_summary = defaultdict(list)
        entity_bonus   = 0.0

        for ner_result in ner_results:
            entity_type  = ner_result.get("entity_type", "")
            entity_value = ner_result.get("text", "")

            if entity_value:
                entity_summary[entity_type].append(entity_value)

            if entity_type in ENTITY_RISK_SIGNALS:
                entity_bonus += ENTITY_RISK_SIGNALS[entity_type]

        # Cap entity bonus at 0.10
        entity_bonus = min(entity_bonus, 0.10)

        # ── Step 4: Compute overall score ─────────────────────
        # weighted_total is 0.0–1.0, convert to 0–100
        raw_score     = (weighted_total + entity_bonus) * 100
        overall_score = round(min(raw_score, 100.0), 1)

        # ── Step 5: Risk level thresholds ─────────────────────
        if overall_score >= 75:
            risk_level = "CRITICAL"
        elif overall_score >= 50:
            risk_level = "HIGH"
        elif overall_score >= 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # ── Step 6: Missing clauses ───────────────────────────
        missing_clauses = [
            label for label in EXPECTED_CLAUSES
            if not any(
                cr.label == label and cr.present
                for cr in category_risks
            )
        ]

        # ── Step 7: Top risks ─────────────────────────────────
        sorted_risks = sorted(
            category_risks,
            key=lambda cr: cr.risk_score * RISK_WEIGHTS.get(cr.label, 0),
            reverse=True,
        )
        top_risks = [cr.label for cr in sorted_risks[:3]]

        total_ms = inference_ms + (time.time() - start) * 1000

        logger.info(
            f"Risk scored: {filename} | "
            f"score={overall_score} | "
            f"level={risk_level} | "
            f"missing={missing_clauses}"
        )

        return DocumentRisk(
            filename        = filename,
            overall_score   = overall_score,
            risk_level      = risk_level,
            category_risks  = category_risks,
            missing_clauses = missing_clauses,
            entity_summary  = dict(entity_summary),
            top_risks       = top_risks,
            inference_ms    = round(total_ms, 1),
        )

    def score_summary(self, doc_risk: DocumentRisk) -> dict:
        """
        Returns a JSON-serializable summary for the API response.
        This is what the React frontend receives.
        """
        return {
            "filename":      doc_risk.filename,
            "overall_score": doc_risk.overall_score,
            "risk_level":    doc_risk.risk_level,
            "top_risks":     doc_risk.top_risks,
            "missing_clauses": doc_risk.missing_clauses,
            "categories": [
                {
                    "label":          cr.label,
                    "present":        cr.present,
                    "confidence":     cr.max_confidence,
                    "chunk_count":    cr.chunk_count,
                    "risk_score":     cr.risk_score,
                    "top_clause":     cr.top_clause,
                }
                for cr in doc_risk.category_risks
            ],
            "entities":      doc_risk.entity_summary,
            "inference_ms":  doc_risk.inference_ms,
        }


# Singleton
_scorer_instance = None


def get_risk_scorer() -> RiskScorer:
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = RiskScorer()
    return _scorer_instance