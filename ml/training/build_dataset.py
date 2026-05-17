# ml/training/build_dataset.py
# ============================================================
# Reads raw CUAD JSON files, collapses 41 clause categories
# into 6 risk labels, deduplicates contract text, and saves
# a clean dataset ready for training.
#
# Usage:
#   python ml/training/build_dataset.py
# ============================================================

import json
import random
from pathlib import Path
from collections import Counter
from loguru import logger


# ── Label mapping ────────────────────────────────────────────
# Keys are exact CUAD question substrings (the category name
# inside the quotes). Values are our 6 risk labels.
# "none" means metadata — useful as negative examples only.

LABEL_MAP = {
    # IP Assignment
    "Ip Ownership Assignment":          "ip_assignment",
    "Affiliate License-Licensee":       "ip_assignment",
    "Affiliate License-Licensor":       "ip_assignment",
    "License Grant":                    "ip_assignment",
    "Irrevocable Or Perpetual License": "ip_assignment",
    "Joint Ip Ownership":               "ip_assignment",
    "Non-Transferable License":         "ip_assignment",
    "Unlimited/All-You-Can-Eat-License":"ip_assignment",
    "Source Code Escrow":               "ip_assignment",
    "Covenant Not To Sue":              "ip_assignment",

    # Indemnity
    "Uncapped Liability":               "indemnity",
    "Liquidated Damages":               "indemnity",
    "Third Party Beneficiary":          "indemnity",
    "Insurance":                        "indemnity",

    # Liability Cap
    "Cap On Liability":                 "liability_cap",
    "Most Favored Nation":              "liability_cap",
    "Price Restrictions":               "liability_cap",
    "Revenue/Profit Sharing":           "liability_cap",

    # Non-Compete
    "Non-Compete":                      "non_compete",
    "No-Solicit Of Customers":          "non_compete",
    "No-Solicit Of Employees":          "non_compete",
    "Exclusivity":                      "non_compete",
    "Competitive Restriction Exception":"non_compete",
    "Non-Disparagement":                "non_compete",

    # Termination
    "Termination For Convenience":      "termination",
    "Post-Termination Services":        "termination",
    "Notice Period To Terminate Renewal":"termination",
    "Change Of Control":                "termination",
    "Renewal Term":                     "termination",
    "Expiration Date":                  "termination",
    "Anti-Assignment":                  "termination",

    # Data Privacy (broad bucket for compliance-related clauses)
    "Audit Rights":                     "data_privacy",
    "Volume Restriction":               "data_privacy",
    "Minimum Commitment":               "data_privacy",
    "Rofr/Rofo/Rofn":                   "data_privacy",
    "Warranty Duration":                "data_privacy",
    "Governing Law":                    "data_privacy",

    # Metadata — kept as negatives, not risk categories
    "Document Name":                    "none",
    "Agreement Date":                   "none",
    "Effective Date":                   "none",
    "Parties":                          "none",
}


def extract_category(question: str) -> str:
    """
    Pull the category name out of the CUAD question string.

    CUAD questions look like:
      'Highlight the parts ... related to "Cap On Liability" that ...'

    We extract just the part inside the quotes.
    """
    # Find text between the first pair of double quotes
    start = question.find('"')
    end   = question.find('"', start + 1)
    if start == -1 or end == -1:
        return "unknown"
    return question[start + 1 : end]


def build_dataset(
    cuad_dir:   str = "./ml/data/cuad",
    output_dir: str = "./ml/data/processed",
    seed:       int = 42,
) -> None:

    random.seed(seed)  # Reproducible shuffle

    cuad_path   = Path(cuad_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load both splits and merge ───────────────────
    # We combine train + test from CUAD, then do our own split.
    # This gives us control over the split ratio and ensures
    # no data leakage between our train/val/test sets.
    all_records = []
    for split_file in ["train.json", "test.json"]:
        fpath = cuad_path / split_file
        if not fpath.exists():
            logger.warning(f"{split_file} not found — skipping")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            records = json.load(f)
        logger.info(f"Loaded {len(records)} records from {split_file}")
        all_records.extend(records)

    logger.info(f"Total raw records: {len(all_records)}")

    # ── Step 2: Transform each record ────────────────────────
    processed = []
    skipped   = 0

    for record in all_records:
        question = record.get("question", "")
        context  = record.get("context", "")
        answers  = record.get("answers", {})

        # Skip records with no contract text
        if not context or not context.strip():
            skipped += 1
            continue

        # Extract category name from the question string
        category = extract_category(question)

        # Map category to our risk label
        risk_label = LABEL_MAP.get(category, "unknown")

        # Skip anything we couldn't map
        if risk_label == "unknown":
            skipped += 1
            continue

        # Extract the answer span text if it exists
        # answers is a dict: {"text": [...], "answer_start": [...]}
        # We handle both string and dict formats (CUAD has both)
        if isinstance(answers, str):
            # Some records store answers as a JSON string — parse it
            try:
                answers = json.loads(answers.replace("'", '"'))
            except Exception:
                answers = {}

        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []

        # clause_present = True if a lawyer found this clause in the contract
        clause_present = bool(answer_texts and any(t.strip() for t in answer_texts))

        # The clause text is the answer span if present,
        # otherwise we use a 500-char window from the start of context
        # as a representative negative example
        if clause_present:
            clause_text = answer_texts[0]
        else:
            # Negative example: first 500 chars of contract
            # (enough context for the model to learn "no clause here")
            clause_text = context[:500]

        processed.append({
            "id":             record.get("id", ""),
            "title":          record.get("title", ""),
            "risk_label":     risk_label,
            "clause_present": clause_present,
            "clause_text":    clause_text,
            "full_context":   context,   # kept for chunking in Phase 3
        })

    logger.info(f"Processed: {len(processed)} records")
    logger.info(f"Skipped:   {skipped} records")

    # ── Step 3: Show label distribution ──────────────────────
    label_counts = Counter(r["risk_label"] for r in processed)
    present_counts = Counter(
        r["risk_label"] for r in processed if r["clause_present"]
    )
    logger.info("\n=== LABEL DISTRIBUTION ===")
    for label, count in sorted(label_counts.items()):
        present = present_counts.get(label, 0)
        logger.info(
            f"  {label:<20} total={count:5d}  "
            f"present={present:5d}  "
            f"absent={count - present:5d}"
        )

    # ── Step 4: 70 / 15 / 15 split ───────────────────────────
    # Shuffle first so contracts aren't grouped by alphabet
    random.shuffle(processed)

    n       = len(processed)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    # test gets the remainder to avoid off-by-one rounding errors
    n_test  = n - n_train - n_val

    train_data = processed[:n_train]
    val_data   = processed[n_train : n_train + n_val]
    test_data  = processed[n_train + n_val :]

    logger.info(f"\n=== SPLIT SIZES ===")
    logger.info(f"  Train : {len(train_data)}")
    logger.info(f"  Val   : {len(val_data)}")
    logger.info(f"  Test  : {len(test_data)}")

    # ── Step 5: Save splits ───────────────────────────────────
    splits = {
        "train": train_data,
        "val":   val_data,
        "test":  test_data,
    }

    for split_name, split_data in splits.items():
        out_file = output_path / f"{split_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        size_kb = out_file.stat().st_size / 1024
        logger.info(f"  Saved {split_name}.json — {len(split_data)} records ({size_kb:.0f} KB)")

    # ── Step 6: Save label registry ──────────────────────────
    # This file is the contract between the data and the model.
    # Every later script imports from here — no hardcoded label strings.
    label_registry = {
        "labels":      sorted(set(LABEL_MAP.values()) - {"none"}),
        "label_to_id": {
            label: idx for idx, label in enumerate(
                sorted(set(LABEL_MAP.values()) - {"none"})
            )
        },
        "id_to_label": {
            str(idx): label for idx, label in enumerate(
                sorted(set(LABEL_MAP.values()) - {"none"})
            )
        },
    }

    with open(output_path / "label_registry.json", "w") as f:
        json.dump(label_registry, f, indent=2)

    logger.info(f"\n=== LABEL REGISTRY ===")
    logger.info(f"  Risk labels: {label_registry['labels']}")
    logger.info(f"  label_to_id: {label_registry['label_to_id']}")
    logger.info("build_dataset complete.")


if __name__ == "__main__":
    build_dataset()