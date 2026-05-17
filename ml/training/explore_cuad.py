# ml/training/explore_cuad.py
# ============================================================
# Run this ONCE to understand what the data looks like.
# Nothing is saved — this is pure exploration.
#
# Usage:
#   python ml/training/explore_cuad.py
# ============================================================

import json
from pathlib import Path
from collections import Counter, defaultdict
from loguru import logger


def explore(data_path: str = "./ml/data/cuad/train.json") -> None:

    logger.info(f"Loading {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info(f"Total records: {len(records)}")

    # ── 1. Print one raw record so we can see exact field structure ──
    logger.info("\n=== SAMPLE RECORD (index 0) ===")
    sample = records[0]
    for key, value in sample.items():
        if key == "context":
            # Contract text is huge — just show the first 300 chars
            logger.info(f"  {key}: {str(value)[:300]}...")
        else:
            logger.info(f"  {key}: {value}")

    # ── 2. Count unique contracts ──
    titles = set(r["title"] for r in records)
    logger.info(f"\n=== UNIQUE CONTRACTS: {len(titles)} ===")
    for t in sorted(titles)[:5]:
        logger.info(f"  {t}")
    logger.info("  ... (showing first 5)")

    # ── 3. Count unique clause category questions ──
    questions = Counter(r["question"] for r in records)
    logger.info(f"\n=== UNIQUE QUESTIONS (clause categories): {len(questions)} ===")
    for q, count in sorted(questions.items()):
        logger.info(f"  [{count:5d} records]  {q}")

    # ── 4. Measure how many records have actual answer spans vs empty ──
    has_answer = 0
    no_answer  = 0
    for r in records:
        answers = r.get("answers", {})
        # answers is a dict with key "text" holding a list of strings
        texts = answers.get("text", [])
        if texts and any(t.strip() for t in texts):
            has_answer += 1
        else:
            no_answer += 1

    logger.info(f"\n=== ANSWER COVERAGE ===")
    logger.info(f"  Records WITH an answer span : {has_answer}")
    logger.info(f"  Records WITHOUT an answer   : {no_answer}")
    logger.info(f"  Coverage: {has_answer / len(records) * 100:.1f}%")

    # ── 5. Show answer span examples for 3 different clause types ──
    logger.info(f"\n=== SAMPLE ANSWER SPANS ===")
    seen = set()
    for r in records:
        q = r["question"]
        texts = r.get("answers", {}).get("text", [])
        if texts and q not in seen:
            seen.add(q)
            logger.info(f"\n  Question : {q}")
            logger.info(f"  Answer   : {texts[0][:200]}")
        if len(seen) >= 3:
            break

    # ── 6. Measure context (contract text) lengths ──
    lengths = [len(r["context"].split()) for r in records]
    logger.info(f"\n=== CONTEXT LENGTH (words) ===")
    logger.info(f"  Min    : {min(lengths)}")
    logger.info(f"  Max    : {max(lengths)}")
    logger.info(f"  Average: {sum(lengths) // len(lengths)}")


if __name__ == "__main__":
    explore()