# ml/training/download_cuad.py — FIXED VERSION
# Only change: dataset name on line 31

import json
from pathlib import Path
from datasets import load_dataset
from loguru import logger


def download_cuad(save_dir: str = "./ml/data/cuad") -> None:

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CUAD dataset from HuggingFace Hub...")
    logger.info("This is a ~50MB download — happens once only.")

    # FIXED: correct dataset name is 'cuad' not 'theatticusproject/cuad'
    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)

    logger.info(f"Dataset splits available: {list(dataset.keys())}")

    for split_name, split_data in dataset.items():
        output_file = save_path / f"{split_name}.json"
        logger.info(f"Saving split '{split_name}' — {len(split_data)} examples...")

        records = []
        for example in split_data:
            records.append({
                "id":       example.get("id", ""),
                "title":    example.get("title", ""),
                "context":  example.get("context", ""),
                "question": example.get("question", ""),
                "answers":  example.get("answers", {}),
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(records)} records to {output_file}")

    metadata = {
        "source":      "cuad",
        "num_splits":  len(dataset),
        "splits": {
            name: len(data) for name, data in dataset.items()
        },
        "description": "Contract Understanding Atticus Dataset — 510 contracts, 41 clause categories",
    }

    with open(save_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Download complete. Files saved:")
    for f in sorted(save_path.iterdir()):
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {f.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    download_cuad()