# backend/config.py

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """
    Pydantic reads matching variables from your .env file automatically.
    If a required variable is missing, the app crashes at startup with a
    clear error — much better than a silent None that breaks later.
    """

    # Database
    database_url: str = "postgresql://claritylens:claritylens@localhost:5432/claritylens"

    # Where uploaded PDFs land on disk
    storage_uploads_dir: Path = Path("./storage/uploads")
    storage_extracted_dir: Path = Path("./storage/extracted")

    # Where trained model weights are saved
    ml_embedder_path: Path = Path("./ml/models/embedder")
    ml_classifier_path: Path = Path("./ml/models/classifier")
    ml_ner_path: Path = Path("./ml/models/ner")
    ml_qa_path: Path = Path("./ml/models/qa")

    # CPU NOTE: 256 tokens is the single most important CPU optimisation.
    # DistilBERT attention scales as O(n²) — cutting 512→256 tokens
    # reduces attention computation by 4x.
    ml_max_seq_len: int = 256
    ml_batch_size: int = 8       # chunks processed together in one forward pass
    ml_chunk_size: int = 200     # words per clause chunk
    ml_chunk_overlap: int = 50   # overlap between adjacent chunks

    # CUAD dataset paths
    cuad_local_path: Path = Path("./ml/data/cuad")
    cuad_processed_path: Path = Path("./ml/data/processed")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    class Config:
        env_file = ".env"             # reads from .env in project root
        env_file_encoding = "utf-8"


# Create one instance here. Everyone imports this object — never
# instantiate Settings() anywhere else.
settings = Settings()