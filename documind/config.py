"""Central configuration. Everything you'd want to tune lives here."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # --- Models (all open-source, downloaded once from Hugging Face) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast on CPU
    # flan-t5-small (~300MB) is far lighter on disk/memory than -base (~1GB).
    # Bump to "google/flan-t5-base" for higher quality once you have disk space.
    llm_model: str = "google/flan-t5-small"  # instruction-tuned SLM, runs on CPU

    # --- Chunking ---
    chunk_size: int = 600        # characters per chunk
    chunk_overlap: int = 100     # overlap keeps context across chunk boundaries

    # --- Retrieval ---
    top_k: int = 4               # how many chunks to feed the LLM as context

    # --- Generation ---
    max_new_tokens: int = 256

    # --- Storage ---
    index_dir: Path = Path("storage")   # where the FAISS index + chunks are persisted


settings = Settings()
