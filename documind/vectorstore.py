"""Vector store: build, persist, load, and search — pure NumPy.

We store normalized embeddings in a single matrix and rank chunks by cosine
similarity (a dot product, since the vectors are L2-normalized). This is exact
nearest-neighbour search and is plenty fast for document-sized corpora.

Note: an earlier version used FAISS (IndexFlatIP). FAISS and PyTorch each ship
their own native OpenMP/BLAS runtime, and loading both in one process crashes on
some macOS / Python 3.13 setups. NumPy avoids that conflict entirely. For
very large corpora (millions of vectors) you'd reintroduce FAISS (IVF/HNSW) or a
managed vector DB in a separate service — a good thing to mention as 'scaling'.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from .config import settings
from . import embeddings


class VectorStore:
    def __init__(self):
        self.matrix: np.ndarray | None = None   # (n_chunks, dim), L2-normalized
        self.records: List[Dict] = []           # parallel chunk metadata

    # ---------- build ----------
    def build(self, records: List[Dict]) -> "VectorStore":
        self.records = records
        # embeddings.embed() already returns L2-normalized float32 vectors
        self.matrix = embeddings.embed([r["text"] for r in records])
        return self

    # ---------- search ----------
    def search(self, query: str, top_k: int | None = None) -> List[Tuple[Dict, float]]:
        top_k = top_k or settings.top_k
        qv = embeddings.embed([query])[0]          # (dim,)
        sims = self.matrix @ qv                     # cosine similarity (normalized)
        k = min(top_k, len(self.records))
        # top-k indices, highest similarity first
        top_idx = np.argsort(-sims)[:k]
        return [(self.records[int(i)], float(sims[int(i)])) for i in top_idx]

    # ---------- persistence ----------
    def save(self, directory: str | Path | None = None):
        directory = Path(directory or settings.index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "matrix.npy", self.matrix)
        with open(directory / "records.pkl", "wb") as f:
            pickle.dump(self.records, f)
        (directory / "meta.json").write_text(
            json.dumps({"count": len(self.records),
                        "embedding_model": settings.embedding_model}, indent=2))

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "VectorStore":
        directory = Path(directory or settings.index_dir)
        store = cls()
        store.matrix = np.load(directory / "matrix.npy")
        with open(directory / "records.pkl", "rb") as f:
            store.records = pickle.load(f)
        return store

    @staticmethod
    def exists(directory: str | Path | None = None) -> bool:
        directory = Path(directory or settings.index_dir)
        return (directory / "matrix.npy").exists()
