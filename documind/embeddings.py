"""Embedding model wrapper.

Embeddings turn text into vectors so that semantically similar passages sit
close together in vector space. We use all-MiniLM-L6-v2: 384 dimensions, small,
fast on CPU, and a strong baseline for retrieval.
"""
from __future__ import annotations
from typing import List
import numpy as np
from .config import settings

_model = None  # lazy singleton so we load the model only once


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """Return L2-normalised float32 embeddings (so dot product == cosine sim)."""
    model = _get_model()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vecs = vecs.astype("float32")
    # normalise -> inner-product search behaves like cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return vecs / norms


def embedding_dim() -> int:
    return _get_model().get_sentence_embedding_dimension()
