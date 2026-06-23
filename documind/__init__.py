"""DocuMind — a local, free Retrieval-Augmented Generation (RAG) document assistant.

Pipeline:  documents -> chunk -> embed -> FAISS index -> retrieve -> prompt -> LLM answer
All components run locally with open-source models. No paid API keys needed.
"""
__version__ = "1.0.0"
