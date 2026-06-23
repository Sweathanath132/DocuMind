"""The RAG orchestrator — ties everything together.

ask() = retrieve top-k chunks -> build grounded prompt -> generate answer ->
return answer WITH the source passages it was grounded on (citations).
"""
from __future__ import annotations
from typing import List, Dict
from .vectorstore import VectorStore
from . import prompts, llm
from .ingest import build_chunks


class DocuMind:
    def __init__(self, store: VectorStore):
        self.store = store

    # ---- construction helpers ----
    @classmethod
    def from_documents(cls, paths: List[str], persist: bool = True) -> "DocuMind":
        records = build_chunks(paths)
        store = VectorStore().build(records)
        if persist:
            store.save()
        return cls(store)

    @classmethod
    def from_disk(cls) -> "DocuMind":
        return cls(VectorStore.load())

    # ---- query ----
    def ask(self, question: str) -> Dict:
        retrieved = self.store.search(question)
        prompt = prompts.build_prompt(question, retrieved)
        answer = llm.generate(prompt)
        return {
            "answer": answer,
            "sources": [
                {"source": rec["source"], "chunk_id": rec["chunk_id"],
                 "score": round(score, 3), "text": rec["text"][:200] + "..."}
                for rec, score in retrieved
            ],
        }
