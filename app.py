"""FastAPI server for DocuMind.

Run:  uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs for an interactive Swagger UI.

Endpoints:
    POST /upload   -> upload one or more documents, (re)build the index
    POST /ask      -> ask a question, get an answer + citations
    GET  /health   -> liveness check
"""
import os
# macOS stability: prevent OpenMP duplicate-runtime segfaults (faiss + torch)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from documind.rag import DocuMind
from documind.vectorstore import VectorStore

app = FastAPI(title="DocuMind", description="Local RAG document assistant", version="1.0.0")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": VectorStore.exists()}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    tmp_dir = Path(tempfile.mkdtemp())
    saved = []
    for f in files:
        dest = tmp_dir / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(str(dest))
    DocuMind.from_documents(saved)   # builds + persists index
    return {"indexed_files": [Path(s).name for s in saved], "status": "index_built"}


@app.post("/ask")
def ask(req: AskRequest):
    if not VectorStore.exists():
        return {"error": "No index yet. Upload documents via /upload first."}
    dm = DocuMind.from_disk()
    return dm.ask(req.question)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>DocuMind — local RAG assistant</h2>
    <p>Use <a href="/docs">/docs</a> to upload documents and ask questions.</p>
    """
