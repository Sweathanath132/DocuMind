 DocuMind — Local Generative AI Document Assistant (RAG)

Ask natural-language questions over your own documents (PDF / TXT / MD) and get
answers grounded in the source text, **with citations**. Everything runs locally
on open-source models — **no paid API keys**.

`documents → chunk → embed → FAISS index → retrieve top-k → grounded prompt → local LLM → answer + sources`

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | 384-dim, fast on CPU, strong retrieval baseline |
| Vector store | **FAISS** `IndexFlatIP` | exact cosine search; industry standard |
| LLM | `google/flan-t5-base` (an SLM, ~250M params) | instruction-tuned, CPU-friendly, low-latency local inference |
| API / UI | **FastAPI** + **Gradio** | REST endpoints and a clickable demo UI |
| Docs | pypdf | PDF text extraction |

## Quickstart

```bash
# 1. (optional) create a virtual env
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3a. Gradio demo UI (recommended)
python app_gradio.py               # open http://127.0.0.1:7860

# 3b. CLI
python cli.py ingest sample_docs/*.txt
python cli.py ask "How much revenue did Sagility report in FY25?"
python cli.py chat                 # interactive

# 3c. REST API
uvicorn app:app --reload           # open http://127.0.0.1:8000/docs
```

> First run downloads the two models from Hugging Face (~80 MB embeddings +
> ~1 GB LLM) and caches them. Subsequent runs are offline.

## How it works 

1. **Chunking** (`documind/ingest.py`) — documents are split into ~600-character
   overlapping windows that break on sentence boundaries. Overlap (100 chars)
   prevents losing context that straddles a boundary.
2. **Embedding** (`documind/embeddings.py`) — each chunk → a normalised 384-dim
   vector. Normalising means inner product = cosine similarity.
3. **Indexing & retrieval** (`documind/vectorstore.py`) — FAISS stores the vectors;
   at query time we embed the question and pull the `top_k=4` nearest chunks.
4. **Grounded generation** (`documind/prompts.py`, `documind/llm.py`) — retrieved
   chunks are injected into a prompt that instructs the model to answer **only**
   from the context and say "I could not find this" otherwise. This is the core
   **anti-hallucination** mechanism.
5. **Citations** (`documind/rag.py`) — every answer is returned with the source
   passages and similarity scores it was grounded on.

## Project layout
```
DocuMind/
├── app.py                 # FastAPI server (/upload, /ask)
├── app_gradio.py          # Gradio demo UI (used for Hugging Face Spaces)
├── cli.py                 # command-line interface
├── Dockerfile             # container for HF Spaces / Render / Railway / Fly
├── requirements.txt
├── DEPLOY.md              # GitHub + hosting instructions
├── sample_docs/           # demo document
└── documind/
    ├── config.py          # all tunable settings
    ├── ingest.py          # load + chunk
    ├── embeddings.py      # MiniLM embedding wrapper
    ├── vectorstore.py     # FAISS build / save / load / search
    ├── prompts.py         # grounded RAG prompt template
    ├── llm.py             # local flan-t5 wrapper
    └── rag.py             # orchestrator: retrieve → prompt → generate
```

 Knobs to tune 
- `chunk_size` / `chunk_overlap` — retrieval granularity vs. context.
- `top_k` — recall vs. prompt-length/noise trade-off.
- `embedding_model` — swap to `bge-small-en` for higher accuracy.
- `llm_model` — `flan-t5-large` for quality, or wire an API client into `llm.py`.

## Possible next steps 
- Re-ranking retrieved chunks with a cross-encoder.
- Hybrid search (BM25 keyword + dense vectors).
- Per-source metadata filtering and access control.
- Streaming responses; evaluation with RAGAS (faithfulness, answer relevance).

## Deploying
See **`DEPLOY.md`** for step-by-step GitHub upload and a free live demo on
Hugging Face Spaces. (Note: Vercel is not suitable — it's for static/serverless
frontends, not a ~1 GB-model Python app.)
