---
title: DocuMind
emoji: 📄
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DocuMind — Generative AI Document Assistant (RAG)

Upload documents (PDF / TXT / MD) and ask natural-language questions answered
**only** from their content, with citations. Runs fully on local open-source
models — FAISS vector search + a flan-t5 small language model. No paid API keys.

Pipeline: `documents → chunk → embed (MiniLM) → FAISS → retrieve top-k → grounded prompt → LLM → answer + sources`

First query downloads the models once (~1 GB) and caches them, then it's fast.
