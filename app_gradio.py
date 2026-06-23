"""Gradio UI for DocuMind — the demo-friendly interface for Hugging Face Spaces.

Upload one or more documents, then ask questions. Answers are grounded in the
documents and shown with their source passages.

Local:   python app_gradio.py   -> http://127.0.0.1:7860
On HF Spaces (Docker SDK): the Dockerfile runs this on port 7860.
"""
from __future__ import annotations
import os
# --- macOS stability: prevent OpenMP duplicate-runtime segfaults (faiss + torch) ---
# Must be set BEFORE torch / faiss are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr

from documind.rag import DocuMind

# module-level holder for the current index (rebuilt whenever docs are uploaded)
_state = {"dm": None}


def build_index(files):
    if not files:
        return "Please upload at least one .pdf / .txt / .md file."
    paths = [f.name for f in files]
    _state["dm"] = DocuMind.from_documents(paths, persist=False)
    names = ", ".join(os.path.basename(p) for p in paths)
    return f"Indexed: {names}. You can ask questions now."


def answer(question):
    if _state["dm"] is None:
        return "Upload and index documents first.", ""
    if not question.strip():
        return "Type a question.", ""
    result = _state["dm"].ask(question)
    sources = "\n\n".join(
        f"[{i+1}] {s['source']} (chunk {s['chunk_id']}, score {s['score']})\n{s['text']}"
        for i, s in enumerate(result["sources"])
    )
    return result["answer"], sources


with gr.Blocks(title="DocuMind — RAG Document Assistant") as demo:
    gr.Markdown(
        "# DocuMind — Generative AI Document Assistant (RAG)\n"
        "Upload documents, then ask questions answered **only** from their content, "
        "with citations. Runs on local open-source models (FAISS + flan-t5)."
    )
    with gr.Row():
        with gr.Column():
            files = gr.File(file_count="multiple",
                            file_types=[".pdf", ".txt", ".md"],
                            label="1. Upload documents")
            build_btn = gr.Button("2. Build index", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column():
            question = gr.Textbox(label="3. Ask a question",
                                  placeholder="e.g. What was the revenue in FY25?")
            ask_btn = gr.Button("Ask", variant="primary")
            ans = gr.Textbox(label="Answer", interactive=False)
            src = gr.Textbox(label="Sources (retrieved passages)",
                             interactive=False, lines=8)

    build_btn.click(build_index, inputs=files, outputs=status)
    ask_btn.click(answer, inputs=question, outputs=[ans, src])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)))
