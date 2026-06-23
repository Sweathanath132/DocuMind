"""Prompt templates.

The single most important anti-hallucination lever in RAG is the instruction:
"answer ONLY from the context; if it's not there, say you don't know." We also
number the context passages so the model can ground its answer and we can show
citations back to the user.
"""

RAG_PROMPT = """You are DocuMind, a careful assistant that answers questions \
using ONLY the context passages provided below. Follow these rules:
- If the answer is not contained in the context, reply exactly: \
"I could not find this in the provided documents."
- Do not use outside knowledge. Do not guess.
- Be concise and factual.

Context passages:
{context}

Question: {question}

Answer:"""


def format_context(retrieved) -> str:
    """retrieved: list of (record, score) tuples."""
    blocks = []
    for i, (rec, score) in enumerate(retrieved, 1):
        blocks.append(f"[{i}] (source: {rec['source']}) {rec['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, retrieved) -> str:
    return RAG_PROMPT.format(context=format_context(retrieved), question=question)
