"""Command-line interface for DocuMind.

Usage:
    python cli.py ingest sample_docs/*.txt      # build the index
    python cli.py ask "What is the refund policy?"
    python cli.py chat                           # interactive loop
"""
import os
# macOS stability: prevent OpenMP duplicate-runtime segfaults (faiss + torch)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import glob
from documind.rag import DocuMind
from documind.vectorstore import VectorStore


def cmd_ingest(patterns):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    if not paths:
        print("No files matched.")
        return
    print(f"Ingesting {len(paths)} file(s): {paths}")
    DocuMind.from_documents(paths)
    print("Index built and saved to ./storage")


def cmd_ask(question):
    if not VectorStore.exists():
        print("No index found. Run `python cli.py ingest <files>` first.")
        return
    dm = DocuMind.from_disk()
    result = dm.ask(question)
    print("\nANSWER:\n" + result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s['source']} #chunk{s['chunk_id']} (score {s['score']})")


def cmd_chat():
    if not VectorStore.exists():
        print("No index found. Run ingest first.")
        return
    dm = DocuMind.from_disk()
    print("DocuMind chat — type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        result = dm.ask(q)
        print("DocuMind:", result["answer"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":
        cmd_ingest(sys.argv[2:])
    elif cmd == "ask":
        cmd_ask(" ".join(sys.argv[2:]))
    elif cmd == "chat":
        cmd_chat()
    else:
        print(__doc__)
