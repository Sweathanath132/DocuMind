# DocuMind container — works on Hugging Face Spaces (Docker SDK), Render, Railway, Fly.io.
FROM python:3.11-slim

# system deps occasionally needed by faiss/torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# cache the model downloads inside a writable dir (HF Spaces needs this)
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860
CMD ["python", "app_gradio.py"]
