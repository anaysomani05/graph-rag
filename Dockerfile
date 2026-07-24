FROM python:3.11-slim

WORKDIR /app

# torch/sentence-transformers need build tools for a couple of transitive deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

# Pre-download the two sentence-transformer models at build time, not on first
# request — avoids a slow, memory-spiky cold start on the Space's first real query.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Hugging Face Spaces (Docker SDK) expects the app to listen on 7860.
EXPOSE 7860

CMD ["uvicorn", "graphrag.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
