# Demo image for a 512 MB free host. No PyTorch.
# Passage vectors are already in data/acme/index. This image only embeds the question.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    AUTORAG_RUNTIME=lite \
    AUTORAG_GENERATOR=extractive \
    OMP_NUM_THREADS=1 \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY autorag ./autorag
RUN pip install --no-cache-dir ".[lite]"

COPY data ./data
COPY scripts ./scripts

RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 7860

# Render sets PORT. Hugging Face Spaces expects 7860, which is the default here.
CMD ["sh", "-c", "autorag serve --host 0.0.0.0 --port ${PORT:-7860} --source data/acme"]
