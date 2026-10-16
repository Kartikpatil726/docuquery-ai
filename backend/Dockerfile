FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    HF_HOME=/opt/hf-cache

ARG WITH_LOCAL_EMBEDDINGS=true
WORKDIR /app
COPY requirements.txt requirements-local.txt ./
RUN pip install -r requirements.txt
RUN if [ "$WITH_LOCAL_EMBEDDINGS" = "true" ]; then \
    pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-local.txt \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"; \
    fi

COPY app ./app
EXPOSE 8000
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
