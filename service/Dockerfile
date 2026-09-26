# Image for the served RAG pipeline (docs/AZURE-PLAN.md, Phase 5).
# Built in the cloud from the REPO ROOT (the service imports the benchmark's
# own modules from rag/, evaluation/, fine_tuning/):
#   az acr build -r <acr> -t capstone-rag:v1 -f service/Dockerfile .
# .dockerignore keeps secrets and bulk data out of the build context.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch first, from its own index (see service/requirements.txt).
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
COPY service/requirements.txt service/requirements.txt
RUN pip install -r service/requirements.txt

# Bake the embedding model into the image at BUILD time: a cold start must not
# depend on (or wait for) a Hugging Face download.
RUN python -c "from transformers import AutoModel, AutoTokenizer; m='sentence-transformers/all-MiniLM-L6-v2'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY . .

# Never run as root.
RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
