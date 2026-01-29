FROM python:3.11-slim

# -----------------------
# ✅ Environment variables
# -----------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TESSERACT_CMD=/usr/bin/tesseract \
    CHROMA_DIR=/data/chroma \
    CHROMA_ROOT=/data/chroma \
    RAG_PDF_DIR=/app/pdfs \
    PYTHONPATH=/app \
    ENV=prod

# -----------------------
# 🧩 System dependencies
# -----------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg2 \
        apt-transport-https \
        unixodbc \
        unixodbc-dev \
        ffmpeg \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
    ; \
    mkdir -p /etc/apt/keyrings; \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg; \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17; \
    mkdir -p /data/chroma; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -----------------------
# 🧩 Python dependencies
# -----------------------
RUN python -m pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Add compatibility fix for embeddings
RUN pip install --no-cache-dir sentence-transformers==2.2.2 huggingface-hub==0.24.5

# -----------------------
# 📦 Copy application code
# -----------------------
COPY . /app

# -----------------------
# ✅ Auto-ingest script
# -----------------------
RUN echo '#!/usr/bin/env bash\n\
set -euo pipefail\n\
echo "== Container start ==" \n\
echo "ENV=${ENV:-dev}"\n\
echo "CHROMA_ROOT=${CHROMA_ROOT:-/data/chroma}"\n\
mkdir -p "${CHROMA_ROOT}"\n\
_need_ingest=0\n\
for level in low mid high; do\n\
  lvl_dir="${CHROMA_ROOT}/${level}"\n\
  if [ ! -d "$lvl_dir" ] || [ -z "$(ls -A "$lvl_dir" 2>/dev/null || true)" ]; then\n\
    _need_ingest=1\n\
  fi\n\
done\n\
if [ "${_need_ingest}" -eq 1 ]; then\n\
  echo "No Chroma data found → running ingestion..."\n\
  python -m ragg.ingest_all || echo "WARNING: ingestion returned non-zero exit"\n\
else\n\
  echo "Chroma already present → skipping ingestion."\n\
fi\n\
exec gunicorn --workers 2 --threads 4 --timeout 120 -b 0.0.0.0:7860 verification:app' > /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 7860

# -----------------------
# ✅ Final command
# -----------------------
CMD ["/app/start.sh"]
