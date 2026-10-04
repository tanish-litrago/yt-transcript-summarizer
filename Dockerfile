FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch must come from the CUDA index before requirements.txt, or pip may pull the CPU wheel
RUN pip install torch torchaudio --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]