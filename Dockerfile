# Reproducibility Dockerfile for the EAAI causal-leak study.
# Pinned to PyTorch 2.5.1 + CUDA 12.1 + Python 3.12.
#
# Build:
#   docker build -t causal-leak:eaai .
#
# Run a single end-to-end reproduction (writes results/ to ./mnt):
#   docker run --rm --gpus all -v "$(pwd)/mnt":/work/causal-leak/results \
#     causal-leak:eaai bash src/run_multiseed.sh
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        git wget curl ca-certificates build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt /work/causal-leak/requirements.txt
COPY requirements-eaai.txt /work/causal-leak/requirements-eaai.txt

RUN python3.12 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r /work/causal-leak/requirements.txt && \
    /opt/venv/bin/pip install -r /work/causal-leak/requirements-eaai.txt

ENV PATH=/opt/venv/bin:$PATH

WORKDIR /work/causal-leak
COPY . /work/causal-leak

ENV PYBIN=/opt/venv/bin/python
CMD ["bash"]
