# Reproducibility image for the DMKD article.
# Python 3.12.3 with the exact package versions of requirements-lock.txt
# (PyTorch 2.5.1+cu121 wheels bundle their CUDA libraries, so the image runs on
# CPU, and on GPU with the NVIDIA container toolkit and a CUDA 12 driver).
#
# Build:
#   docker build -t causal-leak:dmkd .
#
# Gate and cited numbers (no data needed):
#   docker run --rm causal-leak:dmkd make verify numbers PAPER_DIR=__none__
#
# Full reproduction (mount the AI Hub data and a results directory):
#   docker run --rm --gpus all \
#     -v /path/to/kcc-water/Dataset:/data -e CAUSAL_LEAK_DATA_DIR=/data \
#     -v "$(pwd)/results":/work/causal-leak/results \
#     causal-leak:dmkd make reproduce
FROM python:3.12.3-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        make git curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/causal-leak
COPY requirements-lock.txt /work/causal-leak/requirements-lock.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-lock.txt

COPY . /work/causal-leak

ENV PYTHON=python \
    PYBIN=python
CMD ["bash"]
