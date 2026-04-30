#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/songze/asim"
LOG_DIR="${ROOT_DIR}/result/qwen_server"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/home/AD/user/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B-local}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-local-qwen}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"

exec "${ROOT_DIR}/.venv_qwen/bin/vllm" serve "${MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}"
