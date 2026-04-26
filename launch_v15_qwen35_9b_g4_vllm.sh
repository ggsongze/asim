#!/usr/bin/env bash
# v15 with vLLM rollout backend.
# Architecture:
#   - Single rank (no DDP)
#   - HF model on cuda:0 (training: backward, optimizer)
#   - vLLM TP=2 across both GPUs (rollout, sleep mode for VRAM swap)
#   - LoRA hot-swap each step: HF saves adapter → vLLM LoRARequest reloads
#
# vLLM gpu_memory_utilization=0.45 leaves ~25GB on GPU 0 for HF (HF needs
# 18GB model + ~10GB activations during fwd/bwd).
set -e

DATE=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="result/gspo/qwen35_9b_v15_vllm_${DATE}"
mkdir -p "$OUTPUT_DIR"
echo "Output: $OUTPUT_DIR"

tmux new-session -d -s grpo_s2 "
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  HF_HOME=/home/AD/user/lab/asim/.model_cache \
  RL_W_ENERGY=3.0 \
  RL_FORECAST_CSV=miami_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6.csv \
  ASIM_ENABLE_THINKING=1 ASIM_ENABLE_PMV_TOOL=1 ASIM_MAX_TOOL_CALLS=30 \
  ASIM_TOOL_FORMAT=xml \
  ASIM_THINKING_GUARD=0 ASIM_DEBUG_THINKING=1 ASIM_DEBUG_KNOTS=1 \
  ASIM_THINKING_JSONL=${OUTPUT_DIR}/thinking_trace.jsonl \
  .venv_qwen35/bin/python train_qwen3_houston_gspo_stage2_steplevel_vllm.py \
      --model-name-or-path Qwen/Qwen3.5-9B \
      --resume-from result/gspo/stage1_checkpoints/miami_qwen3_8b_klguard_gpu0_checkpoint-16_for_stage2_20260414 \
      --kl-reference-from result/gspo/stage1_checkpoints/miami_qwen3_8b_klguard_gpu0_checkpoint-16_for_stage2_20260414 \
      --fresh-lora --kl-beta 0.0 \
      --output-dir ${OUTPUT_DIR} \
      --building-idf miami_stage2_10min.idf \
      --n-rollouts 4 --max-steps 16 \
      --lora-r 64 --lora-alpha 128 \
      --dataset-path result/gspo/miami_gspo_dataset_stage2_10min.jsonl \
      --gamma 0.95 \
      --max-output-tokens 8192 \
      --mode-setpoint-penalty-weight 0.0 --mode-setpoint-local-adv-weight 0.0 \
      --setpoint-only \
      --mode-exploration-steps 0 \
      --setpoint-exploration-prob 0.0 \
      --setpoint-exploration-late-prob 0.0 \
      --setpoint-exploration-max-blocks 0 \
      --device cuda:0 \
      --save-steps 4 --no-wandb \
      > ${OUTPUT_DIR}.log 2>&1
"

sleep 5
tmux ls
ps aux | grep -E "python.*train_qwen3" | grep -v grep | wc -l
echo "Output dir: $OUTPUT_DIR"
echo "Log: ${OUTPUT_DIR}.log"
