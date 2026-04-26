#!/usr/bin/env bash
# v15 with KV-cache-reuse backend (CachedQwen35TransformersSamplingBackend).
# Same as launch_v15_qwen35_9b_g4.sh but adds ASIM_USE_CACHED_BACKEND=1.
# The cached backend eliminates the per-cycle full re-prefill of growing
# context (parent's O(N^2) tokenize + encode), forwarding only new tokens
# via past_key_values. Smoke-test speedup: ~1.14x on a single knot. For RL
# rollout (sampling, not greedy), bf16 numerical noise between cached and
# uncached paths is dwarfed by the sampling temperature, so output quality
# is equivalent.
set -e

DATE=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="result/gspo/qwen35_9b_v15_g4_cached_${DATE}"
mkdir -p "$OUTPUT_DIR"
echo "Output: $OUTPUT_DIR"

tmux new-session -d -s grpo_s2 "
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  HF_HOME=/home/AD/user/lab/asim/.model_cache \
  RL_W_ENERGY=3.0 \
  RL_FORECAST_CSV=miami_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6.csv \
  ASIM_ENABLE_THINKING=1 ASIM_ENABLE_PMV_TOOL=1 ASIM_MAX_TOOL_CALLS=30 \
  ASIM_TOOL_FORMAT=xml \
  ASIM_USE_CACHED_BACKEND=1 \
  ASIM_THINKING_GUARD=0 ASIM_DEBUG_THINKING=1 ASIM_DEBUG_KNOTS=1 \
  ASIM_THINKING_JSONL=${OUTPUT_DIR}/thinking_trace.jsonl \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv_qwen35/bin/torchrun --nproc_per_node=2 --nnodes=1 \
    --master_addr=127.0.0.1 --master_port=29536 \
    train_qwen3_houston_gspo_stage2_steplevel_2gpu.py \
      --model-name-or-path Qwen/Qwen3.5-9B \
      --resume-from result/gspo/stage1_checkpoints/miami_qwen3_8b_klguard_gpu0_checkpoint-16_for_stage2_20260414 \
      --kl-reference-from result/gspo/stage1_checkpoints/miami_qwen3_8b_klguard_gpu0_checkpoint-16_for_stage2_20260414 \
      --fresh-lora --kl-beta 0.0 \
      --output-dir ${OUTPUT_DIR} \
      --building-idf miami_stage2_10min.idf \
      --n-rollouts 4 --max-steps 16 \
      --lora-r 64 --lora-alpha 128 \
      --dataset-path result/gspo/miami_gspo_dataset_stage2_10min.jsonl \
      --advantage-mode return_to_go --gamma 0.95 \
      --max-output-tokens 8192 \
      --format-penalty-weight 0.3 \
      --mode-setpoint-penalty-weight 0.0 --mode-setpoint-local-adv-weight 0.0 \
      --setpoint-only \
      --mode-exploration-steps 0 \
      --setpoint-exploration-prob 0.0 \
      --setpoint-exploration-late-prob 0.0 \
      --setpoint-exploration-max-blocks 0 \
      --save-steps 4 --no-wandb \
      > ${OUTPUT_DIR}.log 2>&1
"

sleep 5
tmux ls
ps aux | grep -E "python.*train_qwen3" | grep -v grep | wc -l
echo "Output dir: $OUTPUT_DIR"
echo "Log: ${OUTPUT_DIR}.log"
