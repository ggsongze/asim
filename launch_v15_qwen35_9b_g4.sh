#!/usr/bin/env bash
# v15: Qwen3.5-9B + XML tool format + G=4 + 2 GPU
# Uses .venv_qwen35 (which has unsloth installed but we DISABLE it for now —
# 9B may be too large for Unsloth with our linear-attn target_modules in
# 2-rank DDP. Plain transformers + peft path is verified.
#
# Hint settings match Qwen3-8B (v14-r3) semantically:
#   - Same PMV_EXPLANATION (shared in llm_setpoint_planner.py)
#   - Same |PMV|>=0.4 buffer warning in tool_response (auto-aligned via
#     Qwen35TransformersSamplingBackend._handle_pmv_tool_call)
#   - Same loop controls: cap=30, mention-trigger, 2-strike force-finalize,
#     dup penalty in reward (via _compute_format_quality_penalty)
#   - Tool format is XML (forced by Qwen3.5 chat template); JSON not possible
set -e

DATE=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="result/gspo/qwen35_9b_v15_g4_${DATE}"
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
