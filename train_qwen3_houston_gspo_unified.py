#!/usr/bin/env python3
"""Unified GRPO trainer: LLM outputs mode+setpoint together.

Gradual transition from fixed-mode rollout to free-mode (LLM chooses mode).
Phase 1: 3 fixed samples (same as old best-of-3)
Phase 2: 3 fixed anchors + 1 free
Phase 3: 3 fixed anchors + 2 free
Phase 4: 3 fixed anchors + 3 free

Block-level: always update (no history best gating).
Day-level: history best gating preserved.
Advantage: flat sample-level GRPO + per-knot partial return.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/songze/asim")
SHARED_SITE_PACKAGES = PROJECT_ROOT / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if SHARED_SITE_PACKAGES.exists() and str(SHARED_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(SHARED_SITE_PACKAGES))

import numpy as np

from grpo_miami_bandit import RESULT_DIR, MiamiGRPOBandit, _plainify
from llm_setpoint_planner import (
    PlannerConstraints,
    PlannerRequest,
    TransformersSamplingBackend,
)
from llm_setpoint_planner_unified import UnifiedBlockPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block-based 3h GSPO trainer for Houston Qwen.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=RESULT_DIR / "houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "qwen3_houston_gspo_block")
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1229)
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--cache-steps", type=int, default=4,
                        help="Write lightweight cache-checkpoint-N every N steps; 0 disables cache checkpoints")
    parser.add_argument("--keep-cache-checkpoints", action="store_true",
                        help="Keep cache-checkpoint-N dirs after each episode instead of deleting them")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.1,
                        help="KL penalty coefficient for GRPO (0 to disable)")
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a PEFT adapter checkpoint to resume training from")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--wandb-project", type=str, default="asim-houston-grpo")
    parser.add_argument("--wandb-group", type=str, default="block-rolling-grpo")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="WandB run name. Defaults to output-dir basename.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--sequential-rollout", action="store_true",
                        help="Run mode rollouts sequentially instead of parallel (avoids EP resource contention for full-param training)")
    parser.add_argument("--no-block-reflection", action="store_true",
                        help="Disable block-level reflection injection (match v3 behavior)")
    parser.add_argument("--no-block-history-best", action="store_true",
                        help="Disable block-level history-best gating (always update)")
    parser.add_argument("--building-idf", type=str, default=None,
                        help="Path to IDF file (default: houston.idf)")
    parser.add_argument("--weather-epw", type=str, default=None,
                        help="Path to EPW file (default: houston EPW)")
    return parser.parse_args()


from llm_setpoint_planner import ALL_CANDIDATE_MODES
CANDIDATE_MODES = ALL_CANDIDATE_MODES  # 3 modes: cooling, balanced, energy_saving

# Phase transition: keep all 3 fixed mode anchors while adding free samples.
# (step_lo, step_hi, n_fixed, n_free, kl_beta, lr)
PHASE_BOUNDARIES = [
    (1, 48, 3, 0, 0.1, 2e-5),       # EP1-3: all fixed, learn mode-specific setpoint behavior.
    (49, 96, 3, 1, 0.15, 1.5e-5),   # EP4-6: add one free sample, tighten KL slightly.
    (97, 128, 3, 2, 0.2, 1e-5),     # EP7-8: add two free samples, stronger KL constraint.
    (129, 160, 3, 3, 0.3, 1e-5),    # EP9-10: free choices compete with all fixed anchors.
]

# KL guard: skip optimizer.step() if block KL exceeds this threshold.
# Prevents a single catastrophic update from destroying the adapter.
KL_GUARD_THRESHOLD = 5e3

def _get_phase(step: int) -> tuple[str, int, int, float, float]:
    """Return (phase_name, n_fixed, n_free, kl_beta, lr) for a given step."""
    for lo, hi, nf, nfr, kl_beta, lr in PHASE_BOUNDARIES:
        if lo <= step <= hi:
            return f"phase_{nf}f{nfr}r", nf, nfr, kl_beta, lr
    return "phase_3f3r", 3, 3, 0.3, 1e-5


class _FreeSamplePlannerProxy:
    """Wraps UnifiedBlockPlanner to intercept plan_knot for free sampling.

    For knot_index==0: calls plan_knot_free() (LLM chooses mode+setpoint).
    For knot_index>0: calls plan_knot() with the LLM-chosen mode.
    Each free sample creates its own proxy (no shared state).
    """

    def __init__(self, real_planner: UnifiedBlockPlanner):
        self._real = real_planner
        self.chosen_mode: str | None = None

    def plan_knot(self, *, block_index, knot_index, block_start, block_end,
                  mode="balanced", observation=None, wallclock=None):
        if knot_index == 0:
            result = self._real.plan_knot_free(
                block_index=block_index, knot_index=knot_index,
                block_start=block_start, block_end=block_end,
                observation=observation, wallclock=wallclock,
            )
            self.chosen_mode = result.get("mode", "balanced")
            return result
        else:
            return self._real.plan_knot(
                block_index=block_index, knot_index=knot_index,
                block_start=block_start, block_end=block_end,
                mode=self.chosen_mode or "balanced",
                observation=observation, wallclock=wallclock,
            )

    def __getattr__(self, name):
        return getattr(self._real, name)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_resume_training_state(checkpoint_dir: Path) -> dict[str, Any]:
    """Validate a resume checkpoint before loading the model.

    Cache checkpoints are intended as crash-recovery points, so missing optimizer
    or reflection state is treated as fatal instead of silently continuing from a
    partial state.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"resume checkpoint is not a directory: {checkpoint_dir}")

    is_cache_checkpoint = checkpoint_dir.name.startswith("cache-checkpoint-")
    required_files = ["adapter_config.json", "training_state.json"]
    if is_cache_checkpoint:
        required_files.extend(["optimizer.pt", "reflections.json"])

    missing = [name for name in required_files if not (checkpoint_dir / name).exists()]
    if not (
        (checkpoint_dir / "adapter_model.safetensors").exists()
        or (checkpoint_dir / "adapter_model.bin").exists()
    ):
        missing.append("adapter_model.safetensors|adapter_model.bin")
    if missing:
        checkpoint_type = "cache checkpoint" if is_cache_checkpoint else "checkpoint"
        raise FileNotFoundError(
            f"incomplete {checkpoint_type} at {checkpoint_dir}; missing: {', '.join(missing)}"
        )

    training_state_path = checkpoint_dir / "training_state.json"
    try:
        training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse {training_state_path}: {exc}") from exc

    step_index = int(training_state.get("step_index", 0))
    if step_index <= 0:
        raise RuntimeError(f"invalid step_index in {training_state_path}: {step_index}")

    return training_state


def _rebuild_history_best_from_metrics(
    *,
    metrics_path: Path,
    rows_per_episode: int,
    history_decay: float,
    max_step_index: int | None = None,
) -> dict[int, float]:
    """Reconstruct day-level history-best state from prior metrics.jsonl."""
    if not metrics_path.exists():
        return {}

    history_best_day_reward: dict[int, float] = {}
    try:
        prior_rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception:
        return {}

    for row in prior_rows:
        step_index = int(row.get("step_index", 0))
        if max_step_index is not None and step_index > int(max_step_index):
            continue
        skip_valid_steps = int(row.get("skip_valid_steps", -1))
        total_winner_reward = float(row.get("total_winner_relative_reward", 0.0))
        if step_index <= 0 or skip_valid_steps < 0:
            continue
        episode = (step_index - 1) // rows_per_episode + 1
        day_in_ep = (step_index - 1) % rows_per_episode
        if day_in_ep == 0 and episode > 1:
            for key in list(history_best_day_reward.keys()):
                history_best_day_reward[key] *= history_decay
        prev_day_best = history_best_day_reward.get(skip_valid_steps, float("-inf"))
        if total_winner_reward > prev_day_best:
            history_best_day_reward[skip_valid_steps] = total_winner_reward

    return history_best_day_reward


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_prompt_text(tokenizer: Any, backend: TransformersSamplingBackend, system_prompt: str, user_prompt: str) -> str:
    user_prompt = backend._maybe_disable_qwen_thinking(user_prompt)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


NUM_ZONES = 8  # Miami model: 8 thermal zones
_VALID_MODE_PREFIXES = ("cooling", "balanced", "energy_saving")

def _validate_setpoint_output(raw_output: str) -> bool:
    """Check raw_output strictly matches expected output template.

    Valid formats (and nothing else):
      fixed: '{"setpoints": [8 floats]}'
      free:  'cooling\\n{"setpoints": [8 floats]}'
             'balanced\\n{"setpoints": [8 floats]}'
             'energy_saving\\n{"setpoints": [8 floats]}'

    Rejects <think> tags, unexpected prefixes, wrong setpoint count, non-numeric values.
    """
    if "<think>" in raw_output:
        return False
    text = raw_output.strip()

    # Split into optional prefix and JSON part
    if text.startswith("{"):
        json_str = text
    elif "\n" in text:
        prefix, json_str = text.split("\n", 1)
        prefix = prefix.strip().lower()
        json_str = json_str.strip()
        if prefix not in _VALID_MODE_PREFIXES:
            return False
    else:
        return False

    # Validate JSON: must be exactly {"setpoints": [8 floats]}
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict) or "setpoints" not in data:
        return False
    sp = data["setpoints"]
    if not isinstance(sp, list) or len(sp) != NUM_ZONES:
        return False
    for v in sp:
        try:
            float(v)
        except (TypeError, ValueError):
            return False
    return True


def _normalize_raw_output(raw_output: str) -> str:
    """Normalize LLM output to canonical JSON form for consistent token count.

    Strips <think> tags, extracts the JSON payload, and re-serializes it
    with compact formatting. This prevents KL explosion caused by extra
    whitespace, newlines, or thinking tokens that the reference model
    assigns near-zero probability to.
    """
    import re
    # Strip thinking tags
    text = re.sub(r"<think>.*?</think>\s*", "", raw_output, flags=re.DOTALL).strip()
    # Try to extract and re-serialize JSON
    try:
        # Find JSON object
        start = text.index("{")
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        json_str = text[start:end]
        data = json.loads(json_str)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except (ValueError, json.JSONDecodeError):
        # If JSON extraction fails, return stripped text as-is
        return text


def _accumulate_block_gradient(
    *,
    model: Any,
    tokenizer: Any,
    backend: TransformersSamplingBackend,
    block_plan: dict[str, Any],
    advantage: float,
    block_divisor: int,
    kl_beta: float = 0.1,
    sft_adapter_state: dict | None = None,
) -> tuple[float, int, float]:
    """Compute GRPO gradient for one block candidate's completion.

    Standard GRPO loss:
        L = -(advantage * log π_θ(output|prompt)) / normalizer
            + β * KL(π_θ || π_ref) / token_count

    KL penalty uses Schulman's approximation:
        KL_t = exp(log_ratio_t) - log_ratio_t - 1
    where log_ratio_t = log π_θ(token_t) - log π_ref(token_t).

    π_ref is the base model (LoRA disabled). For non-LoRA models, KL is skipped.

    Returns (total_logprob, token_count, kl_value).
    """
    import torch

    system_prompt = block_plan.get("system_prompt")
    user_prompt = block_plan.get("user_prompt")
    raw_output = block_plan.get("raw_output")
    if not system_prompt or not user_prompt or not raw_output:
        return 0.0, 0, 0.0

    if not isinstance(raw_output, str):
        raw_output = json.dumps(raw_output, ensure_ascii=False)

    # Skip malformed outputs that would cause KL explosion.
    # Valid output must contain {"setpoints": [float, ...]} with correct zone count.
    # Anything with <think> tags or invalid format is skipped — only train on clean outputs.
    if not _validate_setpoint_output(raw_output):
        return 0.0, 0, 0.0

    device = next(model.parameters()).device
    prompt_text = _build_prompt_text(tokenizer, backend, system_prompt, user_prompt)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    completion_ids = tokenizer(raw_output, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    token_count = int(completion_ids.numel())
    if token_count <= 0:
        return 0.0, 0, 0.0

    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    prompt_len = int(prompt_ids.numel())
    full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)

    normalizer = float(max(token_count, 1) * max(block_divisor, 1))
    advantage_value = float(advantage)

    # --- Reference model log-probs (KL anchor against SFT snapshot) ---
    ref_logprobs = None
    if kl_beta > 0 and sft_adapter_state is not None:
        with torch.no_grad():
            # Temporarily swap in SFT adapter weights
            current_state = {}
            for name, param in model.named_parameters():
                if name in sft_adapter_state:
                    current_state[name] = param.data.clone()
                    param.data.copy_(sft_adapter_state[name])
            ref_outputs = model(input_ids=full_ids, use_cache=False)
            ref_completion_logits = ref_outputs.logits[:, prompt_len - 1: prompt_len - 1 + token_count, :]
            ref_logprobs = torch.log_softmax(ref_completion_logits.float(), dim=-1).gather(
                dim=-1,
                index=completion_ids.view(1, token_count, 1),
            ).squeeze(-1).detach()
            # Restore current adapter weights
            for name, param in model.named_parameters():
                if name in current_state:
                    param.data.copy_(current_state[name])
            del ref_outputs, ref_completion_logits, current_state
    elif kl_beta > 0 and hasattr(model, "disable_adapter_layers"):
        # Fallback: use base model as reference (for non-SFT runs)
        with torch.no_grad():
            model.disable_adapter_layers()
            ref_outputs = model(input_ids=full_ids, use_cache=False)
            ref_completion_logits = ref_outputs.logits[:, prompt_len - 1: prompt_len - 1 + token_count, :]
            ref_logprobs = torch.log_softmax(ref_completion_logits.float(), dim=-1).gather(
                dim=-1,
                index=completion_ids.view(1, token_count, 1),
            ).squeeze(-1).detach()
            model.enable_adapter_layers()
            del ref_outputs, ref_completion_logits

    # --- Current model log-probs (with gradient) ---
    outputs = model(input_ids=full_ids, use_cache=False)
    completion_logits = outputs.logits[:, prompt_len - 1: prompt_len - 1 + token_count, :]
    completion_logprobs = torch.log_softmax(completion_logits.float(), dim=-1).gather(
        dim=-1,
        index=completion_ids.view(1, token_count, 1),
    ).squeeze(-1)

    # --- Old log-probs (from rollout time, for importance ratio clipping) ---
    old_logprobs = block_plan.get("_old_logprobs")
    if old_logprobs is None:
        # Compute old log-probs with no-grad (first time seeing this plan)
        with torch.no_grad():
            old_outputs = model(input_ids=full_ids, use_cache=False)
            old_completion_logits = old_outputs.logits[:, prompt_len - 1: prompt_len - 1 + token_count, :]
            old_logprobs = torch.log_softmax(old_completion_logits.float(), dim=-1).gather(
                dim=-1,
                index=completion_ids.view(1, token_count, 1),
            ).squeeze(-1).detach()
            del old_outputs, old_completion_logits
            # Cache for reuse within same block
            block_plan["_old_logprobs"] = old_logprobs

    # --- Policy gradient loss with PPO-style clipping ---
    CLIP_EPS = 0.2
    log_ratio = completion_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
    # Per-token clipped surrogate
    pg_loss_unclipped = -advantage_value * ratio
    pg_loss_clipped = -advantage_value * clipped_ratio
    if advantage_value >= 0:
        pg_loss_per_token = torch.max(pg_loss_unclipped, pg_loss_clipped)
    else:
        pg_loss_per_token = torch.min(pg_loss_unclipped, pg_loss_clipped)
    pg_loss = pg_loss_per_token.sum() / normalizer

    trajectory_logprob = completion_logprobs.sum()

    # --- KL penalty (Schulman approximation: exp(r) - r - 1, always >= 0) ---
    kl_value = 0.0
    if ref_logprobs is not None:
        kl_log_ratio = completion_logprobs - ref_logprobs
        kl_per_token = torch.exp(kl_log_ratio) - kl_log_ratio - 1.0
        kl_sum = kl_per_token.sum()
        kl_value = float(kl_sum.detach().item())
        kl_loss = kl_beta * kl_sum / float(max(token_count, 1))
    else:
        kl_loss = torch.tensor(0.0, device=device)

    # --- Total GRPO loss ---
    loss = pg_loss + kl_loss
    loss.backward()
    total_logprob = float(trajectory_logprob.detach().item())

    del outputs, completion_logits, completion_logprobs, trajectory_logprob
    del pg_loss, kl_loss, loss, full_ids, prompt_ids, completion_ids
    if ref_logprobs is not None:
        del ref_logprobs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total_logprob, token_count, kl_value


def _grad_norm(model: Any) -> float:
    total_sq = 0.0
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach().float()
        total_sq += float((grad * grad).sum().item())
    return total_sq ** 0.5


def _write_phase(handle: Any, *, step_index: int, phase: str, **payload: Any) -> None:
    row = {"ts": time.time(), "step_index": int(step_index), "phase": phase}
    row.update(payload)
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    print(f"[phase] step={step_index} phase={phase} payload={payload}", flush=True)


def _save_reflections_safely(block_planner: UnifiedBlockPlanner, path: Path) -> bool:
    try:
        block_planner.save_reflections(path)
        return True
    except Exception as exc:
        print(f"Warning: failed to save reflections to {path}: {exc}", flush=True)
        return False


def _save_training_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    block_planner: UnifiedBlockPlanner,
    checkpoint_dir: Path,
    step_index: int,
    phase_name: str,
    include_tokenizer: bool,
    checkpoint_kind: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    if include_tokenizer:
        tokenizer.save_pretrained(checkpoint_dir)
    try:
        import torch as _t
        _t.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        (checkpoint_dir / "training_state.json").write_text(
            json.dumps({
                "step_index": int(step_index),
                "phase": str(phase_name),
                "checkpoint_kind": str(checkpoint_kind),
            }),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"  Warning: failed to save optimizer/training state to {checkpoint_dir}: {exc}", flush=True)
    _save_reflections_safely(block_planner, checkpoint_dir / "reflections.json")


def _cleanup_cache_checkpoints(
    *,
    output_dir: Path,
    episode_start_step: int,
    episode_end_step: int,
    keep_cache_checkpoints: bool,
) -> list[str]:
    if keep_cache_checkpoints:
        return []
    removed: list[str] = []
    for step in range(int(episode_start_step), int(episode_end_step) + 1):
        cache_dir = output_dir / f"cache-checkpoint-{step}"
        if not cache_dir.exists():
            continue
        try:
            shutil.rmtree(cache_dir)
            removed.append(str(cache_dir))
        except Exception as exc:
            print(f"Warning: failed to remove cache checkpoint {cache_dir}: {exc}", flush=True)
    return removed


def _summarize_reward_breakdown(
    *,
    block_reward_trace: list[dict[str, Any]],
    relative_reward: float,
    sample_index: int,
    mode: str,
    sample_type: str,
    energy_weight: float,
) -> dict[str, Any]:
    reward_sum = sum(float(s.get("reward", 0.0)) for s in block_reward_trace)
    hvac_kwh = sum(float(s.get("hvac_kwh", 0.0)) for s in block_reward_trace)
    net_grid_kwh = sum(float(s.get("net_grid_kwh", 0.0)) for s in block_reward_trace)
    pmv_violation = sum(float(s.get("total_pmv_violation", 0.0)) for s in block_reward_trace)
    energy_reward = -0.01 * float(energy_weight) * net_grid_kwh
    pmv_reward = -0.01 * 50.0 * pmv_violation
    return {
        "sample_index": int(sample_index),
        "mode": str(mode),
        "sample_type": str(sample_type),
        "relative_reward": float(relative_reward),
        "reward_sum": float(reward_sum),
        "energy_reward": float(energy_reward),
        "pmv_reward": float(pmv_reward),
        "hvac_kwh": float(hvac_kwh),
        "net_grid_kwh": float(net_grid_kwh),
        "pmv_violation": float(pmv_violation),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    trajectory_path = args.output_dir / "trajectory_samples.jsonl"
    phase_trace_path = args.output_dir / "phase_trace.jsonl"

    rows = _load_rows(args.dataset_path)
    if not rows:
        raise RuntimeError(f"No rows found in dataset: {args.dataset_path}")

    _set_seed(int(args.seed))

    resume_checkpoint_dir = Path(args.resume_from).expanduser().resolve() if args.resume_from else None
    resume_step_index = 1
    resume_training_state: dict[str, Any] = {}
    resume_source_metrics_path: Path | None = None
    if resume_checkpoint_dir is not None:
        resume_training_state = _load_resume_training_state(resume_checkpoint_dir)
        resume_step_index = int(resume_training_state["step_index"]) + 1
        candidate_metrics = resume_checkpoint_dir.parent / "metrics.jsonl"
        if candidate_metrics.exists():
            resume_source_metrics_path = candidate_metrics

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=getattr(torch, args.torch_dtype),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    if args.resume_from:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(resume_checkpoint_dir), is_trainable=True)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print(f"Resumed PEFT adapter from {resume_checkpoint_dir}")
    elif args.use_peft:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(args.learning_rate),
        )
        print("Using 8-bit AdamW (bitsandbytes)", flush=True)
    except ImportError:
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(args.learning_rate),
        )
        print("Using standard AdamW (bitsandbytes not available)", flush=True)

    if resume_checkpoint_dir is not None:
        optimizer_path = resume_checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            try:
                optimizer_state = torch.load(optimizer_path, map_location="cpu")
                optimizer.load_state_dict(optimizer_state)
                print(f"Loaded optimizer state from {optimizer_path}", flush=True)
            except Exception as exc:
                print(f"Warning: failed to load optimizer state from {optimizer_path}: {exc}", flush=True)

    bandit = MiamiGRPOBandit(
        include_forecast=True,
        control_window_start="06:00",
        control_window_end="19:00",
        weekday_only=True,
        request_mode="step_action",
        building_path=args.building_idf,
        weather_path=args.weather_epw,
    )
    backend = TransformersSamplingBackend(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name_or_path,
        max_output_tokens=int(args.max_output_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
    )
    # SFT adapter snapshot disabled (SFT approach abandoned, pure GRPO now)
    _sft_adapter_state = None
    print("KL reference: disabled (pure GRPO, no SFT snapshot)", flush=True)

    block_planner = UnifiedBlockPlanner(
        backend,
        constraints=PlannerConstraints(
            min_setpoint_c=20.0,
            max_setpoint_c=30.0,
            max_delta_per_step_c=2.0,
            fallback_setpoint_c=24.0,
            quantization_c=0.1,
        ),
        zone_ids=bandit.zone_ids,
        max_generation_attempts=2,
    )
    baseline_action = {zone_id: {"thermostat": 23.0} for zone_id in bandit.zone_ids}

    if resume_checkpoint_dir is not None:
        resume_reflections_path = resume_checkpoint_dir / "reflections.json"
        if resume_reflections_path.exists():
            try:
                block_planner.load_reflections(resume_reflections_path)
                print(f"Loaded reflections from {resume_reflections_path}", flush=True)
            except Exception as exc:
                print(f"Warning: failed to load reflections from {resume_reflections_path}: {exc}", flush=True)
        else:
            print(f"No checkpoint reflections found at {resume_reflections_path}", flush=True)

    # --- WandB setup ---
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run_name = args.wandb_name or args.output_dir.name
            wandb_run = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=wandb_run_name,
                job_type="train",
                config={
                    "learning_rate": float(args.learning_rate),
                    "reward_scale": float(args.reward_scale),
                    "kl_beta": float(args.kl_beta),
                    "kl_guard_threshold": KL_GUARD_THRESHOLD,
                    "phase_dependent_kl_lr": True,
                    "phase_boundaries": str(PHASE_BOUNDARIES),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "max_output_tokens": int(args.max_output_tokens),
                    "max_steps": int(args.max_steps),
                    "model_name_or_path": args.model_name_or_path,
                    "use_peft": bool(args.use_peft),
                    "lora_r": int(args.lora_r),
                    "lora_alpha": int(args.lora_alpha),
                    "dataset_path": str(args.dataset_path),
                    "dataset_rows": len(rows),
                    "candidate_modes": CANDIDATE_MODES,
                    "blocks_per_day": len(bandit.BLOCK_DEFINITIONS),
                },
                reinit="finish_previous",
                settings=wandb.Settings(
                    x_disable_stats=True,
                    x_disable_machine_info=True,
                ),
            )
            wandb_run.define_metric("step_index")
            for m in ("winner_relative_reward", "avg_block_reward_std",
                      "avg_block_grad_norm", "avg_block_kl", "blocks_updated",
                      "episode"):
                wandb_run.define_metric(m, step_metric="step_index")
        except Exception:
            wandb_run = None

    started_at = time.time()
    metrics: list[dict[str, Any]] = []
    energy_weight = float(os.environ.get("RL_W_ENERGY", "1.0"))

    # History best per skip_valid_steps (day-level) for update gating
    history_best_day_reward: dict[int, float] = {}
    # History best per (skip_valid_steps, block_index) for block-level update gating.
    # Only update block weights when winner reward exceeds history best for that block.
    # Prevents LoRA drift from baseline-anchored advantage pushing all-same-sign blocks.
    history_best_block_reward: dict[tuple[int, int], float] = {}
    HISTORY_DECAY = 0.95  # decay old best by 5% each episode
    if resume_source_metrics_path is not None:
        history_best_day_reward = _rebuild_history_best_from_metrics(
            metrics_path=resume_source_metrics_path,
            rows_per_episode=len(rows),
            history_decay=HISTORY_DECAY,
            max_step_index=int(resume_training_state.get("step_index", resume_step_index - 1)),
        )
        print(
            f"Rebuilt day history-best from {resume_source_metrics_path} "
            f"({len(history_best_day_reward)} active keys)",
            flush=True,
        )

    metrics_mode = "a" if args.resume_from and metrics_path.exists() else "w"
    trajectory_mode = "a" if args.resume_from and trajectory_path.exists() else "w"
    phase_mode = "a" if args.resume_from and phase_trace_path.exists() else "w"

    with (
        metrics_path.open(metrics_mode, encoding="utf-8") as metrics_handle,
        trajectory_path.open(trajectory_mode, encoding="utf-8") as trajectory_handle,
        phase_trace_path.open(phase_mode, encoding="utf-8") as phase_handle,
    ):
        if args.resume_from:
            _write_phase(
                phase_handle,
                step_index=resume_step_index,
                phase="resume_loaded",
                resume_from=str(resume_checkpoint_dir),
                start_step=resume_step_index,
                restored_history_keys=len(history_best_day_reward),
            )
        for step_index in range(resume_step_index, int(args.max_steps) + 1):
            row = rows[(step_index - 1) % len(rows)]
            skip_valid_steps = int(row["skip_valid_steps"])
            _write_phase(phase_handle, step_index=step_index, phase="step_start",
                         dataset_index=(step_index - 1) % len(rows), skip_valid_steps=skip_valid_steps)

            # --- Get 24°C baseline for the full day (cached) ---
            baseline_result = bandit._rollout_baseline_full_day_blocks(
                skip_valid_steps=skip_valid_steps,
                baseline_action=baseline_action,
            )
            baseline_block_rewards = baseline_result["block_rewards"]

            # --- Per-block rollout and training ---
            winner_actions_history: list[dict[str, dict[str, float]]] = []
            day_block_results: list[dict[str, Any]] = []
            day_winner_knot_plans: list[list[dict[str, Any]]] = []  # for day-level gradient
            total_block_grad_updates = 0

            # Set current date for reflection context lookup
            block_planner._current_date = str(baseline_result.get("target_date", ""))
            block_planner.clear_block_results()  # reset cross-block context for new day

            # Mode collapse monitoring (reset per day)
            _mode_free_counts: dict[str, int] = {"cooling": 0, "balanced": 0, "energy_saving": 0}
            _total_free_samples = [0]  # mutable for closure access

            model.eval()

            for block_index, (block_start_time, block_end_time) in enumerate(bandit.BLOCK_DEFINITIONS):
                replay_actions = list(winner_actions_history)
                block_candidates: list[dict[str, Any]] = []
                block_rewards: list[float] = []
                block_knot_plans: list[list[dict[str, Any]]] = []  # per-candidate list of knot plans

                _write_phase(phase_handle, step_index=step_index, phase="block_start",
                             block_index=block_index, block_start=str(block_start_time))

                # --- Determine phase: how many fixed vs free samples ---
                phase_name, n_fixed, n_free, phase_kl_beta, phase_lr = _get_phase(step_index)

                # Update optimizer LR for this phase
                for pg in optimizer.param_groups:
                    pg["lr"] = phase_lr

                # Build sample specs
                sample_specs: list[dict] = []
                for i in range(n_fixed):
                    sample_specs.append({"type": "fixed", "mode": CANDIDATE_MODES[i % len(CANDIDATE_MODES)]})
                for _ in range(n_free):
                    sample_specs.append({"type": "free", "mode": None})

                # --- Parallel rollout of all samples ---
                def _rollout_sample(sample_idx: int, spec: dict) -> tuple[int, dict, str, str]:
                    for attempt in range(2):
                        try:
                            if spec["type"] == "fixed":
                                result = bandit._rollout_block_rolling(
                                    skip_valid_steps=skip_valid_steps,
                                    replay_actions=replay_actions,
                                    baseline_action=baseline_action,
                                    planner=block_planner,
                                    block_index=block_index,
                                    block_start=block_start_time,
                                    block_end=block_end_time,
                                    mode=spec["mode"],
                                )
                                return sample_idx, result, spec["mode"], "fixed"
                            else:
                                proxy = _FreeSamplePlannerProxy(block_planner)
                                result = bandit._rollout_block_rolling(
                                    skip_valid_steps=skip_valid_steps,
                                    replay_actions=replay_actions,
                                    baseline_action=baseline_action,
                                    planner=proxy,
                                    block_index=block_index,
                                    block_start=block_start_time,
                                    block_end=block_end_time,
                                    mode="balanced",  # placeholder, proxy overrides
                                )
                                return sample_idx, result, proxy.chosen_mode or "balanced", "free"
                        except Exception as exc:
                            if attempt == 0:
                                print(f"  [retry] block {block_index} sample {sample_idx} failed: {exc}, retrying...", flush=True)
                                time.sleep(2)
                            else:
                                raise

                sample_results: dict[int, tuple] = {}
                if args.sequential_rollout:
                    # Sequential rollout to avoid EP resource contention (for full-param training)
                    for i, s in enumerate(sample_specs):
                        idx, result, actual_mode, stype = _rollout_sample(i, s)
                        sample_results[idx] = (result, actual_mode, stype)
                else:
                    max_parallel = min(len(sample_specs), 3)
                    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                        futures = {executor.submit(_rollout_sample, i, s): i for i, s in enumerate(sample_specs)}
                        for future in as_completed(futures):
                            idx, result, actual_mode, stype = future.result()
                            sample_results[idx] = (result, actual_mode, stype)

                # --- Collect results in sample order ---
                for idx in range(len(sample_specs)):
                    result, actual_mode, sample_type = sample_results[idx]
                    candidate_result = result
                    candidate_block_reward = candidate_result["block_reward"]
                    baseline_block_reward = baseline_block_rewards[block_index]
                    relative_block_reward = float(args.reward_scale) * (candidate_block_reward - baseline_block_reward)

                    knot_plans = candidate_result.get("knot_plans", [])
                    block_rewards.append(relative_block_reward)
                    block_knot_plans.append(knot_plans)
                    block_candidates.append({
                        "mode": actual_mode,
                        "sample_type": sample_type,
                        "block_reward": candidate_block_reward,
                        "baseline_block_reward": baseline_block_reward,
                        "relative_block_reward": relative_block_reward,
                        "block_action_trace": candidate_result.get("block_action_trace", []),
                        "block_reward_trace": candidate_result.get("block_reward_trace", []),
                        "control_steps_applied": candidate_result.get("control_steps_applied", 0),
                        "target_date": candidate_result.get("target_date"),
                        "knot_count": len(knot_plans),
                    })

                    # Extract PMV/energy from block_reward_trace for monitoring
                    _brt = candidate_result.get("block_reward_trace", [])
                    _total_pmv = sum(s.get("total_pmv_violation", 0) for s in _brt)
                    _total_hvac = sum(s.get("hvac_kwh", 0) for s in _brt)
                    _total_netgrid = sum(s.get("net_grid_kwh", 0) for s in _brt)
                    _breakdown = _summarize_reward_breakdown(
                        block_reward_trace=_brt,
                        relative_reward=relative_block_reward,
                        sample_index=idx + 1,
                        mode=actual_mode,
                        sample_type=sample_type,
                        energy_weight=energy_weight,
                    )

                    _write_phase(phase_handle, step_index=step_index, phase="block_candidate_done",
                                 block_index=block_index, mode=actual_mode, sample_type=sample_type,
                                 relative_block_reward=relative_block_reward,
                                 knot_count=len(knot_plans),
                                 pmv_violation=round(_total_pmv, 4),
                                 hvac_kwh=round(_total_hvac, 2),
                                 net_grid_kwh=round(_total_netgrid, 2),
                                 reward_sum=round(_breakdown["reward_sum"], 4),
                                 energy_reward=round(_breakdown["energy_reward"], 4),
                                 pmv_reward=round(_breakdown["pmv_reward"], 4))

                    block_candidates[-1]["reward_breakdown"] = _breakdown

                # --- Pre-compute old log-probs for ratio clipping (BEFORE optimizer.step) ---
                import torch as _torch
                model.eval()
                with _torch.no_grad():
                    for kp_list in block_knot_plans:
                        for knot_plan in kp_list:
                            sp = knot_plan.get("system_prompt")
                            up = knot_plan.get("user_prompt")
                            ro = knot_plan.get("raw_output")
                            if not sp or not up or not ro:
                                continue
                            if not isinstance(ro, str):
                                ro = json.dumps(ro, ensure_ascii=False)
                            _device = next(model.parameters()).device
                            _prompt = _build_prompt_text(tokenizer, backend, sp, up)
                            _p_ids = tokenizer(_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).to(_device)
                            _c_ids = tokenizer(ro, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).to(_device)
                            _tc = int(_c_ids.numel())
                            if _tc <= 0:
                                continue
                            _plen = int(_p_ids.numel())
                            _full = _torch.cat([_p_ids, _c_ids], dim=0).unsqueeze(0)
                            _out = model(input_ids=_full, use_cache=False)
                            _logits = _out.logits[:, _plen - 1: _plen - 1 + _tc, :]
                            _old_lp = _torch.log_softmax(_logits.float(), dim=-1).gather(
                                dim=-1, index=_c_ids.view(1, _tc, 1)).squeeze(-1).detach()
                            knot_plan["_old_logprobs"] = _old_lp
                            del _out, _logits, _full, _p_ids, _c_ids
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()

                # --- Flat sample-level GRPO advantage ---
                rewards_tensor = torch.tensor(block_rewards, dtype=torch.float32, device=device)
                block_reward_std = float(rewards_tensor.std(unbiased=False).item()) if len(block_rewards) > 1 else 0.0
                reward_std = rewards_tensor.std(unbiased=False)
                if reward_std > 1e-6:
                    sample_advantages = rewards_tensor / (reward_std + 1e-4)
                else:
                    sample_advantages = torch.zeros_like(rewards_tensor)

                # Build per-sample step reward traces
                sample_step_rewards: list[list[float]] = []
                for cand in block_candidates:
                    trace = cand.get("block_reward_trace", [])
                    sample_step_rewards.append([float(e["reward"]) for e in trace])

                block_token_counts = []
                block_loss_value = 0.0
                block_kl_value = 0.0
                has_signal = float(torch.max(torch.abs(sample_advantages)).item()) >= 1e-8

                # Block-level history-best gating: only update when winner beats history best
                _block_hb_key = (skip_valid_steps, block_index)
                _block_winner_reward = max(block_rewards)
                _block_prev_best = history_best_block_reward.get(_block_hb_key, float("-inf"))
                _block_beats_history = _block_winner_reward > _block_prev_best
                if _block_beats_history:
                    history_best_block_reward[_block_hb_key] = _block_winner_reward

                _do_block_update = has_signal and (_block_beats_history or args.no_block_history_best)

                if _do_block_update:
                    optimizer.zero_grad(set_to_none=True)
                    model.train()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    n_samples = len(sample_specs)
                    for sample_idx in range(n_samples):
                        sample_adv = float(sample_advantages[sample_idx].item())
                        knot_plans = block_knot_plans[sample_idx]
                        step_rewards = sample_step_rewards[sample_idx]
                        n_knots = len(knot_plans)
                        divisor = max(n_samples * max(n_knots, 1), 1)
                        cand_tc = 0

                        # Per-knot partial return
                        knot_partial_returns = [sum(step_rewards[k:]) if k < len(step_rewards) else 0.0 for k in range(n_knots)]
                        if knot_partial_returns:
                            pr_mean = sum(knot_partial_returns) / len(knot_partial_returns)
                            pr_std = (sum((r - pr_mean)**2 for r in knot_partial_returns) / len(knot_partial_returns)) ** 0.5
                            knot_partial_advantages = (
                                [(r - pr_mean) / (pr_std + 1e-4) for r in knot_partial_returns]
                                if pr_std > 1e-6 else [0.0] * len(knot_partial_returns)
                            )
                        else:
                            knot_partial_advantages = [0.0] * n_knots

                        for knot_idx, knot_plan in enumerate(knot_plans):
                            knot_adv = knot_partial_advantages[knot_idx] if knot_idx < len(knot_partial_advantages) else 0.0
                            total_advantage = sample_adv + 0.3 * knot_adv
                            logprob, tc, kl_val = _accumulate_block_gradient(
                                model=model, tokenizer=tokenizer, backend=backend,
                                block_plan=knot_plan, advantage=total_advantage,
                                block_divisor=divisor, kl_beta=phase_kl_beta,
                                sft_adapter_state=_sft_adapter_state,
                            )
                            cand_tc += tc
                            block_kl_value += kl_val
                        block_token_counts.append(cand_tc)

                    import torch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    block_grad_norm = _grad_norm(model)

                    # KL guard: skip optimizer.step() if block KL exceeds threshold
                    _kl_guard_triggered = block_kl_value > KL_GUARD_THRESHOLD
                    if _kl_guard_triggered:
                        optimizer.zero_grad(set_to_none=True)
                        model.eval()
                        print(f"  [KL GUARD] step={step_index} block={block_index} "
                              f"block_kl={block_kl_value:.2e} > {KL_GUARD_THRESHOLD:.0e}, "
                              f"skipping optimizer.step()", flush=True)
                        _write_phase(phase_handle, step_index=step_index, phase="kl_guard_skip",
                                     block_index=block_index, grad_norm=block_grad_norm,
                                     block_kl=block_kl_value, phase_name=phase_name,
                                     kl_guard_threshold=KL_GUARD_THRESHOLD,
                                     sample_modes=[c["mode"] for c in block_candidates])
                    else:
                        optimizer.step()
                        total_block_grad_updates += 1
                        model.eval()

                        # Mode collapse monitoring for free samples
                        for cand in block_candidates:
                            if cand.get("sample_type") == "free":
                                _mode_free_counts[cand["mode"]] = _mode_free_counts.get(cand["mode"], 0) + 1
                                _total_free_samples[0] += 1

                        _write_phase(phase_handle, step_index=step_index, phase="block_optimizer_step",
                                     block_index=block_index, grad_norm=block_grad_norm,
                                     block_reward_std=block_reward_std, block_loss=block_loss_value,
                                     block_kl=block_kl_value, phase_name=phase_name,
                                     kl_beta=phase_kl_beta, lr=phase_lr,
                                     sample_modes=[c["mode"] for c in block_candidates])
                elif has_signal and not _do_block_update:
                    block_grad_norm = 0.0
                    block_kl_value = 0.0
                    _kl_guard_triggered = False
                    _write_phase(phase_handle, step_index=step_index, phase="block_skip_below_history",
                                 block_index=block_index, block_reward_std=block_reward_std,
                                 winner_reward=_block_winner_reward, prev_best=_block_prev_best)
                else:
                    block_grad_norm = 0.0
                    block_kl_value = 0.0
                    _kl_guard_triggered = False
                    _write_phase(phase_handle, step_index=step_index, phase="block_skip_zero_advantage",
                                 block_index=block_index, block_reward_std=block_reward_std)

                # --- Select winner and reconstruct actions for replay ---
                winner_idx = int(max(range(len(block_rewards)), key=lambda i: block_rewards[i]))
                winner_knot_plans = block_knot_plans[winner_idx]
                winner_knots = [kp["knot"] for kp in winner_knot_plans]
                winner_block_actions = bandit._expand_knots_to_env_steps(
                    winner_knots, block_index=block_index, allow_partial=True,
                )
                winner_actions_history.extend(winner_block_actions)
                day_winner_knot_plans.extend(winner_knot_plans)  # collect for day-level gradient

                # Collect raw_outputs per candidate for trajectory logging
                _candidate_raw_outputs = []
                for _kp_list in block_knot_plans:
                    _outputs = [kp.get("raw_output", "") for kp in _kp_list]
                    _candidate_raw_outputs.append(_outputs)

                day_block_results.append({
                    "block_index": block_index,
                    "block_start": str(block_start_time),
                    "block_end": str(block_end_time),
                    "winner_index": winner_idx,
                    "winner_mode": block_candidates[winner_idx]["mode"],
                    "block_rewards": block_rewards,
                    "block_reward_std": block_reward_std,
                    "block_grad_norm": block_grad_norm,
                    "block_loss": block_loss_value,
                    "block_kl": block_kl_value,
                    "candidate_modes": CANDIDATE_MODES,
                    "token_counts": block_token_counts,
                    "knots_per_candidate": [len(kp) for kp in block_knot_plans],
                    "kl_guard_skip": _kl_guard_triggered if has_signal else False,
                    "candidate_raw_outputs": _candidate_raw_outputs,
                })

                # Extract winner's PMV/HVAC from block_reward_trace for logging
                _winner_brt = block_candidates[winner_idx].get("block_reward_trace", [])
                _winner_pmv = sum(s.get("total_pmv_violation", 0) for s in _winner_brt)
                _winner_hvac = sum(s.get("hvac_kwh", 0) for s in _winner_brt)

                _write_phase(phase_handle, step_index=step_index, phase="block_done",
                             block_index=block_index, winner_mode=block_candidates[winner_idx]["mode"],
                             winner_reward=block_rewards[winner_idx])

                # Record block result for cross-block context injection
                block_planner.record_block_result(
                    block_index=block_index,
                    block_start=str(block_start_time),
                    block_end=str(block_end_time),
                    winner_mode=block_candidates[winner_idx]["mode"],
                    winner_reward=block_rewards[winner_idx],
                    hvac_kwh=_winner_hvac,
                    pmv_violation=_winner_pmv,
                )

                # --- Per-block reflection with zone PMV penalty attribution ---
                try:
                    zone_pmv_lines = []
                    winner_kp = block_knot_plans[winner_idx] if winner_idx < len(block_knot_plans) else []
                    import re as _re
                    mode_target = block_candidates[winner_idx]["mode"]
                    pmv_limits = {"cooling": (-0.5, 0.0), "balanced": (-0.1, 0.2), "energy_saving": (0.2, 0.5)}
                    lo, hi = pmv_limits.get(mode_target, (-0.5, 0.5))

                    # Accumulate per-zone PMV violation across ALL knots
                    zone_violation_total: dict[str, float] = {z: 0.0 for z in bandit.zone_ids}
                    zone_pmv_samples: dict[str, list[float]] = {z: [] for z in bandit.zone_ids}
                    for kp in winner_kp:
                        up = kp.get("user_prompt", "")
                        for zone_id in bandit.zone_ids:
                            m = _re.search(rf'{zone_id}.*?PMV=([+-]?[0-9.]+)', up)
                            if m:
                                pmv_val = float(m.group(1))
                                zone_pmv_samples[zone_id].append(pmv_val)
                                if pmv_val > hi:
                                    zone_violation_total[zone_id] += pmv_val - hi
                                elif pmv_val < lo:
                                    zone_violation_total[zone_id] += lo - pmv_val

                    # Report top violating zones with their penalty contribution
                    total_penalty = sum(zone_violation_total.values())
                    if total_penalty > 0.1:
                        sorted_zones = sorted(zone_violation_total.items(), key=lambda x: -x[1])
                        for zone_id, viol in sorted_zones:
                            if viol > 0.05:
                                pmvs = zone_pmv_samples[zone_id]
                                avg_pmv = sum(pmvs) / len(pmvs) if pmvs else 0
                                pct = viol / total_penalty * 100
                                if avg_pmv > hi:
                                    zone_pmv_lines.append(
                                        f"  {zone_id}: avg PMV={avg_pmv:+.2f} (target {lo:+.1f}~{hi:+.1f}), penalty={viol:.2f} ({pct:.0f}% of total) → needs LOWER setpoint")
                                elif avg_pmv < lo:
                                    zone_pmv_lines.append(
                                        f"  {zone_id}: avg PMV={avg_pmv:+.2f} (target {lo:+.1f}~{hi:+.1f}), penalty={viol:.2f} ({pct:.0f}% of total) → needs HIGHER setpoint")

                    zone_pmv_summary = ""
                    if zone_pmv_lines:
                        zone_pmv_summary = f"Zone PMV penalty attribution (total={total_penalty:.2f}):\n" + "\n".join(zone_pmv_lines)

                    if not args.no_block_reflection:
                        candidate_breakdowns = []
                        for _i, _cand in enumerate(block_candidates):
                            _bd = dict(_cand.get("reward_breakdown", {}))
                            _bd["is_winner"] = _i == winner_idx
                            candidate_breakdowns.append(_bd)

                        block_planner.generate_block_reflection(
                            date=str(baseline_result.get("target_date", f"step{step_index}")),
                            block_index=block_index,
                            block_start=str(block_start_time),
                            block_end=str(block_end_time),
                            all_mode_rewards=dict(zip(CANDIDATE_MODES, block_rewards)),
                            winner_mode=block_candidates[winner_idx]["mode"],
                            candidate_breakdowns=candidate_breakdowns,
                            winner_index=winner_idx,
                            zone_pmv_summary=zone_pmv_summary,
                        )
                except Exception:
                    pass

            # --- Write trajectory info ---
            trajectory_handle.write(json.dumps({
                "step_index": step_index,
                "target_date": baseline_result.get("target_date"),
                "skip_valid_steps": skip_valid_steps,
                "block_results": day_block_results,
                "baseline_block_rewards": baseline_block_rewards,
                "baseline_cache_hit": baseline_result.get("baseline_cache_hit", False),
                "total_block_grad_updates": total_block_grad_updates,
            }, ensure_ascii=False) + "\n")
            trajectory_handle.flush()

            # --- Mode collapse monitoring ---
            if _total_free_samples[0] > 0:
                import math
                _mode_dist = {m: _mode_free_counts.get(m, 0) / _total_free_samples[0] for m in CANDIDATE_MODES}
                _entropy = -sum(p * math.log(p + 1e-10) for p in _mode_dist.values())
                _max_ent = math.log(len(CANDIDATE_MODES))
                _norm_ent = _entropy / _max_ent if _max_ent > 0 else 0.0
                _write_phase(phase_handle, step_index=step_index, phase="mode_distribution",
                             mode_counts=_mode_free_counts, total_free=_total_free_samples[0],
                             distribution=_mode_dist, normalized_entropy=round(_norm_ent, 4))
                print(f"  [mode dist] free={_total_free_samples[0]} dist={_mode_dist} entropy={_norm_ent:.3f}", flush=True)

            # --- Aggregate metrics for this step ---
            total_winner_reward = sum(
                br["block_rewards"][br["winner_index"]] for br in day_block_results
            )

            # --- Day-level history best gating + gradient ---
            day_grad_norm = 0.0
            day_kl_value = 0.0
            DAY_ADVANTAGE_SCALE = 0.3

            # Decay old history bests at start of each new episode
            episode = (step_index - 1) // len(rows) + 1
            day_in_ep = (step_index - 1) % len(rows)
            if day_in_ep == 0 and episode > 1:
                for k in history_best_day_reward:
                    history_best_day_reward[k] *= HISTORY_DECAY
                for k in history_best_block_reward:
                    history_best_block_reward[k] *= HISTORY_DECAY

            prev_day_best = history_best_day_reward.get(skip_valid_steps, float("-inf"))
            day_beats_history = total_winner_reward > prev_day_best
            if day_beats_history:
                history_best_day_reward[skip_valid_steps] = total_winner_reward

            if abs(total_winner_reward) > 1e-6 and day_winner_knot_plans and day_beats_history:
                day_advantage = float(total_winner_reward) * DAY_ADVANTAGE_SCALE
                optimizer.zero_grad(set_to_none=True)
                model.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                divisor = max(len(day_winner_knot_plans), 1)
                for knot_plan in day_winner_knot_plans:
                    _accumulate_block_gradient(
                        model=model,
                        tokenizer=tokenizer,
                        backend=backend,
                        block_plan=knot_plan,
                        advantage=day_advantage,
                        block_divisor=divisor,
                        kl_beta=phase_kl_beta,
                        sft_adapter_state=_sft_adapter_state,
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                day_grad_norm = _grad_norm(model)
                optimizer.step()
                model.eval()
                _write_phase(phase_handle, step_index=step_index, phase="day_level_gradient",
                             day_advantage=day_advantage, day_grad_norm=day_grad_norm,
                             num_winning_knots=len(day_winner_knot_plans),
                             beats_history=True, prev_best=prev_day_best)
            elif not day_beats_history:
                _write_phase(phase_handle, step_index=step_index, phase="day_level_skip",
                             total_winner_reward=total_winner_reward,
                             prev_best=prev_day_best, reason="below_history_best")

            avg_block_reward_std = float(np.mean([br["block_reward_std"] for br in day_block_results]))
            avg_block_grad_norm = float(np.mean([br["block_grad_norm"] for br in day_block_results]))
            avg_block_kl = float(np.mean([br.get("block_kl", 0.0) for br in day_block_results]))

            metric_row = {
                "step_index": step_index,
                "dataset_index": (step_index - 1) % len(rows),
                "skip_valid_steps": skip_valid_steps,
                "target_date": baseline_result.get("target_date"),
                "total_winner_relative_reward": total_winner_reward,
                "avg_block_reward_std": avg_block_reward_std,
                "avg_block_grad_norm": avg_block_grad_norm,
                "total_block_grad_updates": total_block_grad_updates,
                "block_summaries": [
                    {
                        "block_index": br["block_index"],
                        "winner_mode": br["winner_mode"],
                        "block_rewards": br["block_rewards"],
                        "block_reward_std": br["block_reward_std"],
                        "block_grad_norm": br["block_grad_norm"],
                    }
                    for br in day_block_results
                ],
                "phase_kl_beta": phase_kl_beta,
                "phase_lr": phase_lr,
                "kl_guard_skips": sum(1 for br in day_block_results if br.get("kl_guard_skip")),
                "elapsed_s": time.time() - started_at,
            }
            metrics.append(metric_row)
            metrics_handle.write(json.dumps(metric_row, ensure_ascii=False) + "\n")
            metrics_handle.flush()

            _write_phase(phase_handle, step_index=step_index, phase="step_done",
                         total_winner_reward=total_winner_reward,
                         avg_block_reward_std=avg_block_reward_std,
                         phase_kl_beta=phase_kl_beta, phase_lr=phase_lr)

            # --- Reflexion: generate day reflection and inject into planner ---
            try:
                reflection_block_results = [
                    {
                        "block_index": br["block_index"],
                        "block_start": br["block_start"],
                        "block_end": br["block_end"],
                        "winner_mode": br["winner_mode"],
                        "winner_reward": br["block_rewards"][br["winner_index"]],
                        "baseline_reward": baseline_block_rewards[br["block_index"]],
                        "relative_reward": br["block_rewards"][br["winner_index"]],
                        "all_mode_rewards": dict(zip(br["candidate_modes"], br["block_rewards"])),
                    }
                    for br in day_block_results
                ]
                # Build weather summary from block action traces
                weather_lines = []
                for br in day_block_results:
                    trace = br.get("block_action_trace", []) or []
                    if not trace:
                        # Try from candidates
                        for cand in block_candidates:
                            if cand.get("mode") == br.get("winner_mode"):
                                trace = cand.get("block_action_trace", [])
                                break
                    if trace:
                        first = trace[0] if trace else {}
                        last = trace[-1] if trace else {}
                        weather_lines.append(
                            f"  Block {br['block_index']+1} ({br['block_start']}-{br['block_end']}): "
                            f"start={first.get('wallclock','?')}, end={last.get('wallclock','?')}"
                        )
                # Extract PV and temperature from the dataset row's observation
                row_obs = row.get("observation", {})
                first_zone = list(row_obs.values())[0] if row_obs else {}
                pv_kwh = float(row.get("payload", {}).get("pv_kwh", 0))
                forecast = row.get("payload", {}).get("forecast", {})
                forecast_available = bool(forecast.get("available", False))
                weather_summary = f"Morning PV: {pv_kwh:.1f} kWh\n"
                if forecast_available:
                    temp_6h = forecast.get("temperature_6h_c", [])
                    cloud_6h = forecast.get("cloudcover_6h_pct", []) or []
                    precip_6h = forecast.get("precip_prob_6h_pct", []) or []
                    if temp_6h:
                        weather_summary += f"Forecast temperature (6h): {[round(t,1) for t in temp_6h[:6]]}\n"
                    if cloud_6h:
                        weather_summary += f"Forecast cloud cover (6h): {[round(c,1) for c in cloud_6h[:6]]}\n"
                    if precip_6h:
                        weather_summary += f"Forecast precip prob (6h): {[round(p,1) for p in precip_6h[:6]]}\n"

                reflection = block_planner.generate_day_reflection(
                    date=str(baseline_result.get("target_date", f"step{step_index}")),
                    block_results=reflection_block_results,
                    total_reward=total_winner_reward + sum(baseline_block_rewards),
                    baseline_reward=sum(baseline_block_rewards),
                    weather_summary=weather_summary,
                )
                _write_phase(phase_handle, step_index=step_index, phase="reflection",
                             reflection=reflection[:500])
                latest_reflections_path = args.output_dir / "reflections.latest.json"
                if _save_reflections_safely(block_planner, latest_reflections_path):
                    _write_phase(phase_handle, step_index=step_index, phase="reflections_latest_saved",
                                 reflections_path=str(latest_reflections_path))
            except Exception as exc:
                _write_phase(phase_handle, step_index=step_index, phase="reflection_error",
                             error=str(exc)[:200])

            # --- WandB logging ---
            if wandb_run is not None:
                episode = (step_index - 1) // len(rows) + 1
                wandb_run.log({
                    "step_index": step_index,
                    "episode": episode,
                    "winner_relative_reward": total_winner_reward,
                    "avg_block_reward_std": avg_block_reward_std,
                    "avg_block_grad_norm": avg_block_grad_norm,
                    "avg_block_kl": avg_block_kl,
                    "blocks_updated": total_block_grad_updates,
                    "elapsed_s": time.time() - started_at,
                })
                # Per-block metrics
                for br in day_block_results:
                    bi = br["block_index"]
                    wandb_run.log({
                        "step_index": step_index,
                        f"block{bi}/reward_std": br["block_reward_std"],
                        f"block{bi}/grad_norm": br["block_grad_norm"],
                        f"block{bi}/kl": br.get("block_kl", 0.0),
                        f"block{bi}/winner_reward": br["block_rewards"][br["winner_index"]],
                    })

            # --- Checkpoint ---
            wrote_full_checkpoint = False
            if int(args.save_steps) > 0 and step_index % int(args.save_steps) == 0:
                checkpoint_dir = args.output_dir / f"checkpoint-{step_index}"
                _save_training_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    block_planner=block_planner,
                    checkpoint_dir=checkpoint_dir,
                    step_index=step_index,
                    phase_name=phase_name,
                    include_tokenizer=True,
                    checkpoint_kind="full",
                )
                wrote_full_checkpoint = True
                _write_phase(phase_handle, step_index=step_index, phase="checkpoint_saved",
                             checkpoint_dir=str(checkpoint_dir))

            if (
                int(args.cache_steps) > 0
                and step_index % int(args.cache_steps) == 0
                and not wrote_full_checkpoint
            ):
                cache_dir = args.output_dir / f"cache-checkpoint-{step_index}"
                _save_training_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    block_planner=block_planner,
                    checkpoint_dir=cache_dir,
                    step_index=step_index,
                    phase_name=phase_name,
                    include_tokenizer=False,
                    checkpoint_kind="cache",
                )
                _write_phase(phase_handle, step_index=step_index, phase="cache_checkpoint_saved",
                             checkpoint_dir=str(cache_dir))

            if step_index % len(rows) == 0:
                episode_start_step = step_index - len(rows) + 1
                episode_checkpoint_dir = args.output_dir / f"checkpoint-{step_index}"
                if episode_checkpoint_dir.exists():
                    removed_cache = _cleanup_cache_checkpoints(
                        output_dir=args.output_dir,
                        episode_start_step=episode_start_step,
                        episode_end_step=step_index,
                        keep_cache_checkpoints=bool(args.keep_cache_checkpoints),
                    )
                    if removed_cache:
                        _write_phase(phase_handle, step_index=step_index, phase="cache_checkpoints_cleaned",
                                     episode_start_step=episode_start_step,
                                     episode_end_step=step_index,
                                     removed_count=len(removed_cache))
                else:
                    _write_phase(phase_handle, step_index=step_index, phase="cache_cleanup_skipped",
                                 reason="no_full_episode_checkpoint",
                                 episode_start_step=episode_start_step,
                                 episode_end_step=step_index)

    summary = {
        "status": "completed",
        "output_dir": str(args.output_dir),
        "dataset_path": str(args.dataset_path),
        "model_name_or_path": args.model_name_or_path,
        "max_steps": int(args.max_steps),
        "candidate_modes": CANDIDATE_MODES,
        "blocks_per_day": len(bandit.BLOCK_DEFINITIONS),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "learning_rate_base": float(args.learning_rate),
        "kl_beta_base": float(args.kl_beta),
        "kl_guard_threshold": KL_GUARD_THRESHOLD,
        "phase_boundaries": str(PHASE_BOUNDARIES),
        "reward_scale": float(args.reward_scale),
        "use_peft": bool(args.use_peft),
        "rows_available": len(rows),
        "elapsed_s": time.time() - started_at,
        "last_metric": metrics[-1] if metrics else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)

    # --- Save and compress reflections ---
    try:
        ref_path = output_dir / "reflections.json"
        block_planner.save_reflections(ref_path)
        print(f"Saved reflections to {ref_path}", flush=True)

        print("Compressing reflections into rules...", flush=True)
        rules = block_planner.compress_reflections()
        print(f"Compressed rules:\n{rules}", flush=True)

        # Re-save with compressed rules
        block_planner.save_reflections(ref_path)
        print(f"Updated reflections with compressed rules", flush=True)
    except Exception as exc:
        print(f"Warning: failed to save/compress reflections: {exc}", flush=True)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
