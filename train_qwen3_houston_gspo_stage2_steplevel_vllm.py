#!/usr/bin/env python3
"""Stage 2: Day-level GRPO trainer.

Resumes from Stage 1 checkpoint (block-level GRPO) and trains with day-level
reward comparison. Each step runs 3 full-day rollouts (13 blocks each, all
free-mode), then applies a single GRPO gradient update per day.

Advantage = day_adv + 0.3 * block_cross_rollout_adv

Update gating:
  - update whenever the sampled day rollouts have advantage signal
  - history-best is tracked for diagnostics only, not used as an optimizer gate
  - no signal (std ≈ 0) → skip

Mode-PMV consistency penalty: penalise when actual PMV deviates from the
selected mode's target range (training-only signal, not used in eval).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/AD/user/lab/asim")
SHARED_SITE_PACKAGES = PROJECT_ROOT / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if SHARED_SITE_PACKAGES.exists() and str(SHARED_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(SHARED_SITE_PACKAGES))

import numpy as np

from grpo_miami_bandit import MiamiGRPOBandit, RESULT_DIR, _plainify
from llm_setpoint_planner import (
    PlannerConstraints,
    TransformersSamplingBackend,
    _as_float,
    estimate_zone_pmv,
)
from llm_setpoint_planner_unified import UnifiedBlockPlanner

# Import shared utilities from Stage 1 (no modification to Stage 1 files)
from train_qwen3_houston_gspo_unified import (
    _accumulate_block_gradient,
    _build_prompt_text,
    _grad_norm,
    _load_rows,
    _normalize_raw_output,
    _rebuild_history_best_from_metrics,
    _save_reflections_safely,
    _save_training_checkpoint,
    _set_seed,
    _summarize_reward_breakdown,
    _validate_setpoint_output,
    _write_phase,
    NUM_ZONES,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

KL_GUARD_THRESHOLD = 5e3
HISTORY_DECAY = 0.95

PMV_TARGETS: dict[str, tuple[float, float]] = {
    "cooling": (-0.5, 0.0),
    "balanced": (-0.1, 0.2),
    "energy_saving": (0.2, 0.5),
}


def _resume_start_step(resume_checkpoint_dir: Path) -> int:
    """Continue from Stage 2 checkpoints while keeping Stage 1 starts at step 1."""
    state_path = resume_checkpoint_dir / "training_state.json"
    if not state_path.exists():
        return 1
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to read training state from {state_path}: {exc}", flush=True)
        return 1
    if state.get("phase") != "stage2":
        return 1
    try:
        completed_step = int(state.get("step_index", 0))
    except (TypeError, ValueError):
        return 1
    if completed_step <= 0:
        return 1
    return completed_step + 1


def _validate_miami_forecast_binding(bandit: MiamiGRPOBandit) -> None:
    """Fail early if the Miami Stage 2 env silently fell back to another forecast."""
    env_value = os.environ.get("RL_FORECAST_CSV")
    if not env_value:
        raise RuntimeError(
            "RL_FORECAST_CSV is not set. The env cell defaults to the Houston "
            "forecast file, so Miami Stage 2 must set "
            "RL_FORECAST_CSV=miami_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6.csv."
        )

    env_mod = bandit.env_mod
    forecast_path_value = getattr(env_mod, "FORECAST_CSV_PATH", None)
    if forecast_path_value is None:
        raise RuntimeError("Env module does not expose FORECAST_CSV_PATH.")

    forecast_path = Path(forecast_path_value).expanduser().resolve()
    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast CSV does not exist: {forecast_path}")
    if "miami" not in forecast_path.name.lower():
        raise RuntimeError(
            f"Miami Stage 2 is bound to a non-Miami forecast CSV: {forecast_path}. "
            "Check RL_FORECAST_CSV before training."
        )

    forecast_reader = getattr(env_mod, "forecast_reader", None)
    if forecast_reader is None:
        raise RuntimeError(f"Forecast reader failed to initialize for {forecast_path}.")

    missing_summary = getattr(forecast_reader, "missing_summary", None)
    forecast_years = getattr(forecast_reader, "_years", None)
    print(
        f"Forecast CSV binding: RL_FORECAST_CSV={env_value} -> {forecast_path}",
        flush=True,
    )
    if forecast_years:
        print(f"Forecast years: {sorted(int(y) for y in forecast_years)}", flush=True)
    if missing_summary is not None:
        print(f"Forecast missing/fill summary: {missing_summary}", flush=True)


def _adapter_state_path(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def _adapter_key_candidates(key: str) -> list[str]:
    candidates = [key]
    for adapter_name in ("default",):
        candidates.append(key.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight"))
        candidates.append(key.replace(".lora_B.weight", f".lora_B.{adapter_name}.weight"))
    if not key.startswith("base_model.model."):
        candidates.append(f"base_model.model.{key}")
    return list(dict.fromkeys(candidates))


def _load_adapter_snapshot_for_model(
    *,
    model: Any,
    adapter_dir: Path,
    torch_module: Any,
) -> dict[str, Any]:
    """Load adapter weights from disk using this model's parameter names."""
    adapter_path = _adapter_state_path(adapter_dir)
    if adapter_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        raw_state = load_file(str(adapter_path), device="cpu")
    else:
        raw_state = torch_module.load(adapter_path, map_location="cpu")

    named_params = dict(model.named_parameters())
    snapshot: dict[str, Any] = {}
    unmatched: list[str] = []
    n_padded = 0
    for raw_name, raw_tensor in raw_state.items():
        matched_name = None
        for candidate in _adapter_key_candidates(raw_name):
            if candidate in named_params:
                matched_name = candidate
                break
        if matched_name is None:
            unmatched.append(raw_name)
            continue
        param = named_params[matched_name]
        raw_tensor = raw_tensor.to(device=param.device, dtype=param.dtype)
        if raw_tensor.shape != param.data.shape:
            # Zero-pad smaller KL reference (e.g. r=16) into current larger LoRA (e.g. r=32)
            padded = torch_module.zeros_like(param.data)
            slices = tuple(
                slice(0, min(o, n))
                for o, n in zip(raw_tensor.shape, param.data.shape)
            )
            padded[slices] = raw_tensor[slices]
            snapshot[matched_name] = padded.clone()
            n_padded += 1
        else:
            snapshot[matched_name] = raw_tensor.clone()
    if n_padded > 0:
        print(
            f"KL reference auto-padded {n_padded} params to match current LoRA shape",
            flush=True,
        )

    missing_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and name not in snapshot
    ]
    if missing_trainable:
        preview = ", ".join(missing_trainable[:5])
        raise RuntimeError(
            f"KL reference {adapter_dir} does not cover all trainable adapter "
            f"parameters; missing {len(missing_trainable)} keys, e.g. {preview}"
        )
    if unmatched:
        print(
            f"Warning: {len(unmatched)} KL reference adapter keys were not used "
            f"from {adapter_path}",
            flush=True,
        )
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: Day-level GRPO trainer.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=RESULT_DIR / "miami_gspo_dataset_stage2_30min.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "miami_grpo_stage2")
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm-tp", type=int, default=2,
                        help="vLLM tensor parallel size (1 = single-GPU shared with HF, 2 = TP across both)")
    parser.add_argument("--vllm-gpu-mem-util", type=float, default=0.45,
                        help="vLLM gpu_memory_utilization. With TP=1 sharing a GPU with HF, lower this (e.g. 0.4)")
    # Occupancy-aware fallback (when planner parser fails) — back-compat:
    # leave both --fallback-low-occ and --fallback-high-occ unset to keep the
    # old static 24°C fallback behavior.
    parser.add_argument("--fallback-low-occ", type=float, default=None,
                        help="Fallback setpoint when avg zone occ <= --fallback-occ-low-thr (e.g. 30.0 for off-AC)")
    parser.add_argument("--fallback-high-occ", type=float, default=None,
                        help="Fallback setpoint when avg zone occ >= --fallback-occ-high-thr (e.g. 23.5 for safe cooling)")
    parser.add_argument("--fallback-occ-low-thr", type=float, default=0.15,
                        help="Below this occ, use --fallback-low-occ (default 0.15)")
    parser.add_argument("--fallback-occ-high-thr", type=float, default=0.5,
                        help="At/above this occ, use --fallback-high-occ (default 0.5)")
    parser.add_argument("--seed", type=int, default=1229)
    parser.add_argument("--save-steps", type=int, default=8)
    parser.add_argument("--cache-steps", type=int, default=4)
    parser.add_argument("--keep-cache-checkpoints", action="store_true")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.15)
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--resume-from", type=str, required=True,
                        help="Path to Stage 1 or Stage 2 checkpoint")
    parser.add_argument(
        "--kl-reference-from",
        type=str,
        default=None,
        help=(
            "Adapter checkpoint used as KL reference. Defaults to --resume-from. "
            "When resuming Stage 2, pass the original Stage 1 checkpoint so KL "
            "does not reset at the recovery point."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--fresh-lora", action="store_true",
                        help="Create fresh LoRA adapter with --lora-r/--lora-alpha instead of loading from --resume-from")
    parser.add_argument("--n-rollouts", type=int, default=3)
    parser.add_argument("--parallel-rollouts", type=int, default=1,
                        help="If >1, run rollouts in parallel via ThreadPoolExecutor. "
                             "LLM serialises on GPU, but EnergyPlus subprocesses truly parallelise.")
    parser.add_argument("--consistency-penalty-weight", type=float, default=0.01)
    parser.add_argument("--mode-setpoint-penalty-weight", type=float, default=0.05)
    parser.add_argument("--mode-setpoint-local-adv-weight", type=float, default=0.2)
    parser.add_argument("--mode-exploration-steps", type=int, default=16)
    parser.add_argument("--setpoint-only", action="store_true")
    parser.add_argument("--setpoint-exploration-steps", type=int, default=32)
    parser.add_argument("--setpoint-exploration-prob", type=float, default=0.40)
    parser.add_argument("--setpoint-exploration-late-prob", type=float, default=0.15)
    parser.add_argument("--setpoint-exploration-max-blocks", type=int, default=3)
    parser.add_argument("--block-cross-adv-weight", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for step-level return-to-go")
    parser.add_argument("--advantage-mode", choices=["return_to_go", "per_knot"], default="return_to_go",
                        help="'return_to_go' (default, current best): discount future knot rewards "
                             "with gamma then z-score per knot across rollouts. 'per_knot': z-score "
                             "each knot's own reward across rollouts without any future rollup — "
                             "cleaner per-knot attribution, no cross-knot leakage.")
    parser.add_argument("--filter-truncated", action="store_true", default=True,
                        help="Skip gradient contribution from knots whose raw_output failed to "
                             "parse into a valid setpoint (e.g. thinking truncated without </think> "
                             "or without JSON). Prevents GRPO from training the model to emit "
                             "truncated-thinking patterns that happen to produce acceptable 24°C "
                             "fallback setpoints.")
    parser.add_argument("--no-filter-truncated", action="store_false", dest="filter_truncated")
    parser.add_argument("--format-penalty-weight", type=float, default=0.3,
                        help="Weight of the format-quality penalty subtracted from the knot "
                             "advantage. Penalises: (a) no closing </think> tag, (b) thinking that "
                             "quotes/rehashes the prompt instructions, (c) highly repetitive short "
                             "phrases. 0 to disable.")
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--kl-guard-threshold", type=float, default=KL_GUARD_THRESHOLD)
    parser.add_argument("--wandb-project", type=str, default="asim-miami-stage2-grpo")
    parser.add_argument("--wandb-group", type=str, default="day-level-grpo")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--building-idf", type=str, default=str(PROJECT_ROOT / "miami_stage2.idf"))
    parser.add_argument("--weather-epw", type=str, default=None)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Mode-PMV consistency penalty
# --------------------------------------------------------------------------- #

def _compute_mode_pmv_consistency_penalty(
    block_results: list[dict[str, Any]],
    weight: float,
) -> float:
    """Penalise when actual PMV deviates from the selected mode's target range.

    For each block, check the per-zone PMV from block_reward_trace against the
    mode's PMV target range.  Penalty is proportional to deviation.
    """
    penalty = 0.0
    for br in block_results:
        mode = br.get("mode", "balanced")
        pmv_lo, pmv_hi = PMV_TARGETS.get(mode, (-0.5, 0.5))
        for step_data in br.get("block_reward_trace", []):
            zone_pmvs = step_data.get("zone_pmvs", {})
            for _zone_id, pmv_val in zone_pmvs.items():
                pmv_val = float(pmv_val)
                if pmv_val > pmv_hi:
                    penalty += weight * (pmv_val - pmv_hi)
                elif pmv_val < pmv_lo:
                    penalty += weight * (pmv_lo - pmv_val)
    return penalty


def _annotate_mode_setpoint_semantic_violations(
    knot_plans: list[dict[str, Any]],
    penalty_weight: float,
) -> float:
    """Annotate knot-level mode/setpoint contradictions and return diagnostic penalty.

    The returned penalty is logged for monitoring.  The actual training signal is
    applied later as a per-knot local advantage so only the offending action is
    suppressed.
    """
    penalty = 0.0
    for knot_plan in knot_plans:
        mode = str(knot_plan.get("mode", "balanced"))
        knot = knot_plan.get("knot", {})
        if not isinstance(knot, dict) or not knot:
            knot_plan["mode_setpoint_mean"] = None
            knot_plan["mode_setpoint_violation_c"] = 0.0
            knot_plan["mode_setpoint_penalty"] = 0.0
            continue

        values: list[float] = []
        for value in knot.values():
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            knot_plan["mode_setpoint_mean"] = None
            knot_plan["mode_setpoint_violation_c"] = 0.0
            knot_plan["mode_setpoint_penalty"] = 0.0
            continue

        mean_setpoint = sum(values) / len(values)
        violation_c = 0.0
        if mode == "cooling":
            # Cooling should correspond to low cooling setpoints.
            violation_c = max(mean_setpoint - 24.5, 0.0)
        elif mode == "energy_saving":
            # Energy saving should correspond to reduced cooling.
            violation_c = max(24.5 - mean_setpoint, 0.0)
        elif mode == "balanced":
            # Keep balanced broad; only penalise extreme contradictions.
            violation_c = 0.25 * max(mean_setpoint - 27.0, 0.0)
            violation_c += 0.25 * max(20.5 - mean_setpoint, 0.0)

        semantic_penalty = max(float(penalty_weight), 0.0) * violation_c
        knot_plan["mode_setpoint_mean"] = mean_setpoint
        knot_plan["mode_setpoint_violation_c"] = violation_c
        knot_plan["mode_setpoint_penalty"] = semantic_penalty
        penalty += semantic_penalty

    return penalty


# --------------------------------------------------------------------------- #
# Day-level advantage
# --------------------------------------------------------------------------- #

def _compute_day_advantages(day_rewards: list[float]) -> list[float]:
    rewards = np.array(day_rewards, dtype=np.float64)
    std = float(rewards.std())
    if std > 1e-6:
        return ((rewards - rewards.mean()) / (std + 1e-4)).tolist()
    return [0.0] * len(rewards)


def _compute_block_cross_advantages(
    rollout_results: list[dict[str, Any]],
    n_blocks: int,
) -> list[list[float]]:
    """Per-block cross-rollout advantages.

    Returns advs[block_i][rollout_j].
    """
    n_rollouts = len(rollout_results)
    advs: list[list[float]] = []
    for block_i in range(n_blocks):
        block_rewards = []
        for j in range(n_rollouts):
            brs = rollout_results[j].get("block_results", [])
            if block_i < len(brs):
                block_rewards.append(float(brs[block_i].get("relative_reward", 0.0)))
            else:
                block_rewards.append(0.0)
        arr = np.array(block_rewards, dtype=np.float64)
        std = float(arr.std())
        if std > 1e-6:
            advs.append(((arr - arr.mean()) / (std + 1e-4)).tolist())
        else:
            advs.append([0.0] * n_rollouts)
    return advs


# --------------------------------------------------------------------------- #
# Step-level (knot-level) advantage
# --------------------------------------------------------------------------- #

def _extract_knot_rewards(
    rollout_result: dict,
    baseline_block_rewards: list[float],
    reward_scale: float,
    n_blocks: int,
) -> list[float]:
    """Extract per-knot relative rewards from a single rollout.

    Returns list of length sum(knots_per_block) = 26 for Miami Stage 2.
    Each knot reward = reward_scale * (knot_candidate_reward - knot_baseline_reward).
    """
    knot_rewards: list[float] = []
    block_results = rollout_result["block_results"]
    for bi in range(n_blocks):
        br = block_results[bi]
        brt = br.get("block_reward_trace", [])
        n_knots = max(int(br.get("knot_count", 2)), 1)
        steps_per_knot = max(len(brt) // n_knots, 1) if brt else 1

        # Baseline block reward, split evenly across knots
        baseline_per_knot = float(baseline_block_rewards[bi]) / n_knots

        for ki in range(n_knots):
            start = ki * steps_per_knot
            end = min(start + steps_per_knot, len(brt))
            knot_env_reward = sum(float(s.get("reward", 0.0)) for s in brt[start:end])
            knot_relative = reward_scale * (knot_env_reward - baseline_per_knot)
            knot_rewards.append(knot_relative)
    return knot_rewards


def _compute_step_level_advantages(
    rollout_results: list[dict],
    baseline_block_rewards: list[float],
    reward_scale: float,
    n_blocks: int,
    gamma: float = 0.99,
    advantage_mode: str = "return_to_go",
) -> list[list[float]]:
    """Compute per-knot advantages across G rollouts.

    Modes:
      - 'return_to_go' (original): discount future knot rewards with gamma,
        then z-score each column (knot position) across rollouts.
      - 'per_knot': z-score each knot's own reward (no future rollup). Cleaner
        per-knot attribution, no bleed from later knots' rewards.

    Returns advs[rollout_idx][knot_idx].
    """
    G = len(rollout_results)

    # 1) Extract per-knot rewards: shape (G, T)
    reward_matrix = []
    for rollout in rollout_results:
        knot_r = _extract_knot_rewards(
            rollout, baseline_block_rewards, reward_scale, n_blocks,
        )
        reward_matrix.append(knot_r)
    r = np.array(reward_matrix, dtype=np.float64)  # (G, T)
    T = r.shape[1]

    if advantage_mode == "per_knot":
        # Pure per-knot: z-score each column without return-to-go rollup.
        mean = r.mean(axis=0, keepdims=True)
        std = r.std(axis=0, keepdims=True)
        A = np.where(std > 1e-6, (r - mean) / (std + 1e-4), 0.0)
        return A.tolist()

    # Default: return-to-go + column z-score (original behaviour).
    G_ret = np.zeros_like(r)
    running = np.zeros(G, dtype=np.float64)
    for t in reversed(range(T)):
        running = r[:, t] + gamma * running
        G_ret[:, t] = running
    mean = G_ret.mean(axis=0, keepdims=True)
    std = G_ret.std(axis=0, keepdims=True)
    A = np.where(std > 1e-6, (G_ret - mean) / (std + 1e-4), 0.0)
    return A.tolist()


_META_REFLECTION_PHRASES: tuple[str, ...] = (
    "the user's instruction",
    "the user says",
    "the problem says",
    "the problem states",
    "the task is",
    "the user wants me to",
    "i need to understand",
    "let me re-read",
    "as mentioned earlier",
    "as stated above",
    "since the problem",
)


def _detect_fallback_used(raw_output: str) -> bool:
    """Detect parser-fallback by inspecting raw_output.

    The planner returns a fallback knot dict (all 30°C uniform) whenever the
    model fails to emit a parseable `{"setpoints":[...]}` payload. Since the
    fallback dict is non-None, the trainer can't distinguish it from a real
    deliberate 30°C decision via `parsed_knot is None`. This function detects
    fallback by inspecting the raw_output:

      - Locate text after the LAST `</think>` (the answer region).
      - If `"setpoints"` keyword is missing there → model never emitted JSON
        → planner had to fall back → True.

    A genuine end-of-day "off AC" decision will write `"setpoints": [30.0, ...]`
    after `</think>`, so it correctly evaluates to False here.
    """
    if not isinstance(raw_output, str) or not raw_output:
        return True
    last_close = raw_output.rfind("</think>")
    answer_region = raw_output[last_close + len("</think>"):] if last_close >= 0 else raw_output
    return '"setpoints"' not in answer_region


def _compute_format_quality_penalty(raw_output: str, parsed_knot: dict | None) -> float:
    """Heuristic format-quality penalty in [0, ~2]. Higher = worse output.

    Signals:
      - no closing </think> tag when <think> was opened   → +0.5 (truncated)
      - parse failed to extract a valid setpoint dict      → +0.5
      - parser-fallback used (no JSON in output)           → +1.5 (dominant)
      - thinking contains meta-reflection phrases          → +0.1 each (cap +0.3)
      - thinking has a >=40-char substring repeated twice  → +0.2
      - duplicate PMV tool_call with identical args        → +0.1 per duplicate (cap +0.6)

    The fallback signal (+1.5) is intentionally large so that even after the
    `format_penalty_weight` multiplier, fallback knots receive a clear
    negative advantage on their model-generated tokens (typically the
    bloated tool_call sequence that exhausted the budget).

    Returns a penalty score (not yet scaled by the weight arg).
    """
    import re as _re, json as _json
    if not isinstance(raw_output, str) or not raw_output:
        return 1.5
    penalty = 0.0
    has_open = "<think>" in raw_output
    has_close = "</think>" in raw_output
    if has_open and not has_close:
        penalty += 0.5
    if parsed_knot is None:
        penalty += 0.5
    if _detect_fallback_used(raw_output):
        penalty += 1.5
    think_text = ""
    if has_open:
        m = _re.search(r"<think>(.*?)(</think>|$)", raw_output, flags=_re.DOTALL)
        if m:
            think_text = m.group(1).lower()
    else:
        think_text = raw_output[:4000].lower()
    # Meta-reflection: quoting the prompt back.
    meta_hits = sum(1 for ph in _META_REFLECTION_PHRASES if ph in think_text)
    penalty += min(0.3, 0.1 * meta_hits)
    # Gross repetition: 40+ char substring appearing twice.
    n = len(think_text)
    if n > 120:
        stride = 40
        seen: set[str] = set()
        repeated = False
        for i in range(0, n - stride, stride // 2):
            chunk = think_text[i : i + stride]
            if chunk in seen:
                repeated = True
                break
            seen.add(chunk)
        if repeated:
            penalty += 0.2
    # Duplicate tool_call penalty: same (temp, humidity, radiant) args called
    # more than once wastes compute and signals the model isn't learning from
    # the tool response. 0.1 per duplicate up to +0.6 (6 duplicates).
    tool_calls = _re.findall(r"<tool_call>(.*?)</tool_call>", raw_output, flags=_re.DOTALL)
    if tool_calls:
        seen_args: set[tuple[float, float, float]] = set()
        dup_count = 0
        for call_body in tool_calls:
            try:
                payload = _json.loads(call_body.strip())
                args = payload.get("arguments", {}) or {}
                key = (
                    round(float(args.get("temp", 0.0)), 2),
                    round(float(args.get("humidity", 0.0)), 1),
                    round(float(args.get("radiant", 0.0)), 2),
                )
            except Exception:
                continue
            if key in seen_args:
                dup_count += 1
            else:
                seen_args.add(key)
        if dup_count > 0:
            penalty += min(0.6, 0.1 * dup_count)
    # Cap raised to 3.5 to accommodate fallback (+1.5) + dup (+0.6) +
    # missing close (+0.5) + parse fail (+0.5) + repetition (+0.2) + meta (+0.3)
    return float(min(penalty, 3.5))


def _lora_target_modules(model) -> list[str]:
    """Decide LoRA target_modules from the model's actual linear layers.

    Qwen3-8B (full attention only): q_proj/k_proj/v_proj/o_proj.
    Qwen3.5-4B (hybrid linear + full attn): also has in_proj_qkv / in_proj_a /
    in_proj_b / in_proj_z / out_proj on linear-attention layers. Without
    targeting those, LoRA only covers ~25% of layers (the full-attn ones at
    every 4th position) and most of the model stays frozen.
    """
    import torch.nn as _nn
    seen: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, _nn.Linear):
            seen.add(name.rsplit(".", 1)[-1])
    # Always-include set; intersect with what the model actually has.
    candidates = {
        "q_proj", "k_proj", "v_proj", "o_proj",   # full attention
        "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",  # linear attention (Qwen3.5)
    }
    targets = sorted(candidates & seen)
    if not targets:
        # Fallback to the standard 4 in case nothing matches (model surgery).
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    return targets


def _expected_day_control_steps(bandit: MiamiGRPOBandit) -> int:
    """Return the number of eligible control steps in one Stage 2 day.

    Respects the control window: if it is narrower than the total span of
    BLOCK_DEFINITIONS, the steps outside the window don't get env_steps
    allocated (EP's seek_start skips them), so the "expected" total should
    only count blocks that fit inside the window.
    """
    cws = getattr(bandit, "control_window_start", None)
    cwe = getattr(bandit, "control_window_end", None)
    if cws is None or cwe is None:
        return sum(int(bandit._block_env_steps(i))
                   for i in range(len(bandit.BLOCK_DEFINITIONS)))
    duration_min = (cwe.hour * 60 + cwe.minute) - (cws.hour * 60 + cws.minute)
    step_minutes = getattr(bandit, "STEP_MINUTES", 10)
    return max(0, duration_min // int(step_minutes))


def _expected_day_knots(bandit: MiamiGRPOBandit) -> int:
    """Return the number of planned knots in one Stage 2 day.

    Same control-window gating as _expected_day_control_steps.
    """
    expected_steps = _expected_day_control_steps(bandit)
    knot_env_steps = getattr(bandit, "KNOT_ENV_STEPS", 3)
    return expected_steps // int(knot_env_steps)


def _stage2_skip_for_dataset_index(
    *,
    rows: list[dict[str, Any]],
    dataset_index: int,
    expected_day_control_steps: int,
) -> int:
    """Map a day-indexed dataset row to the current Stage 2 control schedule.

    Some Stage 2 datasets were collected with the old 06:30-19:00 window, so
    their stored skip_valid_steps advance by 25 half-hour steps per day.  Stage 2
    training uses 06:00-19:00, which is 26 half-hour steps.  Recomputing here
    keeps the dataset prompts/dates usable while aligning EnergyPlus rollouts
    with the actual Stage 2 block schedule.
    """
    if not rows:
        return 0
    base_skip = int(rows[0].get("skip_valid_steps", 0))
    return base_skip + int(dataset_index) * int(expected_day_control_steps)


def _serialize_knot_output(knot_plan: dict[str, Any], *, global_knot_index: int) -> dict[str, Any]:
    """Compact trajectory record for a planned knot."""
    raw_original = knot_plan.get("raw_output_original")
    raw_output = raw_original if raw_original is not None else knot_plan.get("raw_output")
    exploration_hint = knot_plan.get("exploration_mode_hint")
    exploration_hint_label = (
        "free"
        if "exploration_mode_hint" in knot_plan and exploration_hint is None
        else exploration_hint
    )
    setpoint_hint = knot_plan.get("setpoint_exploration_hint")
    setpoint_hint_label = (
        setpoint_hint.get("label")
        if isinstance(setpoint_hint, dict)
        else setpoint_hint
    )
    return {
        "global_knot_index": global_knot_index,
        "block_index": knot_plan.get("block_index"),
        "knot_index": knot_plan.get("knot_index"),
        "mode": knot_plan.get("mode"),
        "mode_source": knot_plan.get("mode_source"),
        "exploration_mode_hint": exploration_hint_label,
        "setpoint_exploration_hint": setpoint_hint_label,
        "setpoint_exploration_detail": setpoint_hint,
        "setpoints": _plainify(knot_plan.get("knot", {})),
        "mean_setpoint": knot_plan.get("mode_setpoint_mean"),
        "mode_setpoint_violation_c": knot_plan.get("mode_setpoint_violation_c", 0.0),
        "mode_setpoint_penalty": knot_plan.get("mode_setpoint_penalty", 0.0),
        "mode_setpoint_local_adv": knot_plan.get("mode_setpoint_local_adv", 0.0),
        "total_advantage": knot_plan.get("total_advantage"),
        "raw_output": raw_output,
    }


def _serialize_rollout_trajectory(
    rollout: dict[str, Any],
    *,
    rollout_index: int,
    day_reward: float,
) -> dict[str, Any]:
    """Compact trajectory record for one full-day rollout."""
    knot_plans = rollout.get("all_knot_plans", [])
    return {
        "rollout_index": rollout_index,
        "day_reward": round(float(day_reward), 4),
        "exploration_mode_hint": rollout.get("exploration_mode_hint") or "free",
        "setpoint_only": bool(rollout.get("setpoint_only", False)),
        "setpoint_macro_labels": rollout.get("setpoint_macro_labels", []),
        "setpoint_exploration_blocks": rollout.get("setpoint_exploration_blocks", 0),
        "modes": rollout.get("all_modes", []),
        "n_knots": len(knot_plans),
        "knot_outputs": [
            _serialize_knot_output(knot_plan, global_knot_index=knot_idx)
            for knot_idx, knot_plan in enumerate(knot_plans)
        ],
    }


def _build_trajectory_sample(
    *,
    step_index: int,
    target_date: str,
    rollout_results: list[dict[str, Any]],
    day_rewards: list[float],
    updated: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build a trajectory record with raw mode/setpoint outputs."""
    winner_idx = int(np.argmax(day_rewards)) if day_rewards else 0
    winner = rollout_results[winner_idx] if rollout_results else {}
    sample = {
        "step_index": step_index,
        "target_date": target_date,
        "updated": updated,
        "rollout_index": winner_idx,
        "day_reward": round(float(day_rewards[winner_idx]), 4) if day_rewards else 0.0,
        "modes": winner.get("all_modes", []),
        "n_knots": len(winner.get("all_knot_plans", [])),
        "winner_knot_outputs": [
            _serialize_knot_output(knot_plan, global_knot_index=knot_idx)
            for knot_idx, knot_plan in enumerate(winner.get("all_knot_plans", []))
        ],
        "rollout_trajectories": [
            _serialize_rollout_trajectory(
                rollout,
                rollout_index=rollout_idx,
                day_reward=day_rewards[rollout_idx],
            )
            for rollout_idx, rollout in enumerate(rollout_results)
        ],
        "block_summaries": [
            {
                "block": br["block_index"],
                "mode": br["mode"],
                "reward": round(br["relative_reward"], 4),
                "hvac_kwh": round(br["hvac_kwh"], 3),
                "pmv_viol": round(br["pmv_violation"], 3),
            }
            for br in winner.get("block_results", [])
        ],
    }
    if reason is not None:
        sample["reason"] = reason
    return sample


# --------------------------------------------------------------------------- #
# Full-day free rollout
# --------------------------------------------------------------------------- #

MODE_EXPLORATION_HINTS: tuple[str | None, ...] = ("cooling", None, "energy_saving")


def _mode_exploration_hint_for_rollout(
    *,
    step_index: int,
    rollout_idx: int,
    exploration_steps: int,
) -> str | None:
    """Return a soft mode exploration hint for early Stage 2 rollouts."""
    if exploration_steps <= 0 or step_index > exploration_steps:
        return None
    return MODE_EXPLORATION_HINTS[rollout_idx % len(MODE_EXPLORATION_HINTS)]


SETPOINT_MACRO_WEIGHTS: dict[str, float] = {
    "morning_precool": 0.20,
    "pv_comfort_cooling": 0.18,
    "cloud_rain_setback": 0.22,
    "late_setback": 0.22,
    "rare_high_setback": 0.06,
}


def _setpoint_exploration_probability(
    *,
    relative_step_index: int,
    warmup_steps: int,
    warmup_prob: float,
    late_prob: float,
) -> float:
    if warmup_steps <= 0:
        return max(float(late_prob), 0.0)
    if relative_step_index <= int(warmup_steps):
        return max(float(warmup_prob), 0.0)
    return max(float(late_prob), 0.0)


def _block_hour(block_start: Any, wallclock: Any) -> int:
    hour = getattr(block_start, "hour", None)
    if hour is not None:
        return int(hour)
    try:
        import pandas as pd

        return int(pd.Timestamp(wallclock).hour)
    except Exception:
        return 12


def _forecast_values(observation: dict[str, dict[str, Any]], key: str) -> list[float]:
    if not observation:
        return []
    first_zone = next(iter(observation.values()))
    values = first_zone.get(key, [])
    out: list[float] = []
    for value in list(values)[:6]:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            pass
    return out


def _zone_comfort_summary(
    observation: dict[str, dict[str, Any]],
) -> dict[str, float | bool]:
    occupied_hot = False
    occupied_count = 0
    max_occupied_pmv = -999.0
    max_occupied_temp = -999.0
    for zone_obs in observation.values():
        drybulb = _as_float(zone_obs.get("temperature_drybulb"))
        humidity = _as_float(zone_obs.get("humidity"))
        occupancy = _as_float(zone_obs.get("occupancy"))
        radiant = _as_float(zone_obs.get("temperature:radiant", drybulb))
        pmv = estimate_zone_pmv(
            temperature_drybulb=drybulb,
            temperature_radiant=radiant,
            humidity=humidity,
        )
        if occupancy > 0.1:
            occupied_count += 1
            max_occupied_pmv = max(max_occupied_pmv, float(pmv))
            max_occupied_temp = max(max_occupied_temp, float(drybulb))
            if pmv > 0.55 or drybulb > 26.5:
                occupied_hot = True
    return {
        "occupied_hot": occupied_hot,
        "occupied_count": float(occupied_count),
        "max_occupied_pmv": max_occupied_pmv if occupied_count else 0.0,
        "max_occupied_temp": max_occupied_temp if occupied_count else 0.0,
    }


def _eligible_setpoint_macros(
    *,
    block_start: Any,
    wallclock: Any,
    observation: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    hour = _block_hour(block_start, wallclock)
    temps = _forecast_values(observation, "forecast_temperature_6h")
    clouds = _forecast_values(observation, "forecast_cloudcover_6h")
    precip_probs = _forecast_values(observation, "forecast_precip_prob_6h")
    precip = _forecast_values(observation, "forecast_precip_6h")
    max_temp = max(temps) if temps else 31.0
    first_temp = temps[0] if temps else max_temp
    last_temp = temps[-1] if temps else first_temp
    mean_cloud = sum(clouds) / len(clouds) if clouds else 0.0
    max_precip_prob = max(precip_probs) if precip_probs else 0.0
    precip_sum = sum(precip) if precip else 0.0
    comfort = _zone_comfort_summary(observation)
    setback_allowed = not bool(comfort["occupied_hot"])
    storm_signal = mean_cloud >= 65.0 or max_precip_prob >= 50.0 or precip_sum >= 0.8
    sunny_signal = mean_cloud < 65.0 and max_precip_prob < 45.0
    reason = (
        f"hour={hour}, maxT6h={max_temp:.1f}C, meanCloud6h={mean_cloud:.0f}%, "
        f"maxPrecipProb6h={max_precip_prob:.0f}%, precip6h={precip_sum:.1f}mm"
    )

    macros: list[dict[str, str]] = []
    if 6 <= hour < 9 and max_temp >= 31.5:
        macros.append({
            "label": "morning_precool",
            "setpoint_band": "21.5-23.0 C for warm/occupied or solar-exposed zones; avoid below 21.0 C unless PMV is hot",
            "instruction": "Test modest morning pre-cooling before the Miami daytime heat ramp.",
            "reason": reason,
            "guardrail": "Do not overcool zones with PMV below -0.3.",
        })
    if 10 <= hour < 14 and max_temp >= 32.0 and sunny_signal:
        macros.append({
            "label": "pv_comfort_cooling",
            "setpoint_band": "22.0-23.5 C, using lower values only where PMV or solar exposure is high",
            "instruction": "Use high PV / high solar hours for comfort-safe cooling before the afternoon peak.",
            "reason": reason,
            "guardrail": "Do not force low setpoints in unoccupied cool zones.",
        })
    if 13 <= hour < 17 and storm_signal and setback_allowed:
        macros.append({
            "label": "cloud_rain_setback",
            "setpoint_band": "24.5-26.0 C where comfort allows; keep hotter occupied zones cooler",
            "instruction": "Test reduced cooling when clouds/rain lower solar load or the forecast cools down.",
            "reason": reason,
            "guardrail": "Abort setback if any occupied zone is already hot.",
        })
    if 17 <= hour < 19 and setback_allowed:
        macros.append({
            "label": "late_setback",
            "setpoint_band": "24.5-27.0 C, strongest in unoccupied zones",
            "instruction": "Test end-of-day cooling reduction after the main solar/PV peak.",
            "reason": reason,
            "guardrail": "Keep occupied hot zones within comfort limits.",
        })
    if setback_allowed and (max_temp < 31.0 or storm_signal or hour >= 17) and last_temp <= first_temp + 0.5:
        macros.append({
            "label": "rare_high_setback",
            "setpoint_band": "26.0-28.0 C only for clearly unoccupied or comfort-safe zones",
            "instruction": "Rarely test a stronger setback when weather and occupancy make it safe.",
            "reason": reason,
            "guardrail": "Never apply to occupied zones with PMV above 0.3.",
        })
    return macros


def _choose_setpoint_exploration_hint(
    *,
    block_start: Any,
    wallclock: Any,
    observation: dict[str, dict[str, Any]],
    probability: float,
    rng: random.Random,
    state: dict[str, Any],
) -> dict[str, str] | None:
    max_blocks = int(state.get("max_blocks", 0))
    if max_blocks <= 0 or int(state.get("used_blocks", 0)) >= max_blocks:
        return None
    if rng.random() >= max(float(probability), 0.0):
        return None
    macros = _eligible_setpoint_macros(
        block_start=block_start,
        wallclock=wallclock,
        observation=observation,
    )
    if not macros:
        return None
    weights = [SETPOINT_MACRO_WEIGHTS.get(str(m.get("label")), 0.1) for m in macros]
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for macro, weight in zip(macros, weights):
        acc += weight
        if pick <= acc:
            state["used_blocks"] = int(state.get("used_blocks", 0)) + 1
            return macro
    state["used_blocks"] = int(state.get("used_blocks", 0)) + 1
    return macros[-1]


class _Stage2FreeSamplePlannerProxy:
    """Free-mode proxy that can inject a rollout-level soft mode hint."""

    def __init__(
        self,
        real_planner: UnifiedBlockPlanner,
        *,
        exploration_mode_hint: str | None = None,
        setpoint_only: bool = False,
        setpoint_exploration_probability: float = 0.0,
        setpoint_exploration_state: dict[str, Any] | None = None,
        rng: random.Random | None = None,
    ):
        self._real = real_planner
        self.exploration_mode_hint = exploration_mode_hint
        self.setpoint_only = bool(setpoint_only)
        self.setpoint_exploration_probability = float(setpoint_exploration_probability)
        self.setpoint_exploration_state = setpoint_exploration_state if setpoint_exploration_state is not None else {}
        self.rng = rng if rng is not None else random.Random()
        self.chosen_mode: str | None = None
        self.setpoint_exploration_hint: dict[str, str] | None = None

    def plan_knot(
        self,
        *,
        block_index,
        knot_index,
        block_start,
        block_end,
        mode="balanced",
        observation=None,
        wallclock=None,
    ):
        if self.setpoint_only:
            if knot_index == 0:
                self.setpoint_exploration_hint = _choose_setpoint_exploration_hint(
                    block_start=block_start,
                    wallclock=wallclock,
                    observation=observation or {},
                    probability=self.setpoint_exploration_probability,
                    rng=self.rng,
                    state=self.setpoint_exploration_state,
                )
            result = self._real.plan_knot_setpoint_only(
                block_index=block_index,
                knot_index=knot_index,
                block_start=block_start,
                block_end=block_end,
                observation=observation,
                wallclock=wallclock,
                setpoint_exploration_hint=self.setpoint_exploration_hint,
            )
            label = (
                self.setpoint_exploration_hint.get("label")
                if self.setpoint_exploration_hint
                else "setpoint_only"
            )
            result["mode"] = label
            result["mode_source"] = "setpoint_macro" if self.setpoint_exploration_hint else "setpoint_only"
            return result

        if knot_index == 0:
            result = self._real.plan_knot_free(
                block_index=block_index,
                knot_index=knot_index,
                block_start=block_start,
                block_end=block_end,
                observation=observation,
                wallclock=wallclock,
                exploration_mode_hint=self.exploration_mode_hint,
            )
            self.chosen_mode = result.get("mode", "balanced")
            return result
        return self._real.plan_knot(
            block_index=block_index,
            knot_index=knot_index,
            block_start=block_start,
            block_end=block_end,
            mode=self.chosen_mode or "balanced",
            observation=observation,
            wallclock=wallclock,
        )

    def __getattr__(self, name):
        return getattr(self._real, name)

def _rollout_full_day_free(
    *,
    bandit: MiamiGRPOBandit,
    block_planner: UnifiedBlockPlanner,
    skip_valid_steps: int,
    baseline_action: dict[str, dict[str, float]],
    baseline_block_rewards: list[float],
    reward_scale: float,
    energy_weight: float,
    consistency_penalty_weight: float,
    mode_setpoint_penalty_weight: float,
    exploration_mode_hint: str | None,
    setpoint_only: bool,
    setpoint_exploration_probability: float,
    setpoint_exploration_max_blocks: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Run a complete day (13 blocks), all free-mode, collecting knot plans."""
    all_knot_plans: list[dict[str, Any]] = []
    block_results: list[dict[str, Any]] = []
    total_day_reward = 0.0
    all_modes: list[str] = []
    knot_to_block: list[int] = []
    setpoint_macro_labels: list[str] = []
    setpoint_exploration_state = {
        "used_blocks": 0,
        "max_blocks": int(setpoint_exploration_max_blocks),
    }

    block_planner.clear_block_results()
    block_planner.clear_knot_results()

    # SINGLE-EP FULL-DAY rollout (replaces per-block EP + replay pattern).
    # Opens ONE EnergyPlus simulation for 06:00-19:00, queries the planner at
    # each knot boundary (26 knots per day). Returns per-block rewards,
    # knot_plans, and per-block reward traces. ~3x faster than the 13-EP
    # pattern because we avoid repeated EP startups and replay overhead.
    proxy = _Stage2FreeSamplePlannerProxy(
        block_planner,
        exploration_mode_hint=exploration_mode_hint,
        setpoint_only=setpoint_only,
        setpoint_exploration_probability=setpoint_exploration_probability,
        setpoint_exploration_state=setpoint_exploration_state,
        rng=rng,
    )
    day_result = bandit._rollout_workday_with_knot_planner(
        skip_valid_steps=skip_valid_steps,
        planner=proxy,
        mode=("setpoint_only" if setpoint_only else "balanced"),
    )
    day_block_rewards = day_result.get("block_rewards", [0.0] * len(bandit.BLOCK_DEFINITIONS))
    day_knot_plans = day_result.get("knot_plans", [])
    day_block_reward_traces = day_result.get("block_reward_traces", [[] for _ in range(len(bandit.BLOCK_DEFINITIONS))])

    # Per-block post-processing: compute relative rewards, update planner state
    # for cross-rollout prev_block context, expand knots for replay in
    # subsequent rollouts, accumulate training metadata.
    import os as _os_blk
    _skip_s = _os_blk.environ.get("ASIM_SKIP_BLOCK_INDICES", "").strip()
    _skip_set = set()
    if _skip_s:
        for part in _skip_s.split(","):
            part = part.strip()
            if part.isdigit():
                _skip_set.add(int(part))

    knot_env_steps = bandit.KNOT_ENV_STEPS

    for block_index, (block_start, block_end) in enumerate(bandit.BLOCK_DEFINITIONS):
        block_reward = float(day_block_rewards[block_index])
        baseline_reward = float(baseline_block_rewards[block_index])
        relative_reward = reward_scale * (block_reward - baseline_reward)
        if int(block_index) not in _skip_set:
            total_day_reward += relative_reward

        brt = day_block_reward_traces[block_index] if block_index < len(day_block_reward_traces) else []

        # Knots for this block: day_knot_plans has block_index tagged on each.
        knot_plans = [kp for kp in day_knot_plans if int(kp.get("block_index", -1)) == block_index]

        # Mode label (consistent with prior per-block behavior). For setpoint_only,
        # use the last exploration-hint label seen during this block's knots; for
        # mode-based rollouts, chosen_mode is set by proxy at knot_index==0.
        if setpoint_only:
            chosen_mode = "setpoint_only"
            for kp in knot_plans:
                if kp.get("mode_source") == "setpoint_macro":
                    chosen_mode = str(kp.get("mode", "setpoint_only"))
                    break
            setpoint_macro_labels.append(chosen_mode)
        else:
            chosen_mode = "balanced"
            for kp in knot_plans:
                m = kp.get("mode")
                if m:
                    chosen_mode = str(m)
                    break
        all_modes.append(chosen_mode)

        all_knot_plans.extend(knot_plans)
        knot_to_block.extend([block_index] * len(knot_plans))

        hvac_kwh = sum(float(s.get("hvac_kwh", 0.0)) for s in brt)
        pmv_violation = sum(float(s.get("total_pmv_violation", 0.0)) for s in brt)

        # Record for NEXT rollout's cross-block context injection (the current
        # rollout already ran inside the single EP so this is for the planner
        # state seen by the next rollout on this rank).
        block_planner.record_block_result(
            block_index=block_index,
            block_start=str(block_start),
            block_end=str(block_end),
            winner_mode=chosen_mode,
            winner_reward=relative_reward,
            hvac_kwh=hvac_kwh,
            pmv_violation=pmv_violation,
        )

        # Per-knot in-context feedback: slice block_reward_trace into knots.
        for ki, kp in enumerate(knot_plans):
            sps = kp.get("knot") or kp.get("setpoints") or {}
            if not sps:
                continue
            s0 = ki * knot_env_steps
            s1 = s0 + knot_env_steps
            knot_brt = brt[s0:s1]
            if not knot_brt:
                continue
            k_hvac = sum(float(st.get("hvac_kwh", 0.0)) for st in knot_brt)
            per_zone_viol: dict[str, float] = {}
            per_zone_occ: dict[str, float] = {}
            for st in knot_brt:
                pz = st.get("per_zone_pmv_violation") or {}
                for z, v in pz.items():
                    cur = per_zone_viol.get(z, 0.0)
                    if abs(float(v)) > abs(cur):
                        per_zone_viol[z] = float(v)
            knot_wall = knot_brt[0].get("wallclock") or str(block_start)
            block_planner.record_knot_result(
                block_index=int(block_index),
                knot_index=int(kp.get("knot_index", ki)),
                wallclock=str(knot_wall),
                setpoints=dict(sps),
                hvac_kwh=float(k_hvac),
                pmv_violation_per_zone=per_zone_viol,
                occupancy_per_zone=per_zone_occ,
            )

        block_results.append({
            "block_index": block_index,
            "mode": chosen_mode,
            "block_reward": block_reward,
            "baseline_reward": baseline_reward,
            "relative_reward": relative_reward,
            "hvac_kwh": hvac_kwh,
            "pmv_violation": pmv_violation,
            "knot_count": len(knot_plans),
            "block_reward_trace": brt,
        })

    # Training-only shaping penalties.
    if setpoint_only:
        pmv_penalty = 0.0
        semantic_penalty = 0.0
        _annotate_mode_setpoint_semantic_violations(all_knot_plans, 0.0)
    else:
        pmv_penalty = _compute_mode_pmv_consistency_penalty(
            block_results, consistency_penalty_weight,
        )
        semantic_penalty = _annotate_mode_setpoint_semantic_violations(
            all_knot_plans, mode_setpoint_penalty_weight,
        )
    # PMV consistency remains a day-level shaping term.  Mode/setpoint semantic
    # contradictions are handled later as per-knot local advantages so the blame
    # stays on the offending action instead of the whole rollout.
    day_reward_penalty = pmv_penalty
    diagnostic_total_penalty = pmv_penalty + semantic_penalty

    return {
        "day_reward": total_day_reward - day_reward_penalty,
        "day_reward_raw": total_day_reward,
        "mode_setpoint_penalty": diagnostic_total_penalty,
        "day_reward_penalty": day_reward_penalty,
        "pmv_consistency_penalty": pmv_penalty,
        "mode_setpoint_semantic_penalty": semantic_penalty,
        "block_results": block_results,
        "all_knot_plans": all_knot_plans,
        "knot_to_block": knot_to_block,
        "all_modes": all_modes,
        "exploration_mode_hint": exploration_mode_hint,
        "setpoint_only": setpoint_only,
        "setpoint_macro_labels": setpoint_macro_labels,
        "setpoint_exploration_blocks": int(setpoint_exploration_state.get("used_blocks", 0)),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    # --- Distributed initialisation (torchrun-managed) ---
    import torch
    import torch.distributed as dist

    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_distributed:
        # NCCL default timeout is 10 min; gradient accumulation over 156 knots
        # on r=128 LoRA can exceed that. Give collectives 60 min headroom.
        from datetime import timedelta
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        # Map local_rank → physical GPU. When world_size > num_visible_GPUs
        # (e.g. 4 ranks on 2 GPUs for tighter rollout parallelism), wrap with %
        # so two ranks share each GPU. They both compute on the same SMs and
        # share VRAM but each owns its own model + KV cache.
        num_gpus = torch.cuda.device_count()
        physical_gpu = local_rank % max(1, num_gpus)
        torch.cuda.set_device(physical_gpu)
        device = torch.device(f"cuda:{physical_gpu}")
        print(
            f"[dist] rank={rank}/{world_size} local_rank={local_rank} "
            f"physical_gpu={physical_gpu} device={device}",
            flush=True,
        )
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device(args.device)
        print(f"[single-gpu] device={device}", flush=True)

    is_master = rank == 0

    if args.n_rollouts % world_size != 0:
        raise ValueError(
            f"--n-rollouts ({args.n_rollouts}) must be divisible by world_size ({world_size})"
        )
    n_per_rank = args.n_rollouts // world_size

    # Only rank 0 creates directories / opens files
    if is_master:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    trajectory_path = args.output_dir / "trajectory_samples.jsonl"
    phase_trace_path = args.output_dir / "phase_trace.jsonl"

    rows = _load_rows(args.dataset_path)
    if not rows:
        raise RuntimeError(f"No rows found in dataset: {args.dataset_path}")
    if is_master:
        print(f"Dataset: {len(rows)} training days from {args.dataset_path}", flush=True)

    # Seed differently per rank so each GPU's sampling RNG diverges — critical
    # for T>0 generation diversity across rollouts on different ranks.
    _set_seed(int(args.seed) + rank * 7919)

    # --- Environment timestep: 10-min env step, 10-min knot (1 env step per knot)
    # so the model decides every 10 min and immediately observes the result.
    # This avoids the setpoint→actual-temp transient gap that fooled the PMV tool
    # under the old 30-min knot scheme.
    MiamiGRPOBandit.STEP_MINUTES = 10
    MiamiGRPOBandit.KNOT_ENV_STEPS = 1

    # --- Resume from Stage 1 or Stage 2 checkpoint ---
    resume_checkpoint_dir = Path(args.resume_from).expanduser().resolve()
    if not resume_checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint_dir}")
    start_step_index = _resume_start_step(resume_checkpoint_dir)
    kl_reference_dir = (
        Path(args.kl_reference_from).expanduser().resolve()
        if args.kl_reference_from
        else resume_checkpoint_dir
    )
    if not kl_reference_dir.exists():
        raise FileNotFoundError(f"KL reference checkpoint not found: {kl_reference_dir}")

    # Optional Unsloth fast-path. Set ASIM_USE_UNSLOTH=1 to enable. Unsloth
    # auto-patches transformers internals so it MUST be imported before
    # AutoModelForCausalLM is touched.
    _use_unsloth = bool(int(os.environ.get("ASIM_USE_UNSLOTH", "0")))
    if _use_unsloth:
        from unsloth import FastLanguageModel  # noqa: F401  (patches transformers)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if _use_unsloth:
        # Unsloth's loader returns the patched model (it ignores the dtype kw
        # name; pass via dtype=). Tokenizer reuse: Unsloth's tokenizer matches
        # the one we just loaded.
        model, _unsloth_tok = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=int(args.max_output_tokens) + 4096,
            dtype=getattr(torch, args.torch_dtype),
            load_in_4bit=False,
            full_finetuning=False,
            trust_remote_code=True,
        )
        # Move to this rank's device — Unsloth loads onto cuda:0 by default.
        if device is not None:
            model.to(device)
    else:
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

    # --- Load PEFT adapter ---
    # Detect whether LoRA rank needs to be expanded from checkpoint.
    from peft import PeftModel, LoraConfig, get_peft_model
    import json as _json_mod

    ckpt_adapter_config = resume_checkpoint_dir / "adapter_config.json"
    ckpt_r = None
    if ckpt_adapter_config.exists():
        with open(ckpt_adapter_config) as _f:
            _cfg = _json_mod.load(_f)
            ckpt_r = _cfg.get("r")

    need_expand = (
        args.fresh_lora
        or (ckpt_r is not None and args.lora_r != ckpt_r)
    )

    if need_expand:
        # Create new LoRA with target rank, then zero-pad checkpoint weights in.
        target_r = args.lora_r
        target_alpha = args.lora_alpha
        if _use_unsloth:
            # Unsloth's get_peft_model returns the same patched model with
            # custom triton-accelerated LoRA forward kernels.
            model = FastLanguageModel.get_peft_model(
                model,
                r=target_r,
                lora_alpha=target_alpha,
                target_modules=_lora_target_modules(model),
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        else:
            lora_config = LoraConfig(
                r=target_r,
                lora_alpha=target_alpha,
                target_modules=_lora_target_modules(model),
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # Zero-pad old weights into the new larger adapter.
        # Skip when --fresh-lora: user wants a truly blank start.
        if (not args.fresh_lora) and ckpt_r is not None and ckpt_r < target_r:
            from safetensors.torch import load_file as _load_safetensors
            old_adapter_path = resume_checkpoint_dir / "adapter_model.safetensors"
            if not old_adapter_path.exists():
                old_adapter_path = resume_checkpoint_dir / "adapter_model.bin"
            if old_adapter_path.exists():
                if old_adapter_path.suffix == ".safetensors":
                    old_state = _load_safetensors(str(old_adapter_path), device="cpu")
                else:
                    old_state = torch.load(old_adapter_path, map_location="cpu")
                n_padded = 0
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    # Build candidates: original name + strip .default. + strip base_model.model.
                    candidates = [name]
                    # Strip ".default." (new PEFT adds it, old checkpoints don't have it)
                    if ".default." in name:
                        candidates.append(name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B."))
                    # Add ".default." (old PEFT doesn't have it, new model does)
                    for c in list(candidates):
                        candidates.append(c.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"))
                    # Strip "base_model.model." prefix
                    for c in list(candidates):
                        if c.startswith("base_model.model."):
                            candidates.append(c[len("base_model.model."):])
                    candidates = list(dict.fromkeys(candidates))  # dedupe
                    old_key = None
                    for c in candidates:
                        if c in old_state:
                            old_key = c
                            break
                    if old_key is None:
                        continue
                    old_tensor = old_state[old_key].to(dtype=param.dtype)
                    # lora_A: (r, hidden) → pad rows; lora_B: (hidden, r) → pad cols
                    if old_tensor.shape != param.data.shape:
                        padded = torch.zeros_like(param.data)
                        slices = tuple(
                            slice(0, min(o, n))
                            for o, n in zip(old_tensor.shape, param.data.shape)
                        )
                        padded[slices] = old_tensor[slices]
                        param.data.copy_(padded)
                    else:
                        param.data.copy_(old_tensor)
                    n_padded += 1
                print(
                    f"Expanded LoRA r={ckpt_r}→{target_r} alpha={target_alpha}: "
                    f"zero-padded {n_padded} params from checkpoint",
                    flush=True,
                )
            else:
                print(
                    f"Created fresh LoRA r={target_r} alpha={target_alpha} "
                    f"(no adapter weights found in checkpoint to pad from)",
                    flush=True,
                )
        else:
            print(
                f"Created fresh LoRA r={target_r} alpha={target_alpha}",
                flush=True,
            )
    else:
        # Normal resume: load adapter from checkpoint as-is.
        model = PeftModel.from_pretrained(model, str(resume_checkpoint_dir), is_trainable=True)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print(f"Loaded PEFT adapter from {resume_checkpoint_dir}", flush=True)

    # --- KL reference ---
    # When expanding LoRA, snapshot the current (padded) adapter as KL anchor.
    # When normal resume, load KL reference from the specified directory.
    if need_expand:
        sft_adapter_state = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        print(
            f"KL reference: using current adapter as anchor "
            f"({len(sft_adapter_state)} params)",
            flush=True,
        )
    else:
        sft_adapter_state = _load_adapter_snapshot_for_model(
            model=model,
            adapter_dir=kl_reference_dir,
            torch_module=torch,
        )
        print(
            f"Loaded {len(sft_adapter_state)} adapter params as KL reference from "
            f"{kl_reference_dir}",
            flush=True,
        )

    # --- Optimizer ---
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.learning_rate),
        )
        print("Using 8-bit AdamW (bitsandbytes)", flush=True)
    except ImportError:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(args.learning_rate),
        )
        print("Using standard AdamW (bitsandbytes not available)", flush=True)

    # Load optimizer state only when LoRA dimensions match checkpoint.
    if not need_expand:
        optimizer_path = resume_checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            try:
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
                print(f"Loaded optimizer state from {optimizer_path}", flush=True)
            except Exception as exc:
                print(f"Warning: failed to load optimizer state: {exc}", flush=True)
    else:
        print("LoRA expanded: skipping optimizer state load (incompatible dimensions)", flush=True)

    # --- Environment ---
    bandit = MiamiGRPOBandit(
        include_forecast=True,
        control_window_start="07:00",
        control_window_end="19:00",
        weekday_only=True,
        request_mode="step_action",
        building_path=args.building_idf,
        weather_path=args.weather_epw,
        fallback_setpoint_low_occ_c=(
            float(args.fallback_low_occ) if args.fallback_low_occ is not None else None
        ),
        fallback_setpoint_high_occ_c=(
            float(args.fallback_high_occ) if args.fallback_high_occ is not None else None
        ),
        fallback_occ_low_threshold=float(args.fallback_occ_low_thr),
        fallback_occ_high_threshold=float(args.fallback_occ_high_thr),
    )
    _validate_miami_forecast_binding(bandit)

    # --- Planner ---
    # Pick the backend that matches the model's tool_call format.
    # Qwen3-8B uses JSON; Qwen3.5-4B/9B uses XML. Selected via env var
    # ASIM_TOOL_FORMAT (json|xml) which is also read by the prompt builder.
    # Optional ASIM_USE_CACHED_BACKEND=1 swaps in a KV-cache-reuse generate
    # loop — same external behavior but ~3-5x faster on tool-call-heavy
    # rollouts (avoids the O(N²) re-encode every cycle).
    # vLLM-only path: this script ALWAYS uses VLLMQwen35Backend.
    # Architecture:
    #   - HF model on cuda:0 only (training: backward, optimizer)
    #   - vLLM TP=2 across both GPUs (rollout, sleep_mode for VRAM swap)
    #   - LoRA hot-swap each step: HF saves adapter → vLLM LoRARequest reloads
    print(f"[VLLM] Initializing vLLM engine (TP={args.vllm_tp}, gpu_mem={args.vllm_gpu_mem_util}, sleep_mode, LoRA enabled)...", flush=True)
    import time as _time_vllm_init
    _t_vllm = _time_vllm_init.time()
    from vllm import LLM as _vllm_LLM
    from vllm.lora.request import LoRARequest as _LoRARequest
    _vllm_engine = _vllm_LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=int(args.vllm_tp),
        dtype=args.torch_dtype if args.torch_dtype in ("bfloat16", "float16") else "bfloat16",
        gpu_memory_utilization=float(args.vllm_gpu_mem_util),
        enable_prefix_caching=True,
        max_model_len=12288,
        enforce_eager=False,
        enable_lora=True,
        max_lora_rank=int(args.lora_r),
        max_loras=1,
        enable_sleep_mode=True,
    )
    print(f"[VLLM]   loaded in {_time_vllm_init.time()-_t_vllm:.1f}s", flush=True)

    _lora_export_dir = args.output_dir / "vllm_lora_adapter"
    _lora_export_dir.mkdir(parents=True, exist_ok=True)
    _lora_request_seq = [0]

    def _save_and_swap_lora() -> "LoRARequest":
        model.save_pretrained(str(_lora_export_dir))
        _lora_request_seq[0] += 1
        return _LoRARequest(f"step_{_lora_request_seq[0]}", _lora_request_seq[0], str(_lora_export_dir))

    _initial_lora_req = _save_and_swap_lora()
    print(f"[VLLM]   initial LoRA exported to {_lora_export_dir}, request id={_lora_request_seq[0]}", flush=True)

    _vllm_engine.sleep()
    print("[VLLM]   slept; HF retains GPU. Wake on first rollout step.", flush=True)

    from llm_setpoint_planner_vllm import VLLMQwen35Backend
    backend = VLLMQwen35Backend(
        llm_engine=_vllm_engine,
        tokenizer=tokenizer,
        model_name=args.model_name_or_path,
        max_output_tokens=int(args.max_output_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
        lora_request=_initial_lora_req,
    )
    block_planner = UnifiedBlockPlanner(
        backend,
        constraints=PlannerConstraints(
            min_setpoint_c=20.0,
            max_setpoint_c=30.0,
            max_delta_per_step_c=2.0,
            # Set to 30°C (no cooling). Parse-fail during occupied periods
            # will spike PMV and incur large comfort penalty, making
            # malformed outputs STRICTLY worse than any successful generation.
            fallback_setpoint_c=30.0,
            quantization_c=0.1,
        ),
        zone_ids=bandit.zone_ids,
        max_generation_attempts=2,
    )
    baseline_action = {zone_id: {"thermostat": 23.0} for zone_id in bandit.zone_ids}

    # Load reflections from Stage 1 — skipped when --fresh-lora (starting from
    # scratch means we don't want the previous stage's learned language tokens
    # polluting the prompt, e.g. "energy_saving mode" leaking into thinking).
    reflections_path = resume_checkpoint_dir / "reflections.json"
    if reflections_path.exists() and not args.fresh_lora:
        try:
            block_planner.load_reflections(reflections_path)
            print(f"Loaded reflections from {reflections_path}", flush=True)
        except Exception as exc:
            print(f"Warning: failed to load reflections: {exc}", flush=True)
    elif args.fresh_lora:
        print(f"Skipped reflections load (--fresh-lora): {reflections_path}", flush=True)

    # --- WandB (rank 0 only) ---
    wandb_run = None
    if is_master and not args.no_wandb:
        try:
            import wandb
            wandb_run_name = args.wandb_name or args.output_dir.name
            wandb_run = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=wandb_run_name,
                job_type="train-stage2",
                config={
                    "stage": 2,
                    "learning_rate": float(args.learning_rate),
                    "reward_scale": float(args.reward_scale),
                    "kl_beta": float(args.kl_beta),
                    "kl_guard_threshold": float(args.kl_guard_threshold),
                    "n_rollouts": int(args.n_rollouts),
                    "consistency_penalty_weight": float(args.consistency_penalty_weight),
                    "mode_setpoint_penalty_weight": float(args.mode_setpoint_penalty_weight),
                    "mode_setpoint_local_adv_weight": float(args.mode_setpoint_local_adv_weight),
                    "mode_exploration_steps": int(args.mode_exploration_steps),
                    "setpoint_only": bool(args.setpoint_only),
                    "setpoint_exploration_steps": int(args.setpoint_exploration_steps),
                    "setpoint_exploration_prob": float(args.setpoint_exploration_prob),
                    "setpoint_exploration_late_prob": float(args.setpoint_exploration_late_prob),
                    "setpoint_exploration_max_blocks": int(args.setpoint_exploration_max_blocks),
                    "block_cross_adv_weight": float(args.block_cross_adv_weight),
                    "max_grad_norm": float(args.max_grad_norm),
                    "max_steps": int(args.max_steps),
                    "model_name_or_path": args.model_name_or_path,
                    "resume_from": str(resume_checkpoint_dir),
                    "kl_reference_from": str(kl_reference_dir),
                    "dataset_rows": len(rows),
                    "step_minutes": MiamiGRPOBandit.STEP_MINUTES,
                    "knot_env_steps": MiamiGRPOBandit.KNOT_ENV_STEPS,
                },
                reinit="finish_previous",
                settings=wandb.Settings(
                    x_disable_stats=True,
                    x_disable_machine_info=True,
                    console="off",              # disable console capture (big memory buffer)
                    _disable_meta=True,         # disable metadata collection
                    _disable_stats=True,
                ),
            )
            wandb_run.define_metric("step_index")
            for m in ("day_reward_mean", "day_reward_std", "day_reward_best",
                       "total_kl", "grad_norm", "consistency_penalty_mean",
                       "pmv_consistency_penalty_mean",
                       "mode_setpoint_semantic_penalty_mean",
                       "mode_setpoint_violation_knots_mean",
                       "mode_setpoint_local_adv_mean",
                       "setpoint_exploration_blocks_mean",
                       "day_reward_penalty_mean",
                       "total_penalty_mean",
                       "grad_contributions", "beats_history", "updated",
                       "has_signal", "all_negative", "episode"):
                wandb_run.define_metric(m, step_metric="step_index")
        except Exception:
            wandb_run = None

    # --- Training state ---
    started_at = time.time()
    energy_weight = float(os.environ.get("RL_W_ENERGY", "1.0"))
    history_best_day_reward: dict[int, float] = {}
    n_blocks = len(bandit.BLOCK_DEFINITIONS)
    expected_day_steps = _expected_day_control_steps(bandit)
    expected_day_knots = _expected_day_knots(bandit)
    print(
        f"Stage 2 schedule: {n_blocks} blocks, {expected_day_steps} control steps/day, "
        f"{expected_day_knots} knots/day",
        flush=True,
    )
    if is_master:
        print(
            f"Stage 2 loop will run steps {start_step_index}..{args.max_steps}",
            flush=True,
        )

    # File handles: rank 0 writes, rank 1 gets a no-op sink
    class _NullHandle:
        def write(self, *a, **kw): pass
        def flush(self): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    if is_master:
        _metrics_handle_ctx = metrics_path.open("w", encoding="utf-8")
        _trajectory_handle_ctx = trajectory_path.open("w", encoding="utf-8")
        _phase_handle_ctx = phase_trace_path.open("w", encoding="utf-8")
    else:
        _metrics_handle_ctx = _NullHandle()
        _trajectory_handle_ctx = _NullHandle()
        _phase_handle_ctx = _NullHandle()

    with _metrics_handle_ctx as metrics_handle, \
         _trajectory_handle_ctx as trajectory_handle, \
         _phase_handle_ctx as phase_handle:
        _write_phase(phase_handle, step_index=0, phase="stage2_init",
                     resume_from=str(resume_checkpoint_dir),
                     kl_reference_from=str(kl_reference_dir),
                     start_step_index=start_step_index,
                     n_rollouts=args.n_rollouts, max_steps=args.max_steps,
                     setpoint_only=bool(args.setpoint_only),
                     setpoint_exploration_steps=int(args.setpoint_exploration_steps),
                     setpoint_exploration_prob=float(args.setpoint_exploration_prob),
                     setpoint_exploration_late_prob=float(args.setpoint_exploration_late_prob),
                     setpoint_exploration_max_blocks=int(args.setpoint_exploration_max_blocks))

        # Episode-level reward accumulator: collect all day_reward_mean within current episode
        episode_rewards: list[float] = []

        for step_index in range(start_step_index, args.max_steps + 1):
            dataset_index = (step_index - 1) % len(rows)
            row = rows[dataset_index]
            dataset_skip_valid_steps = int(row["skip_valid_steps"])
            skip_valid_steps = _stage2_skip_for_dataset_index(
                rows=rows,
                dataset_index=dataset_index,
                expected_day_control_steps=expected_day_steps,
            )
            target_date = str(row.get("wallclock", f"step{step_index}")).split(" ")[0]
            if dataset_skip_valid_steps != skip_valid_steps:
                print(
                    f"  [SKIP REMAP] step={step_index} dataset_index={dataset_index} "
                    f"dataset_skip={dataset_skip_valid_steps} stage2_skip={skip_valid_steps} "
                    f"(expected {expected_day_steps} control steps/day)",
                    flush=True,
                )

            _write_phase(phase_handle, step_index=step_index, phase="step_start",
                         dataset_index=dataset_index,
                         dataset_skip_valid_steps=dataset_skip_valid_steps,
                         skip_valid_steps=skip_valid_steps,
                         expected_day_control_steps=expected_day_steps,
                         expected_day_knots=expected_day_knots,
                         target_date=target_date)

            step_start_time = time.time()

            # ---- 1. Baseline (cached) ----
            baseline_result = bandit._rollout_baseline_full_day_blocks(
                skip_valid_steps=skip_valid_steps,
                baseline_action=baseline_action,
            )
            baseline_block_rewards = baseline_result["block_rewards"]

            # ---- 2. Full-day rollouts (sequential or parallel) ----
            model.eval()

            def _run_one_rollout(rollout_idx: int) -> dict[str, Any]:
                # Per-rollout temperature sweep to force behavioral diversity
                # across the 6 rollouts even when hints are disabled. Range
                # 0.6..1.4 covers conservative → aggressive sampling. Base
                # temperature (args.temperature) is used as the center if
                # rollout_idx == 3 (middle of 6).
                _temp_schedule = [0.6, 0.8, 1.0, 1.1, 1.2, 1.4]
                _per_rollout_temp = _temp_schedule[rollout_idx % len(_temp_schedule)]
                backend.temperature = float(_per_rollout_temp)
                # Per-rollout seed: critical for vLLM backend. Without explicit
                # seed, vLLM's engine-level RNG was shared across rollouts and
                # gave identical outputs at different temperatures (GRPO advantage = 0).
                backend.seed = int(args.seed) * 1_000_003 + step_index * 101 + rollout_idx
                relative_step_index = step_index - start_step_index + 1
                exploration_hint = (
                    None
                    if args.setpoint_only
                    else _mode_exploration_hint_for_rollout(
                        step_index=step_index,
                        rollout_idx=rollout_idx,
                        exploration_steps=int(args.mode_exploration_steps),
                    )
                )
                setpoint_exploration_probability = (
                    _setpoint_exploration_probability(
                        relative_step_index=relative_step_index,
                        warmup_steps=int(args.setpoint_exploration_steps),
                        warmup_prob=float(args.setpoint_exploration_prob),
                        late_prob=float(args.setpoint_exploration_late_prob),
                    )
                    if args.setpoint_only
                    else 0.0
                )
                rollout_rng = random.Random(
                    int(args.seed) * 1_000_003 + step_index * 101 + rollout_idx
                )
                return _rollout_full_day_free(
                    bandit=bandit,
                    block_planner=block_planner,
                    skip_valid_steps=skip_valid_steps,
                    baseline_action=baseline_action,
                    baseline_block_rewards=baseline_block_rewards,
                    reward_scale=float(args.reward_scale),
                    energy_weight=energy_weight,
                    consistency_penalty_weight=float(args.consistency_penalty_weight),
                    mode_setpoint_penalty_weight=float(args.mode_setpoint_penalty_weight),
                    exploration_mode_hint=exploration_hint,
                    setpoint_only=bool(args.setpoint_only),
                    setpoint_exploration_probability=setpoint_exploration_probability,
                    setpoint_exploration_max_blocks=int(args.setpoint_exploration_max_blocks),
                    rng=rollout_rng,
                )

            # Distribute rollouts across ranks.
            # Rank R runs rollouts [R * n_per_rank, (R+1) * n_per_rank), global rollout_idx.
            local_rollout_indices = list(range(rank * n_per_rank, (rank + 1) * n_per_rank))

            # Wake vLLM and refresh LoRA with current HF weights before rollouts.
            _t_wake = time.time()
            _vllm_engine.wake_up()
            backend.lora_request = _save_and_swap_lora()
            print(
                f"  [VLLM] step={step_index} wake_up={time.time()-_t_wake:.2f}s "
                f"lora_id={_lora_request_seq[0]}", flush=True,
            )

            t_rollout_start = time.time()
            # Sequential rollouts within each rank: concurrent EP subprocesses contended
            # for timing windows and caused block-drop bugs (e.g. a block seeing 0 knots).
            # Cross-rank parallelism (GPU0 vs GPU1) still gives 2x speedup vs single-GPU
            # sequential. EP stays as 1 subprocess per rank at any time.
            local_results = []
            for idx in local_rollout_indices:
                _r = _run_one_rollout(idx)
                _t = time.time() - t_rollout_start
                # Real-time per-rollout summary printed by every rank as soon as
                # its local rollout finishes — gives visibility well before the
                # collective [UPDATE] line at the end of the step.
                print(
                    f"  [ROLLOUT_DONE] rank={rank} idx={idx} "
                    f"day_reward={_r['day_reward']:.4f} "
                    f"raw={_r['day_reward_raw']:.4f} "
                    f"pmv_pen={_r['pmv_consistency_penalty']:.4f} "
                    f"local_elapsed={_t:.1f}s",
                    flush=True,
                )
                local_results.append(_r)
            t_local_rollout = time.time() - t_rollout_start

            # Sleep vLLM to free VRAM for HF training step's backward pass.
            _t_sleep = time.time()
            _vllm_engine.sleep()
            print(f"  [VLLM] step={step_index} sleep={time.time()-_t_sleep:.2f}s", flush=True)

            if is_distributed:
                # all_gather_object: every rank ends up with results from every rank.
                gathered: list[list[dict[str, Any]]] = [None] * world_size
                dist.all_gather_object(gathered, local_results)
                # Flatten in rank order so global rollout_idx mapping is preserved.
                rollout_results = [r for rank_results in gathered for r in rank_results]
                t_gather = time.time() - t_rollout_start
                if is_master:
                    print(
                        f"  [PARALLEL] step={step_index} n_rollouts={args.n_rollouts} "
                        f"world_size={world_size} local={t_local_rollout:.1f}s "
                        f"gather_total={t_gather:.1f}s",
                        flush=True,
                    )
            else:
                rollout_results = local_results
                if is_master:
                    print(
                        f"  [LOCAL] step={step_index} n_rollouts={args.n_rollouts} "
                        f"time={t_local_rollout:.1f}s",
                        flush=True,
                    )

            for rollout_idx, result in enumerate(rollout_results):
                actual_knots = len(result.get("all_knot_plans", []))
                if actual_knots != expected_day_knots:
                    block_knot_counts = [
                        int(br.get("knot_count", 0))
                        for br in result.get("block_results", [])
                    ]
                    raise RuntimeError(
                        f"Stage 2 rollout produced {actual_knots} knots, expected "
                        f"{expected_day_knots}. This usually means skip_valid_steps "
                        f"does not point to the start of a full control day. "
                        f"step={step_index}, rollout={rollout_idx}, "
                        f"dataset_index={dataset_index}, dataset_skip={dataset_skip_valid_steps}, "
                        f"stage2_skip={skip_valid_steps}, block_knot_counts={block_knot_counts}"
                    )
                hint_label = result.get("exploration_mode_hint") or "free"
                if is_master:
                    print(f"  [ROLLOUT] step={step_index} rollout={rollout_idx} "
                          f"day_reward={result['day_reward']:.4f} "
                          f"(raw={result['day_reward_raw']:.4f} "
                          f"pmv_day_penalty={result['pmv_consistency_penalty']:.4f} "
                          f"semantic_diag={result['mode_setpoint_semantic_penalty']:.4f}) "
                          f"hint={hint_label} "
                          f"setpoint_macros={result.get('setpoint_macro_labels', [])} "
                          f"modes={result['all_modes']}", flush=True)
                    # Per-knot action + reward dump (gated by ASIM_DEBUG_KNOTS).
                    # Useful to see what the model actually decided per 30-min
                    # block and how EP rewarded it, without waiting for the
                    # end-of-step trajectory JSON.
                    import os as _os_knots
                    if bool(int(_os_knots.environ.get("ASIM_DEBUG_KNOTS", "0"))):
                        all_knots = result.get("all_knot_plans", [])
                        block_results = result.get("block_results", [])
                        # Flatten per-env-step rewards from all blocks in order.
                        step_rewards: list[float] = []
                        for br in block_results:
                            brt = br.get("block_reward_trace", [])
                            for st in brt:
                                step_rewards.append(float(st.get("reward", 0.0)))
                        for ki, kp in enumerate(all_knots):
                            sp_dict = kp.get("knot") or kp.get("setpoints") or {}
                            sp_mean = (sum(sp_dict.values()) / len(sp_dict)) if sp_dict else float("nan")
                            sp_min = min(sp_dict.values()) if sp_dict else float("nan")
                            sp_max = max(sp_dict.values()) if sp_dict else float("nan")
                            knot_r = step_rewards[ki] if ki < len(step_rewards) else float("nan")
                            mode_k = kp.get("mode", "?")
                            bidx = kp.get("block_index", "?")
                            raw_ok = bool(kp.get("knot") or kp.get("setpoints"))
                            print(
                                f"    [KNOT_DBG] r={rollout_idx} k={ki:02d} blk={bidx} mode={mode_k:>14} "
                                f"sp=[{sp_min:.1f}-{sp_max:.1f} μ={sp_mean:.2f}] r={knot_r:+.4f} "
                                f"parse_ok={int(raw_ok)}",
                                flush=True,
                            )

            # ---- 3. Day-level advantages ----
            day_rewards = [r["day_reward"] for r in rollout_results]
            day_reward_penalties = [r["day_reward_penalty"] for r in rollout_results]
            total_penalties = [r["mode_setpoint_penalty"] for r in rollout_results]
            pmv_penalties = [r["pmv_consistency_penalty"] for r in rollout_results]
            semantic_penalties = [
                r["mode_setpoint_semantic_penalty"] for r in rollout_results
            ]
            semantic_violation_counts = [
                sum(
                    1
                    for kp in r.get("all_knot_plans", [])
                    if float(kp.get("mode_setpoint_violation_c", 0.0)) > 1e-9
                )
                for r in rollout_results
            ]
            semantic_violation_values = [
                float(kp.get("mode_setpoint_violation_c", 0.0))
                for r in rollout_results
                for kp in r.get("all_knot_plans", [])
            ]
            exploration_hints = [
                r.get("exploration_mode_hint") or "free" for r in rollout_results
            ]
            setpoint_macro_labels = [
                r.get("setpoint_macro_labels", []) for r in rollout_results
            ]
            setpoint_exploration_blocks = [
                int(r.get("setpoint_exploration_blocks", 0)) for r in rollout_results
            ]
            advantages = _compute_day_advantages(day_rewards)

            # ---- 4. Update gating ----
            best_reward = max(day_rewards)
            prev_best = history_best_day_reward.get(skip_valid_steps, float("-inf"))
            beats_history = best_reward > prev_best
            all_negative = all(r < 0 for r in day_rewards)
            has_signal = max(abs(a) for a in advantages) > 1e-8
            # History-best remains a diagnostic metric only. Stage 2 relies on
            # cross-rollout day-level advantages, so below-history samples still
            # carry useful reinforce/suppress signal.
            # Previous relaxed gate:
            # do_update = has_signal and (beats_history or all_negative)
            do_update = has_signal

            if beats_history:
                history_best_day_reward[skip_valid_steps] = best_reward

            _write_phase(phase_handle, step_index=step_index, phase="advantages_computed",
                         day_rewards=[round(r, 4) for r in day_rewards],
                         advantages=[round(a, 4) for a in advantages],
                         beats_history=beats_history,
                         all_negative=all_negative,
                         do_update=do_update)

            if not do_update:
                if is_master:
                    print(f"  [SKIP] step={step_index} no update "
                          f"(has_signal={has_signal} beats_hist={beats_history} all_neg={all_negative})",
                          flush=True)
                trajectory_handle.write(json.dumps(_build_trajectory_sample(
                    step_index=step_index,
                    target_date=target_date,
                    rollout_results=rollout_results,
                    day_rewards=day_rewards,
                    updated=False,
                    reason="no_signal" if not has_signal else "no_update",
                ), ensure_ascii=False) + "\n")
                trajectory_handle.flush()
                # Log metrics even when skipping
                step_metric = {
                    "step_index": step_index,
                    "dataset_index": dataset_index,
                    "dataset_skip_valid_steps": dataset_skip_valid_steps,
                    "skip_valid_steps": skip_valid_steps,
                    "target_date": target_date,
                    "day_rewards": [round(r, 4) for r in day_rewards],
                    "day_reward_mean": round(float(np.mean(day_rewards)), 4),
                    "day_reward_std": round(float(np.std(day_rewards)), 4),
                    "day_reward_penalties": [round(p, 4) for p in day_reward_penalties],
                    "total_penalties": [round(p, 4) for p in total_penalties],
                    "pmv_consistency_penalties": [round(p, 4) for p in pmv_penalties],
                    "mode_setpoint_semantic_penalties": [
                        round(p, 4) for p in semantic_penalties
                    ],
                    "mode_setpoint_violation_knots": semantic_violation_counts,
                    "mode_setpoint_violation_c_max": round(
                        max(semantic_violation_values, default=0.0), 4,
                    ),
                    "exploration_hints": exploration_hints,
                    "setpoint_only": bool(args.setpoint_only),
                    "setpoint_macro_labels": setpoint_macro_labels,
                    "setpoint_exploration_blocks": setpoint_exploration_blocks,
                    "updated": False,
                    "reason": "no_signal" if not has_signal else "no_update",
                }
                metrics_handle.write(json.dumps(step_metric, ensure_ascii=False) + "\n")
                metrics_handle.flush()
                continue

            # ---- 5. Step-level (knot-level) advantages ----
            step_advs = _compute_step_level_advantages(
                rollout_results, baseline_block_rewards,
                float(args.reward_scale), n_blocks,
                gamma=args.gamma,
                advantage_mode=str(args.advantage_mode),
            )
            # Keep block cross advs for diagnostic logging only
            block_cross_advs = _compute_block_cross_advantages(rollout_results, n_blocks)

            # ---- 6. Normalize + debug validation ----
            _dbg_total = 0; _dbg_valid_pre = 0; _dbg_valid_post = 0; _dbg_none = 0
            with torch.no_grad():
                for rollout in rollout_results:
                    for knot_plan in rollout["all_knot_plans"]:
                        _dbg_total += 1
                        raw = knot_plan.get("raw_output", "")
                        if raw is None:
                            _dbg_none += 1; continue
                        if isinstance(raw, str) and _validate_setpoint_output(raw):
                            _dbg_valid_pre += 1
                        if isinstance(raw, str):
                            knot_plan.setdefault("raw_output_original", raw)
                            knot_plan["raw_output"] = _normalize_raw_output(raw)
                        norm = knot_plan.get("raw_output", "")
                        if isinstance(norm, str) and _validate_setpoint_output(norm):
                            _dbg_valid_post += 1
            if is_master:
                print(f"  [VALIDATE] step={step_index} total={_dbg_total} none={_dbg_none} "
                      f"valid_pre={_dbg_valid_pre} valid_post={_dbg_valid_post}", flush=True)

            # ---- 7-9. Gradient + KL guard + Optimizer step (rank 0 only) ----
            # Rank 1 waits; both ranks synchronise on the kl_guard decision and
            # on the adapter broadcast at the end of the step.
            total_kl = 0.0
            raw_grad_norm_val = 0.0
            grad_norm_val = 0.0
            grad_contributions = 0
            local_adv_values: list[float] = []
            kl_guard_skip = False

            if is_master:
                optimizer.zero_grad(set_to_none=True)
                model.train()
                for rollout_idx, rollout in enumerate(rollout_results):
                    day_adv = advantages[rollout_idx]
                    rollout_step_advs = step_advs[rollout_idx] if rollout_idx < len(step_advs) else []
                    knot_plans = rollout["all_knot_plans"]
                    knot_to_block = rollout["knot_to_block"]
                    n_knots = len(knot_plans)
                    divisor = args.n_rollouts * max(n_knots, 1)

                    n_skipped_truncated = 0
                    n_format_penalised = 0
                    for knot_idx, knot_plan in enumerate(knot_plans):
                        block_i = knot_to_block[knot_idx] if knot_idx < len(knot_to_block) else 0
                        block_adv = block_cross_advs[block_i][rollout_idx] if block_i < len(block_cross_advs) else 0.0
                        step_adv = rollout_step_advs[knot_idx] if knot_idx < len(rollout_step_advs) else 0.0

                        semantic_violation_c = float(
                            knot_plan.get("mode_setpoint_violation_c", 0.0)
                        )
                        local_adv = (
                            -float(args.mode_setpoint_local_adv_weight)
                            * semantic_violation_c
                        )

                        # Format-quality inspection (applies to any advantage mode).
                        raw_output = knot_plan.get("raw_output", "")
                        parsed_knot = knot_plan.get("knot") or knot_plan.get("setpoints")
                        fmt_penalty_score = _compute_format_quality_penalty(raw_output, parsed_knot)
                        fmt_penalty = -float(args.format_penalty_weight) * float(fmt_penalty_score)
                        if fmt_penalty_score >= 0.5:
                            n_format_penalised += 1

                        # Only skip the most pathological outputs (essentially
                        # empty / fewer than 200 chars — a normal valid response
                        # with `<think>...</think> {"setpoints":[...]}` is
                        # ~80-200 chars depending on whitespace, and a fallback
                        # response is 5000+ chars). For "ran out of tool
                        # budget" cases — which generate 4k-8k tokens of real
                        # reasoning — keep the gradient so the format-penalty
                        # signal (incl. the +1.5 fallback bump) actively pushes
                        # the model away from over-spending its tool budget.
                        #
                        # Earlier bug (2026-04-27): used `.split()` whitespace
                        # token count which incorrectly flagged 70+/72 normal
                        # short JSON-only responses as garbage and skipped
                        # gradient → contributions=0, no learning. Reverted to
                        # plain char count.
                        raw_chars = len(str(raw_output)) if isinstance(raw_output, str) else 0
                        is_extreme_garbage = (raw_chars < 50)
                        if bool(args.filter_truncated) and is_extreme_garbage:
                            knot_plan["step_advantage"] = step_adv
                            knot_plan["day_advantage"] = day_adv
                            knot_plan["block_cross_advantage"] = block_adv
                            knot_plan["mode_setpoint_local_adv"] = local_adv
                            knot_plan["format_penalty_score"] = fmt_penalty_score
                            knot_plan["format_penalty_adv"] = fmt_penalty
                            knot_plan["total_advantage"] = 0.0
                            knot_plan["gradient_skipped_reason"] = "extreme_garbage_lt50_chars"
                            n_skipped_truncated += 1
                            continue

                        total_adv = step_adv + local_adv + fmt_penalty
                        knot_plan["step_advantage"] = step_adv
                        knot_plan["day_advantage"] = day_adv
                        knot_plan["block_cross_advantage"] = block_adv
                        knot_plan["mode_setpoint_local_adv"] = local_adv
                        knot_plan["format_penalty_score"] = fmt_penalty_score
                        knot_plan["format_penalty_adv"] = fmt_penalty
                        knot_plan["total_advantage"] = total_adv
                        local_adv_values.append(local_adv)

                        logprob, tok_count, kl_val = _accumulate_block_gradient(
                            model=model,
                            tokenizer=tokenizer,
                            backend=backend,
                            block_plan=knot_plan,
                            advantage=total_adv,
                            block_divisor=divisor,
                            kl_beta=float(args.kl_beta),
                            sft_adapter_state=sft_adapter_state,
                        )
                        total_kl += kl_val
                        if tok_count > 0:
                            grad_contributions += 1
                    if (n_skipped_truncated > 0 or n_format_penalised > 0) and is_master:
                        print(f"  [FILTER] rollout={rollout_idx} skipped_truncated={n_skipped_truncated} "
                              f"format_penalised={n_format_penalised} of {len(knot_plans)} knots", flush=True)

                # KL guard on rank 0
                if total_kl > float(args.kl_guard_threshold):
                    optimizer.zero_grad(set_to_none=True)
                    model.eval()
                    kl_guard_skip = True
                    print(f"  [KL GUARD] step={step_index} total_kl={total_kl:.1f} > {args.kl_guard_threshold} → skip",
                          flush=True)
                    _write_phase(phase_handle, step_index=step_index, phase="kl_guard_skip",
                                 total_kl=round(total_kl, 2))
                    trajectory_handle.write(json.dumps(_build_trajectory_sample(
                        step_index=step_index,
                        target_date=target_date,
                        rollout_results=rollout_results,
                        day_rewards=day_rewards,
                        updated=False,
                        reason="kl_guard_skip",
                    ), ensure_ascii=False) + "\n")
                    trajectory_handle.flush()
                else:
                    # Optimizer step on rank 0 only
                    raw_grad_norm_val = _grad_norm(model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.max_grad_norm))
                    grad_norm_val = _grad_norm(model)
                    optimizer.step()
                    model.eval()

            # Broadcast kl_guard decision so all ranks agree on skipping
            if is_distributed:
                obj = [kl_guard_skip]
                dist.broadcast_object_list(obj, src=0)
                kl_guard_skip = obj[0]

            if kl_guard_skip:
                # Both ranks still need to broadcast the (unchanged) adapter
                # for consistency on the next step. But since rank 0 didn't
                # update, the weights are unchanged — broadcast is a no-op-equiv.
                if is_distributed:
                    for _, param in model.named_parameters():
                        if param.requires_grad:
                            dist.broadcast(param.data, src=0)
                continue

            # Broadcast updated adapter weights from rank 0 to all ranks
            if is_distributed:
                _t_bcast = time.time()
                for _, param in model.named_parameters():
                    if param.requires_grad:
                        dist.broadcast(param.data, src=0)
                if is_master:
                    print(f"  [BROADCAST] adapter sync {time.time() - _t_bcast:.2f}s", flush=True)

            step_elapsed = time.time() - step_start_time
            _sa_flat = [a for row in step_advs for a in row] if step_advs else [0.0]
            if is_master:
                print(f"  [UPDATE] step={step_index} day_rewards={[round(r,3) for r in day_rewards]} "
                      f"kl={total_kl:.1f} raw_grad_norm={raw_grad_norm_val:.4f} grad_norm={grad_norm_val:.4f} "
                      f"contributions={grad_contributions} "
                      f"step_adv_std={np.std(_sa_flat):.3f} step_adv_max={np.max(np.abs(_sa_flat)):.3f} "
                      f"gamma={args.gamma} time={step_elapsed:.1f}s", flush=True)

            _write_phase(phase_handle, step_index=step_index, phase="optimizer_step",
                         total_kl=round(total_kl, 2),
                         raw_grad_norm=round(raw_grad_norm_val, 4),
                         grad_norm=round(grad_norm_val, 4),
                         grad_contributions=grad_contributions)

            # ---- 10. Metrics ----
            all_modes_flat = []
            for r in rollout_results:
                all_modes_flat.extend(r["all_modes"])

            step_metric = {
                "step_index": step_index,
                "dataset_index": dataset_index,
                "dataset_skip_valid_steps": dataset_skip_valid_steps,
                "skip_valid_steps": skip_valid_steps,
                "target_date": target_date,
                "day_rewards": [round(r, 4) for r in day_rewards],
                "day_reward_mean": round(float(np.mean(day_rewards)), 4),
                "day_reward_std": round(float(np.std(day_rewards)), 4),
                "advantages": [round(a, 4) for a in advantages],
                "total_kl": round(total_kl, 2),
                "raw_grad_norm": round(raw_grad_norm_val, 4),
                "grad_norm": round(grad_norm_val, 4),
                "grad_contributions": grad_contributions,
                "day_reward_penalties": [round(p, 4) for p in day_reward_penalties],
                "total_penalties": [round(p, 4) for p in total_penalties],
                "consistency_penalties": [round(p, 4) for p in pmv_penalties],
                "pmv_consistency_penalties": [round(p, 4) for p in pmv_penalties],
                "mode_setpoint_semantic_penalties": [
                    round(p, 4) for p in semantic_penalties
                ],
                "mode_setpoint_violation_knots": semantic_violation_counts,
                "mode_setpoint_violation_c_max": round(
                    max(semantic_violation_values, default=0.0), 4,
                ),
                "mode_setpoint_local_adv_sum": round(float(sum(local_adv_values)), 4),
                "mode_setpoint_local_adv_mean": round(
                    float(np.mean(local_adv_values)) if local_adv_values else 0.0, 4
                ),
                "exploration_hints": exploration_hints,
                "setpoint_only": bool(args.setpoint_only),
                "setpoint_macro_labels": setpoint_macro_labels,
                "setpoint_exploration_blocks": setpoint_exploration_blocks,
                "beats_history": beats_history,
                "all_negative": all_negative,
                "updated": True,
                "modes_per_rollout": [r["all_modes"] for r in rollout_results],
                "block_cross_advs_summary": [
                    [round(a, 3) for a in block_cross_advs[bi]]
                    for bi in range(min(n_blocks, len(block_cross_advs)))
                ],
                "gamma": args.gamma,
                "step_advantage_mean": round(
                    float(np.mean([abs(a) for row in step_advs for a in row])), 4
                ) if step_advs else 0.0,
                "step_advantage_std": round(
                    float(np.std([a for row in step_advs for a in row])), 4
                ) if step_advs else 0.0,
                "step_advantage_max_abs": round(
                    float(np.max([abs(a) for row in step_advs for a in row])), 4
                ) if step_advs else 0.0,
                "elapsed_seconds": round(step_elapsed, 1),
            }
            metrics_handle.write(json.dumps(step_metric, ensure_ascii=False) + "\n")
            metrics_handle.flush()

            # Trajectory sample: log raw mode/setpoint outputs for every rollout.
            trajectory_sample = _build_trajectory_sample(
                step_index=step_index,
                target_date=target_date,
                rollout_results=rollout_results,
                day_rewards=day_rewards,
                updated=True,
            )
            trajectory_handle.write(json.dumps(trajectory_sample, ensure_ascii=False) + "\n")
            trajectory_handle.flush()

            # WandB
            if wandb_run is not None:
                try:
                    wandb_run.log({
                        "step_index": step_index,
                        "day_reward_mean": float(np.mean(day_rewards)),
                        "day_reward_std": float(np.std(day_rewards)),
                        "day_reward_best": float(best_reward),
                        "total_kl": total_kl,
                        "grad_norm": grad_norm_val,
                        "grad_contributions": grad_contributions,
                        "consistency_penalty_mean": float(np.mean(pmv_penalties)),
                        "pmv_consistency_penalty_mean": float(np.mean(pmv_penalties)),
                        "mode_setpoint_semantic_penalty_mean": float(np.mean(semantic_penalties)),
                        "mode_setpoint_violation_knots_mean": float(np.mean(semantic_violation_counts)),
                        "mode_setpoint_local_adv_mean": float(np.mean(local_adv_values)) if local_adv_values else 0.0,
                        "setpoint_exploration_blocks_mean": float(np.mean(setpoint_exploration_blocks)) if setpoint_exploration_blocks else 0.0,
                        "day_reward_penalty_mean": float(np.mean(day_reward_penalties)),
                        "total_penalty_mean": float(np.mean(total_penalties)),
                        "beats_history": float(beats_history),
                        "updated": 1.0,
                        "has_signal": float(has_signal),
                        "all_negative": float(all_negative),
                        "episode": (step_index - 1) // len(rows) + 1,
                        "step_advantage_mean": float(np.mean([abs(a) for row in step_advs for a in row])) if step_advs else 0.0,
                        "step_advantage_std": float(np.std([a for row in step_advs for a in row])) if step_advs else 0.0,
                        "step_advantage_max_abs": float(np.max([abs(a) for row in step_advs for a in row])) if step_advs else 0.0,
                        "gamma": args.gamma,
                    })
                except Exception:
                    pass

            # ---- 10b. Episode-level reward accumulator ----
            episode_rewards.append(float(np.mean(day_rewards)))

            # ---- 11. Checkpoint (rank 0 only) ----
            if is_master:
                if step_index % args.save_steps == 0:
                    ckpt_dir = args.output_dir / f"checkpoint-{step_index}"
                    _save_training_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        block_planner=block_planner,
                        checkpoint_dir=ckpt_dir,
                        step_index=step_index,
                        phase_name="stage2",
                        include_tokenizer=(step_index % (args.save_steps * 4) == 0),
                        checkpoint_kind="full",
                    )
                    print(f"  [SAVE] checkpoint-{step_index}", flush=True)
                elif args.cache_steps > 0 and step_index % args.cache_steps == 0:
                    cache_dir = args.output_dir / f"cache-checkpoint-{step_index}"
                    _save_training_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        block_planner=block_planner,
                        checkpoint_dir=cache_dir,
                        step_index=step_index,
                        phase_name="stage2",
                        include_tokenizer=False,
                        checkpoint_kind="cache",
                    )

            # ---- 12. Episode boundary: decay history best ----
            episode = (step_index - 1) // len(rows) + 1
            day_in_ep = dataset_index
            if step_index % len(rows) == 0:
                # End of episode: compute episode-level reward stats
                ep_mean = float(np.mean(episode_rewards)) if episode_rewards else 0.0
                ep_std = float(np.std(episode_rewards)) if episode_rewards else 0.0
                ep_min = float(np.min(episode_rewards)) if episode_rewards else 0.0
                ep_max = float(np.max(episode_rewards)) if episode_rewards else 0.0
                ep_positive = sum(1 for r in episode_rewards if r > 0)
                if is_master:
                    print(
                        f"  [EPISODE] ep={episode} steps={len(episode_rewards)} "
                        f"mean={ep_mean:+.3f} std={ep_std:.3f} "
                        f"min={ep_min:+.3f} max={ep_max:+.3f} "
                        f"positive={ep_positive}/{len(episode_rewards)}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        try:
                            wandb_run.log({
                                "episode": episode,
                                "episode/day_reward_mean": ep_mean,
                                "episode/day_reward_std": ep_std,
                                "episode/day_reward_min": ep_min,
                                "episode/day_reward_max": ep_max,
                                "episode/positive_step_ratio": ep_positive / max(len(episode_rewards), 1),
                                "episode/n_steps": len(episode_rewards),
                            })
                        except Exception:
                            pass
                    _write_phase(
                        phase_handle, step_index=step_index, phase="episode_end",
                        episode=episode,
                        day_reward_mean=round(ep_mean, 4),
                        day_reward_std=round(ep_std, 4),
                        day_reward_min=round(ep_min, 4),
                        day_reward_max=round(ep_max, 4),
                        n_steps=len(episode_rewards),
                    )
                episode_rewards = []  # reset for next episode

                # End of episode: clean cache checkpoints (rank 0 only)
                if is_master:
                    episode_start_step = step_index - len(rows) + 1
                    if not args.keep_cache_checkpoints:
                        for s in range(episode_start_step, step_index + 1):
                            cache_dir = args.output_dir / f"cache-checkpoint-{s}"
                            if cache_dir.exists():
                                try:
                                    shutil.rmtree(cache_dir)
                                except Exception:
                                    pass

            # Decay at start of each new episode (except first)
            if day_in_ep == 0 and episode > 1:
                for k in list(history_best_day_reward.keys()):
                    history_best_day_reward[k] *= HISTORY_DECAY
                if is_master:
                    print(f"  [DECAY] episode={episode} history_best decayed by {HISTORY_DECAY}", flush=True)

            # --- End-of-step cleanup to prevent RAM leak ---
            # Each step rollout_results carries all 6 rollouts' knot_plans with
            # raw_output text, prompts, block_reward_trace. Without explicit del
            # these pile up across steps (~500GB leak observed over 40 steps).
            del rollout_results
            if 'local_results' in dir():
                del local_results
            if 'step_advs' in dir():
                del step_advs
            if 'block_cross_advs' in dir():
                del block_cross_advs
            if 'gathered' in dir():
                del gathered
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Done ---
        elapsed = time.time() - started_at
        if is_master:
            print(f"\nStage 2 training complete. {args.max_steps} steps in {elapsed/3600:.1f}h", flush=True)
            _write_phase(phase_handle, step_index=args.max_steps, phase="training_complete",
                         elapsed_hours=round(elapsed / 3600, 2))

    if wandb_run is not None:
        wandb_run.finish()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
