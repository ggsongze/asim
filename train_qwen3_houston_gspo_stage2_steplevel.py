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
        snapshot[matched_name] = raw_tensor.to(
            device=param.device,
            dtype=param.dtype,
        ).clone()

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
    parser.add_argument("--seed", type=int, default=1229)
    parser.add_argument("--save-steps", type=int, default=4)
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
) -> list[list[float]]:
    """Compute per-knot advantages using return-to-go + cross-rollout z-score.

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
    T = r.shape[1]  # 26 for Miami

    # 2) Discounted return-to-go
    G_ret = np.zeros_like(r)
    running = np.zeros(G, dtype=np.float64)
    for t in reversed(range(T)):
        running = r[:, t] + gamma * running
        G_ret[:, t] = running

    # 3) Per-step z-score across G rollouts
    mean = G_ret.mean(axis=0, keepdims=True)  # (1, T)
    std = G_ret.std(axis=0, keepdims=True)    # (1, T)
    A = np.where(std > 1e-6, (G_ret - mean) / (std + 1e-4), 0.0)  # (G, T)

    return A.tolist()  # advs[rollout_idx][knot_idx]


def _expected_day_control_steps(bandit: MiamiGRPOBandit) -> int:
    """Return the number of eligible control steps in one Stage 2 day."""
    return sum(int(bandit._block_env_steps(i)) for i in range(len(bandit.BLOCK_DEFINITIONS)))


def _expected_day_knots(bandit: MiamiGRPOBandit) -> int:
    """Return the number of planned knots in one Stage 2 day."""
    return sum(int(bandit._block_knots(i)) for i in range(len(bandit.BLOCK_DEFINITIONS)))


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
    accumulated_actions: list[dict[str, dict[str, float]]] = []
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

    for block_index, (block_start, block_end) in enumerate(bandit.BLOCK_DEFINITIONS):
        proxy = _Stage2FreeSamplePlannerProxy(
            block_planner,
            exploration_mode_hint=exploration_mode_hint,
            setpoint_only=setpoint_only,
            setpoint_exploration_probability=setpoint_exploration_probability,
            setpoint_exploration_state=setpoint_exploration_state,
            rng=rng,
        )
        result = bandit._rollout_block_rolling(
            skip_valid_steps=skip_valid_steps,
            replay_actions=list(accumulated_actions),
            baseline_action=baseline_action,
            planner=proxy,
            block_index=block_index,
            block_start=block_start,
            block_end=block_end,
            mode="balanced",  # placeholder, proxy overrides via plan_knot_free
        )

        block_reward = float(result.get("block_reward", 0.0))
        baseline_reward = float(baseline_block_rewards[block_index])
        relative_reward = reward_scale * (block_reward - baseline_reward)
        total_day_reward += relative_reward

        chosen_mode = proxy.chosen_mode or "balanced"
        if setpoint_only:
            chosen_mode = (
                proxy.setpoint_exploration_hint.get("label")
                if proxy.setpoint_exploration_hint
                else "setpoint_only"
            )
            setpoint_macro_labels.append(chosen_mode)
        all_modes.append(chosen_mode)

        knot_plans = result.get("knot_plans", [])
        all_knot_plans.extend(knot_plans)
        knot_to_block.extend([block_index] * len(knot_plans))

        # Expand knots to env-step actions for replay in subsequent blocks
        winner_knots = [kp["knot"] for kp in knot_plans if "knot" in kp]
        if winner_knots:
            block_actions = bandit._expand_knots_to_env_steps(
                winner_knots, block_index=block_index, allow_partial=True,
            )
            accumulated_actions.extend(block_actions)

        brt = result.get("block_reward_trace", [])
        hvac_kwh = sum(float(s.get("hvac_kwh", 0.0)) for s in brt)
        pmv_violation = sum(float(s.get("total_pmv_violation", 0.0)) for s in brt)

        # Record for cross-block context injection
        block_planner.record_block_result(
            block_index=block_index,
            block_start=str(block_start),
            block_end=str(block_end),
            winner_mode=chosen_mode,
            winner_reward=relative_reward,
            hvac_kwh=hvac_kwh,
            pmv_violation=pmv_violation,
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
            "block_start_observation": result.get("block_start_observation"),
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    trajectory_path = args.output_dir / "trajectory_samples.jsonl"
    phase_trace_path = args.output_dir / "phase_trace.jsonl"

    rows = _load_rows(args.dataset_path)
    if not rows:
        raise RuntimeError(f"No rows found in dataset: {args.dataset_path}")
    print(f"Dataset: {len(rows)} training days from {args.dataset_path}", flush=True)

    _set_seed(int(args.seed))

    # --- Override environment to 30-min timestep ---
    MiamiGRPOBandit.STEP_MINUTES = 30
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

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
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
        lora_config = LoraConfig(
            r=target_r,
            lora_alpha=target_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # Zero-pad old weights into the new larger adapter
        if ckpt_r is not None and ckpt_r < target_r:
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
        control_window_start="06:00",
        control_window_end="19:00",
        weekday_only=True,
        request_mode="step_action",
        building_path=args.building_idf,
        weather_path=args.weather_epw,
    )
    _validate_miami_forecast_binding(bandit)

    # --- Planner ---
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

    # Load reflections from Stage 1
    reflections_path = resume_checkpoint_dir / "reflections.json"
    if reflections_path.exists():
        try:
            block_planner.load_reflections(reflections_path)
            print(f"Loaded reflections from {reflections_path}", flush=True)
        except Exception as exc:
            print(f"Warning: failed to load reflections: {exc}", flush=True)

    # --- WandB ---
    wandb_run = None
    if not args.no_wandb:
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
                settings=wandb.Settings(x_disable_stats=True, x_disable_machine_info=True),
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
    print(
        f"Stage 2 loop will run steps {start_step_index}..{args.max_steps}",
        flush=True,
    )

    with (
        metrics_path.open("w", encoding="utf-8") as metrics_handle,
        trajectory_path.open("w", encoding="utf-8") as trajectory_handle,
        phase_trace_path.open("w", encoding="utf-8") as phase_handle,
    ):
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

            # ---- 2. Sequential full-day rollouts ----
            model.eval()
            rollout_results: list[dict[str, Any]] = []
            for rollout_idx in range(args.n_rollouts):
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
                t_rollout_start = time.time()
                result = _rollout_full_day_free(
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
                t_rollout_end = time.time()
                rollout_results.append(result)
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
                print(f"  [ROLLOUT] step={step_index} rollout={rollout_idx} "
                      f"day_reward={result['day_reward']:.4f} "
                      f"(raw={result['day_reward_raw']:.4f} "
                      f"pmv_day_penalty={result['pmv_consistency_penalty']:.4f} "
                      f"semantic_diag={result['mode_setpoint_semantic_penalty']:.4f}) "
                      f"hint={exploration_hint or 'free'} "
                      f"setpoint_macros={result.get('setpoint_macro_labels', [])} "
                      f"modes={result['all_modes']} "
                      f"time={t_rollout_end - t_rollout_start:.1f}s", flush=True)

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
            print(f"  [VALIDATE] step={step_index} total={_dbg_total} none={_dbg_none} "
                  f"valid_pre={_dbg_valid_pre} valid_post={_dbg_valid_post}", flush=True)

            # ---- 7. Accumulate gradients ----
            optimizer.zero_grad(set_to_none=True)
            model.train()
            total_kl = 0.0
            total_tokens = 0
            grad_contributions = 0
            local_adv_values: list[float] = []

            for rollout_idx, rollout in enumerate(rollout_results):
                day_adv = advantages[rollout_idx]
                rollout_step_advs = step_advs[rollout_idx] if rollout_idx < len(step_advs) else []
                knot_plans = rollout["all_knot_plans"]
                knot_to_block = rollout["knot_to_block"]
                n_knots = len(knot_plans)
                divisor = args.n_rollouts * max(n_knots, 1)

                for knot_idx, knot_plan in enumerate(knot_plans):
                    block_i = knot_to_block[knot_idx] if knot_idx < len(knot_to_block) else 0
                    block_adv = block_cross_advs[block_i][rollout_idx] if block_i < len(block_cross_advs) else 0.0

                    # Step-level advantage from return-to-go
                    step_adv = rollout_step_advs[knot_idx] if knot_idx < len(rollout_step_advs) else 0.0

                    semantic_violation_c = float(
                        knot_plan.get("mode_setpoint_violation_c", 0.0)
                    )
                    local_adv = (
                        -float(args.mode_setpoint_local_adv_weight)
                        * semantic_violation_c
                    )
                    total_adv = step_adv + local_adv
                    knot_plan["step_advantage"] = step_adv
                    knot_plan["day_advantage"] = day_adv
                    knot_plan["block_cross_advantage"] = block_adv
                    knot_plan["mode_setpoint_local_adv"] = local_adv
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
                    total_tokens += tok_count
                    if tok_count > 0:
                        grad_contributions += 1

            # ---- 8. KL guard ----
            if total_kl > float(args.kl_guard_threshold):
                optimizer.zero_grad(set_to_none=True)
                model.eval()
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
                continue

            # ---- 9. Optimizer step ----
            raw_grad_norm_val = _grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.max_grad_norm))
            grad_norm_val = _grad_norm(model)
            optimizer.step()
            model.eval()

            step_elapsed = time.time() - step_start_time
            _sa_flat = [a for row in step_advs for a in row] if step_advs else [0.0]
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

            # ---- 11. Checkpoint ----
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
                # End of episode: clean cache checkpoints
                episode_start_step = step_index - len(rows) + 1
                if not args.keep_cache_checkpoints:
                    for s in range(episode_start_step, step_index + 1):
                        cache_dir = args.output_dir / f"cache-checkpoint-{s}"
                        if cache_dir.exists():
                            try:
                                shutil.rmtree(cache_dir)
                            except Exception:
                                pass
                _write_phase(phase_handle, step_index=step_index, phase="episode_end",
                             episode=episode)

            # Decay at start of each new episode (except first)
            if day_in_ep == 0 and episode > 1:
                for k in list(history_best_day_reward.keys()):
                    history_best_day_reward[k] *= HISTORY_DECAY
                print(f"  [DECAY] episode={episode} history_best decayed by {HISTORY_DECAY}", flush=True)

        # --- Done ---
        elapsed = time.time() - started_at
        print(f"\nStage 2 training complete. {args.max_steps} steps in {elapsed/3600:.1f}h", flush=True)
        _write_phase(phase_handle, step_index=args.max_steps, phase="training_complete",
                     elapsed_hours=round(elapsed / 3600, 2))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
