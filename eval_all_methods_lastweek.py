#!/usr/bin/env python3
"""Evaluate all methods (GRPO, PPO×2, LSTM×2) on Aug 25-29 (held-out last week).

All methods are evaluated through the same HoustonGSPOBandit with control_window
06:30-19:00 and 24°C baseline, producing directly comparable relative rewards.

Usage:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python eval_all_methods_lastweek.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/songze/asim")
os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("RAY_DASHBOARD_ENABLED", "0")
SHARED_SITE_PACKAGES = PROJECT_ROOT / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if importlib.util.find_spec("pythermalcomfort") is None and SHARED_SITE_PACKAGES.exists():
    sys.path.append(str(SHARED_SITE_PACKAGES))

from gspo_houston_bandit import HoustonGSPOBandit
from llm_setpoint_planner import (
    BlockPlanner,
    LLMSetpointPlanner,
    PlannerConstraints,
    TransformersSamplingBackend,
)

OUTPUT_PATH = PROJECT_ROOT / "result" / "comparisons" / "all_methods_lastweek_eval.json"

# Aug 25-29, 2025 (held-out last week), skip based on 06:30-19:00 window (75/day)
EVAL_DAYS = [
    {"date": "2025-08-25", "skip_valid_steps": 1200},
    {"date": "2025-08-26", "skip_valid_steps": 1275},
    {"date": "2025-08-27", "skip_valid_steps": 1350},
    {"date": "2025-08-28", "skip_valid_steps": 1425},
    {"date": "2025-08-29", "skip_valid_steps": 1500},
]

# PPO/LSTM checkpoints (retrained with cloudcover)
RLLIB_MODELS = {
    "ppo_forecast": {
        "checkpoint": PROJECT_ROOT / "result" / "manual_train" / "houston_aug2025_ppo_cc_ep5000_x300_forecast_window_manual" / "checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "forecast_window",
    },
    "ppo_no_forecast": {
        "checkpoint": PROJECT_ROOT / "result" / "manual_train" / "houston_aug2025_ppo_cc_ep5000_x300_no_forecast_window_manual" / "checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "no_forecast_window",
    },
    "lstm_forecast": {
        "checkpoint": PROJECT_ROOT / "result" / "manual_train" / "houston_aug2025_lstm_cc_ep5000_x300_forecast_window_manual" / "checkpoint",
        "cell_prefix": ".tmp_todo_lstm",
        "variant": "forecast_window",
    },
    "lstm_no_forecast": {
        "checkpoint": PROJECT_ROOT / "result" / "manual_train" / "houston_aug2025_lstm_cc_ep5000_x300_no_forecast_window_manual" / "checkpoint",
        "cell_prefix": ".tmp_todo_lstm",
        "variant": "no_forecast_window",
    },
}

# GRPO checkpoint
GRPO_CHECKPOINT = PROJECT_ROOT / "result" / "gspo" / "qwen3_houston_grpo_3week_8ep_pmv_kl01_lr2e5_rs3_20260331"


class RLLibRollingPlanner:
    """Wraps an RLlib algo to act as a step-action planner for the bandit."""

    def __init__(self, *, algo: Any, zone_ids: tuple[str, ...], step_minutes: int = 10, policy_id: str = "default_policy"):
        self.algo = algo
        self.zone_ids = tuple(zone_ids)
        self.policy_id = str(policy_id)
        self.planner = LLMSetpointPlanner(
            backend=None,
            constraints=PlannerConstraints(
                min_setpoint_c=20.0, max_setpoint_c=30.0,
                max_delta_per_step_c=2.0, fallback_setpoint_c=24.0, quantization_c=0.1,
            ),
            zone_ids=self.zone_ids,
            step_minutes=step_minutes,
            candidate_count=1,
        )
        policy = self.algo.get_policy(self.policy_id)
        self.state = policy.get_initial_state()

    def _raw_action_to_setpoints(self, action: dict[str, Any]) -> dict[str, float]:
        setpoints: dict[str, float] = {}
        for zone_id in self.zone_ids:
            zone_action = action.get(zone_id, {})
            thermostat = zone_action.get("thermostat", 24.0)
            if hasattr(thermostat, "reshape"):
                thermostat = thermostat.reshape(-1)[0]
            elif isinstance(thermostat, (list, tuple)):
                thermostat = thermostat[0]
            setpoints[zone_id] = float(thermostat)
        return setpoints

    def plan_next_action_with_trace(
        self, observation: dict[str, dict[str, Any]], *,
        wallclock: Any = None, previous_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = self.planner.build_request(
            observation, wallclock=wallclock, previous_action=previous_action,
        )
        if self.state:
            raw_result = self.algo.compute_single_action(
                observation, state=self.state, explore=False, policy_id=self.policy_id,
            )
        else:
            raw_result = self.algo.compute_single_action(
                observation, explore=False, policy_id=self.policy_id,
            )
        action = raw_result
        state_out = self.state
        if isinstance(raw_result, tuple):
            action = raw_result[0]
            if self.state:
                for item in raw_result[1:]:
                    if isinstance(item, (list, tuple)) and len(item) == len(self.state):
                        state_out = list(item)
                        break
        self.state = state_out
        raw_setpoints = self._raw_action_to_setpoints(action)
        sanitized = self.planner.sanitize_setpoints(raw_setpoints, previous_action=previous_action)
        setpoints = self.planner.post_check_setpoints(sanitized, request=request, previous_action=previous_action)
        env_action = self.planner.to_env_action(setpoints)
        return {
            "request": request,
            "raw_output": raw_setpoints,
            "sanitized_setpoints": sanitized,
            "setpoints": setpoints,
            "action": env_action,
            "candidate_count": 1,
            "candidate_summaries": [],
        }


def load_namespace(cell_prefix: str) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    for i in range(5):
        path = PROJECT_ROOT / f"{cell_prefix}_cell{i}.py"
        if path.exists():
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return ns


def build_rllib_algo(checkpoint_path: Path, cell_prefix: str, variant: str = "forecast_window") -> Any:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    ns = load_namespace(cell_prefix)
    env_cls = ns["ENV_VARIANTS"][variant]["env_cls"]
    config = PPOConfig().update_from_dict(ns["get_config"](variant))
    config = (
        config
        .environment(env_cls)
        .env_runners(
            enable_connectors=False,
            num_env_runners=0,
            create_env_on_local_worker=False,
        )
        .resources(num_gpus=0)
    )
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=4)
    algo = config.build()
    algo.restore(str(checkpoint_path))
    return algo


def eval_rllib_model(
    name: str,
    model_info: dict,
    bandit: HoustonGSPOBandit,
    baseline_action: dict,
) -> list[dict]:
    import ray

    algo = build_rllib_algo(model_info["checkpoint"], model_info["cell_prefix"], model_info.get("variant", "forecast_window"))
    results = []
    try:
        for day in EVAL_DAYS:
            planner = RLLibRollingPlanner(algo=algo, zone_ids=bandit.zone_ids)
            t0 = time.time()
            result = bandit.evaluate_planner_workday_closed_loop(
                skip_valid_steps=day["skip_valid_steps"],
                planner=planner,
                baseline_action=baseline_action,
            )
            elapsed = time.time() - t0
            row = {
                "method": name,
                "date": day["date"],
                "skip_valid_steps": day["skip_valid_steps"],
                "target_date": result.get("target_date"),
                "relative_day_return": float(result["relative_day_return"]),
                "day_return": float(result["day_return"]),
                "baseline_day_return": float(result["baseline_day_return"]),
                "elapsed_s": elapsed,
            }
            results.append(row)
            print(f"  [{name}] {day['date']} rel={row['relative_day_return']:+.4f} ({elapsed:.0f}s)", flush=True)
    finally:
        algo.stop()
        ray.shutdown()
    return results


def eval_grpo(
    bandit: HoustonGSPOBandit,
    baseline_action: dict,
    checkpoint_dir: Path,
) -> list[dict]:
    """Evaluate GRPO using evaluate_workday_blocks with the latest checkpoint."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Use best checkpoint (ep5 = checkpoint-75) or latest
    best_ckpt = checkpoint_dir / "checkpoint-75"
    if best_ckpt.exists():
        latest_ckpt = best_ckpt
    else:
        ckpts = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if not ckpts:
            raise RuntimeError(f"No checkpoints found in {checkpoint_dir}")
        latest_ckpt = ckpts[-1]
    print(f"  Loading GRPO from {latest_ckpt.name}", flush=True)

    # Load model
    base_model_path = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = PeftModel.from_pretrained(model, str(latest_ckpt), is_trainable=False)
    model.eval()

    backend = TransformersSamplingBackend(
        model=model, tokenizer=tokenizer,
        model_name=base_model_path,
        max_output_tokens=512,
        temperature=0.7, top_p=0.95,
    )
    block_planner = BlockPlanner(
        backend,
        constraints=PlannerConstraints(
            min_setpoint_c=20.0, max_setpoint_c=30.0,
            max_delta_per_step_c=2.0, fallback_setpoint_c=24.0, quantization_c=0.1,
        ),
        zone_ids=bandit.zone_ids,
        max_generation_attempts=2,
    )

    results = []
    for day in EVAL_DAYS:
        t0 = time.time()
        result = bandit.evaluate_workday_blocks(
            skip_valid_steps=day["skip_valid_steps"],
            planner=block_planner,
            baseline_action=baseline_action,
            candidate_modes=["comfort", "balanced", "energy_saving"],
        )
        elapsed = time.time() - t0
        total_rel = result["total_winner_relative_reward"]
        baseline_total = sum(result["baseline_block_rewards"])
        row = {
            "method": "grpo",
            "date": day["date"],
            "skip_valid_steps": day["skip_valid_steps"],
            "target_date": result.get("target_date"),
            "relative_day_return": float(total_rel),
            "day_return": float(baseline_total + total_rel),
            "baseline_day_return": float(baseline_total),
            "elapsed_s": elapsed,
        }
        results.append(row)
        print(f"  [GRPO] {day['date']} rel={total_rel:+.4f} ({elapsed:.0f}s)", flush=True)

    del model, backend
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-grpo", action="store_true", help="Skip GRPO eval")
    parser.add_argument("--skip-rllib", action="store_true", help="Skip PPO/LSTM eval")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bandit = HoustonGSPOBandit(
        include_forecast=True,
        control_window_start="06:30",
        control_window_end="19:00",
        weekday_only=True,
        request_mode="step_action",
    )
    baseline_action = {zone_id: {"thermostat": 24.0} for zone_id in bandit.zone_ids}

    all_results: dict[str, Any] = {"eval_days": [d["date"] for d in EVAL_DAYS]}
    started = time.time()

    # --- GRPO ---
    if not args.skip_grpo:
        print("=== Evaluating GRPO ===", flush=True)
        grpo_results = eval_grpo(bandit, baseline_action, GRPO_CHECKPOINT)
        all_results["grpo"] = {
            "rows": grpo_results,
            "total_relative": sum(r["relative_day_return"] for r in grpo_results),
            "mean_relative": sum(r["relative_day_return"] for r in grpo_results) / len(grpo_results),
        }
        OUTPUT_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- PPO / LSTM ---
    if not args.skip_rllib:
        for name, info in RLLIB_MODELS.items():
            print(f"\n=== Evaluating {name} ===", flush=True)
            results = eval_rllib_model(name, info, bandit, baseline_action)
            all_results[name] = {
                "rows": results,
                "total_relative": sum(r["relative_day_return"] for r in results),
                "mean_relative": sum(r["relative_day_return"] for r in results) / len(results),
            }
            OUTPUT_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - started
    all_results["elapsed_s"] = elapsed
    all_results["status"] = "completed"
    OUTPUT_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    print(f"\n=== Done in {elapsed:.0f}s ===")
    print(f"\n{'Method':<20} {'Total':>8} {'Mean/day':>10}")
    print("-" * 40)
    for key in ["grpo"] + list(RLLIB_MODELS.keys()):
        if key in all_results:
            d = all_results[key]
            print(f"{key:<20} {d['total_relative']:+8.2f} {d['mean_relative']:+10.2f}")


if __name__ == "__main__":
    main()
