#!/usr/bin/env python3
"""Eval a single RLlib model. Supports --eval-set for different date ranges."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/AD/user/lab/asim")
os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("RAY_DASHBOARD_ENABLED", "0")
SHARED_SITE_PACKAGES = PROJECT_ROOT / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if importlib.util.find_spec("pythermalcomfort") is None and SHARED_SITE_PACKAGES.exists():
    sys.path.append(str(SHARED_SITE_PACKAGES))

from gspo_houston_bandit import HoustonGSPOBandit
from llm_setpoint_planner import LLMSetpointPlanner, PlannerConstraints

EVAL_SETS = {
    "aug_lastweek": {
        "idf": PROJECT_ROOT / "houston.idf",
        "days": [
            {"date": "2025-08-25", "skip_valid_steps": 1200},
            {"date": "2025-08-26", "skip_valid_steps": 1275},
            {"date": "2025-08-27", "skip_valid_steps": 1350},
            {"date": "2025-08-28", "skip_valid_steps": 1425},
            {"date": "2025-08-29", "skip_valid_steps": 1500},
        ],
    },
    "sep_week1": {
        "idf": PROJECT_ROOT / "houston_2025_09_eval.idf",
        "days": [
            {"date": "2025-09-01", "skip_valid_steps": 0},
            {"date": "2025-09-02", "skip_valid_steps": 75},
            {"date": "2025-09-03", "skip_valid_steps": 150},
            {"date": "2025-09-04", "skip_valid_steps": 225},
            {"date": "2025-09-05", "skip_valid_steps": 300},
        ],
    },
}

MODELS = {
    "ppo_forecast": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/houston_aug2025_ppo_cc_ep5000_x300_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "forecast_window",
    },
    "ppo_no_forecast": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/houston_aug2025_ppo_cc_ep5000_x300_no_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "no_forecast_window",
    },
    "lstm_forecast": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/houston_aug2025_lstm_cc_ep5000_x300_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_lstm",
        "variant": "forecast_window",
    },
    "lstm_no_forecast": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/houston_aug2025_lstm_cc_ep5000_x300_no_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_lstm",
        "variant": "no_forecast_window",
    },
    # 3-week limited training (Aug 1-22 only, fair comparison with GRPO)
    "ppo_forecast_3wk": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/houston_aug2025_3wk_ppo_cc_ep5000_x300_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "forecast_window",
    },
    "ppo_no_forecast_3wk": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/houston_aug2025_3wk_ppo_cc_ep5000_x300_no_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "no_forecast_window",
    },
    "lstm_forecast_3wk": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/houston_aug2025_3wk_lstm_cc_ep5000_x300_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_lstm",
        "variant": "forecast_window",
    },
    "lstm_no_forecast_3wk": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/houston_aug2025_3wk_lstm_cc_ep5000_x300_no_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_lstm",
        "variant": "no_forecast_window",
    },
}


class RLLibRollingPlanner:
    def __init__(self, *, algo: Any, zone_ids: tuple, step_minutes: int = 10, policy_id: str = "default_policy"):
        self.algo = algo
        self.zone_ids = tuple(zone_ids)
        self.policy_id = policy_id
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
        self.state = self.algo.get_policy(self.policy_id).get_initial_state()

    def plan_next_action_with_trace(self, observation, *, wallclock=None, previous_action=None):
        request = self.planner.build_request(observation, wallclock=wallclock, previous_action=previous_action)
        if self.state:
            raw = self.algo.compute_single_action(observation, state=self.state, explore=False, policy_id=self.policy_id)
        else:
            raw = self.algo.compute_single_action(observation, explore=False, policy_id=self.policy_id)
        action, state_out = raw, self.state
        if isinstance(raw, tuple):
            action = raw[0]
            for item in raw[1:]:
                if isinstance(item, (list, tuple)) and len(item) == len(self.state):
                    state_out = list(item)
                    break
        self.state = state_out
        setpoints = {}
        for zone_id in self.zone_ids:
            t = action.get(zone_id, {}).get("thermostat", 24.0)
            if hasattr(t, "reshape"): t = t.reshape(-1)[0]
            elif isinstance(t, (list, tuple)): t = t[0]
            setpoints[zone_id] = float(t)
        sanitized = self.planner.sanitize_setpoints(setpoints, previous_action=previous_action)
        sp = self.planner.post_check_setpoints(sanitized, request=request, previous_action=previous_action)
        return {"request": request, "setpoints": sp, "action": self.planner.to_env_action(sp),
                "candidate_count": 1, "candidate_summaries": []}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        print(f"Usage: {sys.argv[0]} <model_name> [--eval-set aug_lastweek|sep_week1]")
        print(f"Models: {list(MODELS.keys())}")
        sys.exit(1)

    name = sys.argv[1]
    eval_set_name = "aug_lastweek"
    if "--eval-set" in sys.argv:
        idx = sys.argv.index("--eval-set")
        if idx + 1 < len(sys.argv):
            eval_set_name = sys.argv[idx + 1]
    if eval_set_name not in EVAL_SETS:
        print(f"Unknown eval set: {eval_set_name}. Available: {list(EVAL_SETS.keys())}")
        sys.exit(1)

    eval_set = EVAL_SETS[eval_set_name]
    eval_days = eval_set["days"]
    eval_idf = eval_set["idf"]
    info = MODELS[name]
    out_path = PROJECT_ROOT / f"result/comparisons/eval_{name}_{eval_set_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    ns: dict = {}
    for i in range(5):
        p = PROJECT_ROOT / f"{info['cell_prefix']}_cell{i}.py"
        if p.exists():
            exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), ns)

    variant = info["variant"]
    env_cls = ns["ENV_VARIANTS"][variant]["env_cls"]
    config = (
        PPOConfig()
        .update_from_dict(ns["get_config"](variant))
        .environment(env_cls)
        .env_runners(enable_connectors=False, num_env_runners=0, create_env_on_local_worker=False)
        .resources(num_gpus=0)
    )
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=4)
    algo = config.build()
    algo.restore(str(info["checkpoint"]))

    include_forecast = info["variant"] == "forecast_window"
    bandit = HoustonGSPOBandit(
        include_forecast=include_forecast,
        control_window_start="06:30",
        control_window_end="19:00",
        weekday_only=True,
        request_mode="step_action",
        building_path=eval_idf,
    )
    baseline_action = {z: {"thermostat": 24.0} for z in bandit.zone_ids}

    results = []
    try:
        for day in eval_days:
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
            print(f"[{name}] {day['date']} rel={row['relative_day_return']:+.4f} ({elapsed:.0f}s)", flush=True)
            out_path.write_text(json.dumps({"method": name, "eval_set": eval_set_name, "rows": results,
                "total_relative": sum(r["relative_day_return"] for r in results)}, indent=2), encoding="utf-8")
    finally:
        algo.stop()
        ray.shutdown()

    total = sum(r["relative_day_return"] for r in results)
    mean = total / len(results)
    print(f"\n[{name}] DONE: total={total:+.2f} mean={mean:+.2f}")
    out_path.write_text(json.dumps({"method": name, "eval_set": eval_set_name, "rows": results,
        "total_relative": total, "mean_relative": mean}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
