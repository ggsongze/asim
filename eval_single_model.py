#!/usr/bin/env python3
"""Eval a single RLlib model. Supports --eval-set for different date ranges."""
from __future__ import annotations

import importlib.util
import argparse
from copy import deepcopy
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/songze/asim")
DEFAULT_FORECAST_CSV = "miami_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6.csv"
SWAP27_FORECAST_CSV = (
    "miami_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6_aug9_swap27.csv"
)
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
    # Miami 3-week training
    "miami_ppo_nofc_3wk": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/miami_aug2025_3wk_ppo_nofc_ep5000_x300_no_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "no_forecast_window",
    },
    "miami_ppo_fc_3wk": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/miami_aug2025_3wk_ppo_fc_ep5000_x300_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "forecast_window",
    },
    # Miami 30min control (fair comparison with GRPO 30min knot)
    "miami_ppo_nofc_30min": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/miami_aug2025_3wk_ppo_nofc_30min_ep5000_x300_no_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "no_forecast_window",
    },
    "miami_ppo_fc_30min": {
        "checkpoint": PROJECT_ROOT / "result/manual_train/miami_aug2025_3wk_ppo_fc_30min_ep5000_x300_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "forecast_window",
    },
    "miami_ppo_fc_bl23": {
        "checkpoint": PROJECT_ROOT
        / "result/manual_train/miami_3x_hvac_cooling_bl23_ep5000_x300_forecast_window_manual/checkpoint",
        "cell_prefix": ".tmp_todo_random_start",
        "variant": "forecast_window",
    },
}

# Miami eval sets
EVAL_SETS["miami_aug_lastweek"] = {
    "idf": PROJECT_ROOT / "miami.idf",
    "epw": PROJECT_ROOT / "weather/miami_2025_06_01_2025_09_30_historical_weather_api.epw",
    "forecast_csv": DEFAULT_FORECAST_CSV,
    "days": [
        {"date": "2025-08-25", "skip_valid_steps": 1200},
        {"date": "2025-08-26", "skip_valid_steps": 1275},
        {"date": "2025-08-27", "skip_valid_steps": 1350},
        {"date": "2025-08-28", "skip_valid_steps": 1425},
        {"date": "2025-08-29", "skip_valid_steps": 1500},
    ],
}

EVAL_SETS["miami_aug_lastweek_swap27"] = {
    "idf": PROJECT_ROOT / "miami.idf",
    "epw": PROJECT_ROOT
    / "weather/miami_2025_06_01_2025_09_30_historical_weather_api_aug9_swap27.epw",
    "forecast_csv": SWAP27_FORECAST_CSV,
    "days": EVAL_SETS["miami_aug_lastweek"]["days"],
}

EVAL_SETS["miami_aug_lastweek_7_19"] = {
    "idf": PROJECT_ROOT / "miami.idf",
    "epw": PROJECT_ROOT / "weather/miami_2025_06_01_2025_09_30_historical_weather_api.epw",
    "forecast_csv": DEFAULT_FORECAST_CSV,
    "control_window_start": "07:00",
    "control_window_end": "19:00",
    "days": [
        {"date": "2025-08-25", "skip_valid_steps": 1152},
        {"date": "2025-08-26", "skip_valid_steps": 1224},
        {"date": "2025-08-27", "skip_valid_steps": 1296},
        {"date": "2025-08-28", "skip_valid_steps": 1368},
        {"date": "2025-08-29", "skip_valid_steps": 1440},
    ],
}

EVAL_SETS["miami_aug_lastweek_swap27_7_19"] = {
    "idf": PROJECT_ROOT / "miami.idf",
    "epw": PROJECT_ROOT
    / "weather/miami_2025_06_01_2025_09_30_historical_weather_api_aug9_swap27.epw",
    "forecast_csv": SWAP27_FORECAST_CSV,
    "control_window_start": "07:00",
    "control_window_end": "19:00",
    "days": EVAL_SETS["miami_aug_lastweek_7_19"]["days"],
}


LEGACY_296_OBS_DROP_KEYS = ("energy_building", "outdoor_temp", "cloud_cover")


def _checkpoint_obs_dim(checkpoint: Path) -> int | None:
    policy_path = checkpoint / "policies/default_policy/policy_state.pkl"
    if not policy_path.exists():
        return None
    with policy_path.open("rb") as f:
        state = pickle.load(f)
    weights = state.get("weights", {})
    first_layer = weights.get("_hidden_layers.0._model.0.weight")
    if first_layer is None:
        return None
    return int(first_layer.shape[1])


def _strip_observation_keys_from_space(observation_space: Any, drop_keys: tuple[str, ...]) -> Any:
    stripped = deepcopy(observation_space)
    for zone_space in stripped.spaces.values():
        for key in drop_keys:
            zone_space.spaces.pop(key, None)
    return stripped


def _strip_observation_keys(observation: dict[str, dict[str, Any]], drop_keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    if not drop_keys:
        return observation
    filtered = {}
    for zone_id, zone_obs in observation.items():
        filtered[zone_id] = {
            key: value for key, value in zone_obs.items() if key not in drop_keys
        }
    return filtered


class RLLibRollingPlanner:
    def __init__(
        self,
        *,
        algo: Any,
        zone_ids: tuple,
        step_minutes: int = 10,
        policy_id: str = "default_policy",
        drop_observation_keys: tuple[str, ...] = (),
    ):
        self.algo = algo
        self.zone_ids = tuple(zone_ids)
        self.policy_id = policy_id
        self.drop_observation_keys = tuple(drop_observation_keys)
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
        policy_observation = _strip_observation_keys(observation, self.drop_observation_keys)
        if self.state:
            raw = self.algo.compute_single_action(
                policy_observation,
                state=self.state,
                explore=False,
                policy_id=self.policy_id,
            )
        else:
            raw = self.algo.compute_single_action(
                policy_observation,
                explore=False,
                policy_id=self.policy_id,
            )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_name", choices=sorted(MODELS))
    parser.add_argument("--eval-set", choices=sorted(EVAL_SETS), default="aug_lastweek")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--baseline-setpoint", type=float, default=24.0)
    parser.add_argument("--control-window-start", type=str, default=None)
    parser.add_argument("--control-window-end", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    name = args.model_name
    eval_set = EVAL_SETS[args.eval_set]
    control_window_start = args.control_window_start or eval_set.get("control_window_start", "06:30")
    control_window_end = args.control_window_end or eval_set.get("control_window_end", "19:00")
    eval_days = list(eval_set["days"])
    if args.date is not None:
        eval_days = [day for day in eval_days if day["date"] == args.date]
        if not eval_days:
            raise ValueError(f"date {args.date} is not in eval set {args.eval_set}")
    eval_idf = eval_set["idf"]
    eval_epw = eval_set.get("epw")
    os.environ["RL_IDF"] = str(eval_idf)
    if eval_epw is not None:
        os.environ["RL_EPW"] = str(eval_epw)
    if eval_set.get("forecast_csv"):
        os.environ["RL_FORECAST_CSV"] = str(eval_set["forecast_csv"])
    info = dict(MODELS[name])
    if args.checkpoint is not None:
        info["checkpoint"] = args.checkpoint.expanduser().resolve()
    out_path = args.output or PROJECT_ROOT / f"result/comparisons/eval_{name}_{args.eval_set}.json"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    ns: dict = {}
    # Cell 0/1 define the env variants and PPO config. Later notebook-exported
    # cells build Tune/W&B trainers and must not run during eval.
    for i in range(2):
        p = PROJECT_ROOT / f"{info['cell_prefix']}_cell{i}.py"
        if p.exists():
            exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), ns)

    variant = info["variant"]
    env_cls = ns["ENV_VARIANTS"][variant]["env_cls"]
    drop_observation_keys: tuple[str, ...] = ()
    checkpoint_obs_dim = _checkpoint_obs_dim(Path(info["checkpoint"]))
    if checkpoint_obs_dim == 296 and variant == "forecast_window":
        drop_observation_keys = LEGACY_296_OBS_DROP_KEYS
        eval_config = dict(env_cls.config)
        eval_config["observation_space"] = _strip_observation_keys_from_space(
            eval_config["observation_space"],
            drop_observation_keys,
        )

        class Legacy296EvalEnv(env_cls):
            config = eval_config

        env_cls = Legacy296EvalEnv
        print(
            "[EVAL] checkpoint expects 296-dim obs; dropping "
            f"{list(drop_observation_keys)} for PPO policy input",
            flush=True,
        )
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
        control_window_start=control_window_start,
        control_window_end=control_window_end,
        weekday_only=True,
        request_mode="step_action",
        building_path=eval_idf,
        weather_path=eval_epw,
    )
    baseline_action = {
        z: {"thermostat": float(args.baseline_setpoint)}
        for z in bandit.zone_ids
    }

    results = []
    try:
        for day in eval_days:
            planner = RLLibRollingPlanner(
                algo=algo,
                zone_ids=bandit.zone_ids,
                drop_observation_keys=drop_observation_keys,
            )
            t0 = time.time()
            result = bandit.evaluate_planner_workday_closed_loop(
                skip_valid_steps=day["skip_valid_steps"],
                planner=planner,
                baseline_action=baseline_action,
            )
            elapsed = time.time() - t0
            # Aggregate physical quantities from reward_trace
            cand_trace = result.get("candidate_reward_trace", [])
            bl_trace = result.get("baseline_reward_trace", [])
            def _sum_phys(trace, key):
                return sum(s.get(key, 0) for s in trace)
            row = {
                "method": name,
                "date": day["date"],
                "skip_valid_steps": day["skip_valid_steps"],
                "target_date": result.get("target_date"),
                "relative_day_return": float(result["relative_day_return"]),
                "day_return": float(result["day_return"]),
                "baseline_day_return": float(result["baseline_day_return"]),
                "elapsed_s": elapsed,
                "facility_kwh": round(_sum_phys(cand_trace, "facility_kwh"), 2),
                "pv_kwh": round(_sum_phys(cand_trace, "pv_kwh"), 2),
                "net_grid_kwh": round(_sum_phys(cand_trace, "net_grid_kwh"), 2),
                "pmv_violation_sum": round(_sum_phys(cand_trace, "total_pmv_violation"), 4),
                "bl_facility_kwh": round(_sum_phys(bl_trace, "facility_kwh"), 2),
                "bl_pv_kwh": round(_sum_phys(bl_trace, "pv_kwh"), 2),
                "bl_net_grid_kwh": round(_sum_phys(bl_trace, "net_grid_kwh"), 2),
                "bl_pmv_violation_sum": round(_sum_phys(bl_trace, "total_pmv_violation"), 4),
            }
            results.append(row)
            fac = row["facility_kwh"]; pvk = row["pv_kwh"]; pvu = (1 - row["net_grid_kwh"]/fac)*100 if fac > 0 else 0
            print(f"[{name}] {day['date']} rel={row['relative_day_return']:+.4f} "
                  f"facility={fac:.0f}kWh pv_util={pvu:.0f}% pmv_viol={row['pmv_violation_sum']:.3f} ({elapsed:.0f}s)", flush=True)
            out_path.write_text(json.dumps({
                "method": name,
                "eval_set": args.eval_set,
                "control_window": f"{control_window_start}-{control_window_end}",
                "baseline_setpoint": float(args.baseline_setpoint),
                "building_idf": str(eval_idf),
                "weather_epw": str(eval_epw) if eval_epw else None,
                "forecast_csv": os.environ.get("RL_FORECAST_CSV"),
                "rows": results,
                "total_relative": sum(r["relative_day_return"] for r in results),
            }, indent=2), encoding="utf-8")
    finally:
        algo.stop()
        ray.shutdown()

    total = sum(r["relative_day_return"] for r in results)
    mean = total / len(results)
    print(f"\n[{name}] DONE: total={total:+.2f} mean={mean:+.2f}")
    out_path.write_text(json.dumps({
        "method": name,
        "eval_set": args.eval_set,
        "control_window": f"{control_window_start}-{control_window_end}",
        "baseline_setpoint": float(args.baseline_setpoint),
        "building_idf": str(eval_idf),
        "weather_epw": str(eval_epw) if eval_epw else None,
        "forecast_csv": os.environ.get("RL_FORECAST_CSV"),
        "rows": results,
        "total_relative": total,
        "mean_relative": mean,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
