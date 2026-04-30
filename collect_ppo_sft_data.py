#!/usr/bin/env python3
"""Collect PPO action data as SFT training pairs for LLM distillation.

Runs the PPO checkpoint on training days (Aug 1-21) and records
(knot_system_prompt, knot_user_prompt, setpoint_json) for each control step.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/songze/asim")
SHARED_SITE_PACKAGES = PROJECT_ROOT / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if importlib.util.find_spec("pythermalcomfort") is None and SHARED_SITE_PACKAGES.exists():
    sys.path.append(str(SHARED_SITE_PACKAGES))

from gspo_houston_bandit import HoustonGSPOBandit
from llm_setpoint_planner import (
    BlockPlanner,
    LLMSetpointPlanner,
    PlannerConstraints,
    CANDIDATE_MODE_DESCRIPTIONS,
)


class PPOActionCollector:
    """Wraps a PPO algo as a step-action planner, collecting prompts + actions."""

    def __init__(self, *, algo: Any, zone_ids: tuple, block_planner: BlockPlanner,
                 policy_id: str = "default_policy"):
        self.algo = algo
        self.zone_ids = tuple(zone_ids)
        self.policy_id = policy_id
        self.block_planner = block_planner
        self.planner = LLMSetpointPlanner(
            backend=None,
            constraints=PlannerConstraints(
                min_setpoint_c=20.0, max_setpoint_c=30.0,
                max_delta_per_step_c=2.0, fallback_setpoint_c=24.0, quantization_c=0.1,
            ),
            zone_ids=self.zone_ids,
            step_minutes=10,
            candidate_count=1,
        )
        self.state = self.algo.get_policy(self.policy_id).get_initial_state()
        self.collected: list[dict] = []

    def plan_next_action_with_trace(self, observation, *, wallclock=None, previous_action=None):
        # Get PPO's action
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

        # Extract setpoints from PPO action
        setpoints = {}
        for zone_id in self.zone_ids:
            t = action.get(zone_id, {}).get("thermostat", 24.0)
            if hasattr(t, "reshape"): t = t.reshape(-1)[0]
            elif isinstance(t, (list, tuple)): t = t[0]
            setpoints[zone_id] = round(float(t), 1)

        # Sanitize through planner constraints
        sanitized = self.planner.sanitize_setpoints(setpoints, previous_action=previous_action)
        request = self.planner.build_request(observation, wallclock=wallclock, previous_action=previous_action)
        sp = self.planner.post_check_setpoints(sanitized, request=request, previous_action=previous_action)

        # Build knot-style prompt with balanced mode only (PPO's action matches balanced strategy)
        # Using only balanced avoids teaching model to ignore mode prompt
        # (if all 3 modes map to same action, model learns mode prompt is irrelevant)
        sft_mode = "balanced"
        system_prompt = self.block_planner._build_knot_system_prompt(sft_mode)
        user_prompt = self.block_planner._build_knot_user_prompt(
            block_index=0, knot_index=0,
            block_start="06:30", block_end="19:00",
            mode=sft_mode,
            observation=observation,
            wallclock=wallclock,
        )
        output_json = json.dumps({"setpoints": [sp[z] for z in self.zone_ids]})
        self.collected.append({
            "mode": sft_mode,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "output": output_json,
            "wallclock": str(wallclock) if wallclock else None,
            "setpoints": sp,
        })

        env_action = self.planner.to_env_action(sp)
        return {"request": request, "setpoints": sp, "action": env_action,
                "candidate_count": 1, "candidate_summaries": []}


def main():
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    checkpoint_path = PROJECT_ROOT / "result/manual_train/houston_aug2025_3wk_ppo_cc_ep5000_x300_forecast_window_manual/checkpoint"
    cell_prefix = ".tmp_todo_random_start"
    variant = "forecast_window"
    output_path = PROJECT_ROOT / "result/gspo/ppo_sft_dataset.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cell files
    ns: dict = {}
    for i in range(5):
        p = PROJECT_ROOT / f"{cell_prefix}_cell{i}.py"
        if p.exists():
            exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), ns)

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
    algo.restore(str(checkpoint_path))
    print(f"Loaded PPO checkpoint from {checkpoint_path}", flush=True)

    # Build a BlockPlanner (backend=None, only used for prompt building)
    from llm_setpoint_planner import HeuristicPlannerBackend
    block_planner = BlockPlanner(
        HeuristicPlannerBackend(),
        constraints=PlannerConstraints(
            min_setpoint_c=20.0, max_setpoint_c=30.0,
            max_delta_per_step_c=2.0, fallback_setpoint_c=24.0, quantization_c=0.1,
        ),
        zone_ids=tuple(ns["ZONE_MAP"].keys()),
    )

    bandit = HoustonGSPOBandit(
        include_forecast=True,
        control_window_start="06:30",
        control_window_end="19:00",
        weekday_only=True,
        request_mode="step_action",
    )
    baseline_action = {z: {"thermostat": 24.0} for z in bandit.zone_ids}

    # Eval days: 15 workdays from Aug 1-21 (same as GRPO training set)
    eval_days = []
    for i in range(15):
        eval_days.append({"date": f"day_{i+1}", "skip_valid_steps": i * 75})

    all_collected = []
    for day in eval_days:
        collector = PPOActionCollector(
            algo=algo, zone_ids=bandit.zone_ids, block_planner=block_planner,
        )
        t0 = time.time()
        try:
            result = bandit.evaluate_planner_workday_closed_loop(
                skip_valid_steps=day["skip_valid_steps"],
                planner=collector,
                baseline_action=baseline_action,
            )
            elapsed = time.time() - t0
            rel = result.get("relative_day_return", 0)
            print(f"  {day['date']}: rel={rel:+.4f} steps={len(collector.collected)//len(CANDIDATE_MODE_DESCRIPTIONS)} ({elapsed:.0f}s)", flush=True)
        except Exception as exc:
            print(f"  {day['date']}: FAILED: {exc}", flush=True)
            continue
        all_collected.extend(collector.collected)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_collected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nCollected {len(all_collected)} SFT samples ({len(all_collected)//len(CANDIDATE_MODE_DESCRIPTIONS)} steps × {len(CANDIDATE_MODE_DESCRIPTIONS)} modes)")
    print(f"Saved to {output_path}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
