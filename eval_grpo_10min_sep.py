#!/usr/bin/env python3
"""Eval GRPO 10min knot checkpoint on Sep 1-5 and Aug 25-29."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/AD/user/lab/asim")
SHARED_SITE_PACKAGES = PROJECT_ROOT / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if importlib.util.find_spec("pythermalcomfort") is None and SHARED_SITE_PACKAGES.exists():
    sys.path.append(str(SHARED_SITE_PACKAGES))

from gspo_houston_bandit import HoustonGSPOBandit
from llm_setpoint_planner import BlockPlanner, PlannerConstraints, TransformersSamplingBackend

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

GRPO_CHECKPOINT = PROJECT_ROOT / "result" / "gspo" / "qwen3_houston_grpo_3week_4ep_10min_knot_kl01_lr2e5_rs3_20260401"


def main():
    eval_set_name = sys.argv[1] if len(sys.argv) > 1 else "aug_lastweek"
    ckpt_name = sys.argv[2] if len(sys.argv) > 2 else "checkpoint-30"  # best ep2

    if eval_set_name not in EVAL_SETS:
        print(f"Usage: {sys.argv[0]} <aug_lastweek|sep_week1> [checkpoint-N]")
        sys.exit(1)

    eval_set = EVAL_SETS[eval_set_name]
    ckpt_path = GRPO_CHECKPOINT / ckpt_name
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_path = PROJECT_ROOT / f"result/comparisons/eval_grpo_10min_{eval_set_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading GRPO 10min from {ckpt_name}", flush=True)
    base_model_path = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = PeftModel.from_pretrained(model, str(ckpt_path), is_trainable=False)
    model.eval()

    backend = TransformersSamplingBackend(
        model=model, tokenizer=tokenizer, model_name=base_model_path,
        max_output_tokens=512, temperature=0.7, top_p=0.95,
    )
    bandit = HoustonGSPOBandit(
        include_forecast=True,
        control_window_start="06:30",
        control_window_end="19:00",
        weekday_only=True,
        request_mode="step_action",
        building_path=eval_set["idf"],
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
    baseline_action = {z: {"thermostat": 24.0} for z in bandit.zone_ids}

    results = []
    for day in eval_set["days"]:
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
            "method": "grpo_10min",
            "date": day["date"],
            "skip_valid_steps": day["skip_valid_steps"],
            "target_date": result.get("target_date"),
            "relative_day_return": float(total_rel),
            "day_return": float(baseline_total + total_rel),
            "baseline_day_return": float(baseline_total),
            "elapsed_s": elapsed,
        }
        results.append(row)
        print(f"[GRPO-10min] {day['date']} rel={total_rel:+.4f} ({elapsed:.0f}s)", flush=True)
        output_path.write_text(json.dumps({"method": "grpo_10min", "eval_set": eval_set_name,
            "checkpoint": ckpt_name, "rows": results,
            "total_relative": sum(r["relative_day_return"] for r in results)},
            indent=2), encoding="utf-8")

    total = sum(r["relative_day_return"] for r in results)
    mean = total / len(results)
    print(f"\n[GRPO-10min] DONE: total={total:+.2f} mean={mean:+.2f}")
    output_path.write_text(json.dumps({"method": "grpo_10min", "eval_set": eval_set_name,
        "checkpoint": ckpt_name, "rows": results,
        "total_relative": total, "mean_relative": mean}, indent=2), encoding="utf-8")

    del model, backend
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
