#!/usr/bin/env python3
"""Eval GRPO 30min checkpoint on Sep 1-5 using 30min knot constants.

Run with: CUDA_VISIBLE_DEVICES=1 .venv_qwen/bin/python eval_grpo_30min_sep.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/AD/user/lab/asim")

# .venv_qwen has transformers/torch; append .venv for pythermalcomfort/energyplus
SHARED_SITE_PACKAGES = PROJECT_ROOT / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if importlib.util.find_spec("pythermalcomfort") is None and SHARED_SITE_PACKAGES.exists():
    sys.path.append(str(SHARED_SITE_PACKAGES))

# Load 30min copies as modules (so the running 10min training is not affected)
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

llm_mod = _load_module("llm_setpoint_planner", PROJECT_ROOT / "llm_setpoint_planner_30min.py")
bandit_mod = _load_module("gspo_houston_bandit", PROJECT_ROOT / "gspo_houston_bandit_30min.py")

HoustonGSPOBandit = bandit_mod.HoustonGSPOBandit
BlockPlanner = llm_mod.BlockPlanner
PlannerConstraints = llm_mod.PlannerConstraints
TransformersSamplingBackend = llm_mod.TransformersSamplingBackend

EVAL_DAYS = [
    {"date": "2025-09-01", "skip_valid_steps": 0},
    {"date": "2025-09-02", "skip_valid_steps": 75},
    {"date": "2025-09-03", "skip_valid_steps": 150},
    {"date": "2025-09-04", "skip_valid_steps": 225},
    {"date": "2025-09-05", "skip_valid_steps": 300},
]

GRPO_CHECKPOINT = PROJECT_ROOT / "result" / "gspo" / "qwen3_houston_grpo_3week_8ep_pmv_kl01_lr2e5_rs3_20260331"
OUTPUT_PATH = PROJECT_ROOT / "result" / "comparisons" / "eval_grpo_30min_sep_week1.json"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    best_ckpt = GRPO_CHECKPOINT / "checkpoint-75"
    if not best_ckpt.exists():
        ckpts = sorted(GRPO_CHECKPOINT.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        best_ckpt = ckpts[-1]
    print(f"Loading GRPO from {best_ckpt.name}", flush=True)

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
    model = PeftModel.from_pretrained(model, str(best_ckpt), is_trainable=False)
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
        building_path=PROJECT_ROOT / "houston_2025_09_eval.idf",
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
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
            "method": "grpo_30min",
            "date": day["date"],
            "skip_valid_steps": day["skip_valid_steps"],
            "target_date": result.get("target_date"),
            "relative_day_return": float(total_rel),
            "day_return": float(baseline_total + total_rel),
            "baseline_day_return": float(baseline_total),
            "elapsed_s": elapsed,
        }
        results.append(row)
        print(f"[GRPO-30min] {day['date']} rel={total_rel:+.4f} ({elapsed:.0f}s)", flush=True)
        OUTPUT_PATH.write_text(json.dumps({"method": "grpo_30min", "eval_set": "sep_week1",
            "rows": results, "total_relative": sum(r["relative_day_return"] for r in results)},
            indent=2), encoding="utf-8")

    total = sum(r["relative_day_return"] for r in results)
    mean = total / len(results)
    print(f"\n[GRPO-30min] DONE: total={total:+.2f} mean={mean:+.2f}")
    OUTPUT_PATH.write_text(json.dumps({"method": "grpo_30min", "eval_set": "sep_week1",
        "rows": results, "total_relative": total, "mean_relative": mean}, indent=2), encoding="utf-8")

    del model, backend
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
