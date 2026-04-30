#!/usr/bin/env python3
"""Evaluate the current Miami 10min GRPO checkpoint.

Defaults are aligned with the active v15 run:
  - Qwen/Qwen3.5-9B LoRA checkpoint-8
  - Miami, 10 minute control knots
  - 07:00-19:00 control window, 72 control steps per day
  - setpoint-only policy, no mode candidate search
  - fixed 23.0 C baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/songze/asim")
DEFAULT_RUN_DIR = PROJECT_ROOT / "result/gspo/qwen35_9b_v15_vllm_20260428_2112"
DEFAULT_CHECKPOINT_NAME = "checkpoint-8"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_FORECAST_CSV = "miami_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6.csv"
DEFAULT_EPW = PROJECT_ROOT / "weather/miami_2025_06_01_2025_09_30_historical_weather_api.epw"
SWAP27_FORECAST_CSV = (
    "miami_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6_aug9_swap27.csv"
)
SWAP27_EPW = (
    PROJECT_ROOT
    / "weather/miami_2025_06_01_2025_09_30_historical_weather_api_aug9_swap27.epw"
)

MIAMI_AUG_LASTWEEK = {
    # This IDF extends past Aug22, so it can evaluate Aug25-29. The stage2
    # training IDF only runs through Aug22.
    "idf": PROJECT_ROOT / "miami.idf",
    "epw": DEFAULT_EPW,
    "forecast_csv": DEFAULT_FORECAST_CSV,
    # 07:00-19:00 = 12h = 72 ten-minute valid control steps per weekday.
    "days": [
        {"date": "2025-08-25", "skip_valid_steps": 1152},
        {"date": "2025-08-26", "skip_valid_steps": 1224},
        {"date": "2025-08-27", "skip_valid_steps": 1296},
        {"date": "2025-08-28", "skip_valid_steps": 1368},
        {"date": "2025-08-29", "skip_valid_steps": 1440},
    ],
}

MIAMI_AUG_LASTWEEK_SWAP27 = {
    # Aug9 storm weather swapped onto Aug27. Aug28-29 carry the storm aftermath
    # through the simulator state, matching the README swap27 evaluation setup.
    "idf": PROJECT_ROOT / "miami.idf",
    "epw": SWAP27_EPW,
    "forecast_csv": SWAP27_FORECAST_CSV,
    "days": MIAMI_AUG_LASTWEEK["days"],
}

EVAL_SETS = {
    "aug_lastweek": MIAMI_AUG_LASTWEEK,
    "miami_aug_lastweek": MIAMI_AUG_LASTWEEK,
    "miami_aug_lastweek_swap27": MIAMI_AUG_LASTWEEK_SWAP27,
}


def _set_env_defaults() -> None:
    os.environ.setdefault("HF_HOME", "/mnt/ssd2/songze/.hf_cache")
    os.environ.setdefault("RL_W_ENERGY", "3.0")
    os.environ.setdefault("RL_FORECAST_CSV", DEFAULT_FORECAST_CSV)
    os.environ.setdefault("ASIM_ENABLE_THINKING", "1")
    os.environ.setdefault("ASIM_ENABLE_PMV_TOOL", "1")
    os.environ.setdefault("ASIM_ENABLE_PMV_RANGE_TOOL", "1")
    os.environ.setdefault("ASIM_MAX_TOOL_CALLS", "30")
    os.environ.setdefault("ASIM_TOOL_FORMAT", "xml")
    os.environ.setdefault("ASIM_ENABLE_FALLBACK_RESCUE", "1")
    os.environ.setdefault("ASIM_THINKING_GUARD", "0")


def _append_legacy_site_packages() -> None:
    shared = (
        PROJECT_ROOT
        / ".venv"
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    if shared.exists() and str(shared) not in sys.path:
        sys.path.append(str(shared))


class SetpointOnlyProxy:
    """Adapter expected by MiamiGRPOBandit._rollout_workday_with_knot_planner."""

    def __init__(self, planner: Any):
        self.planner = planner

    def plan_knot(
        self,
        *,
        block_index: int,
        knot_index: int,
        block_start: Any,
        block_end: Any,
        mode: str = "setpoint_only",
        observation: dict[str, dict[str, Any]] | None = None,
        wallclock: Any = None,
    ) -> dict[str, Any]:
        result = self.planner.plan_knot_setpoint_only(
            block_index=block_index,
            knot_index=knot_index,
            block_start=block_start,
            block_end=block_end,
            observation=observation,
            wallclock=wallclock,
            setpoint_exploration_hint=None,
        )
        result["mode"] = "setpoint_only"
        result["mode_source"] = "setpoint_only"
        return result


def _resolve_checkpoint(checkpoint: str, run_dir: Path) -> Path:
    value = Path(checkpoint).expanduser()
    if value.exists():
        return value.resolve()
    candidate = (run_dir / checkpoint).expanduser()
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"checkpoint not found: {checkpoint} or {candidate}")


def _default_output(eval_set: str, checkpoint: Path) -> Path:
    return (
        PROJECT_ROOT
        / "result/comparisons"
        / f"eval_grpo_10min_{eval_set}_{checkpoint.name}.json"
    )


def _sum_phys(trace: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key, 0.0)) for row in trace))


def _compact_knot_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_index": int(plan.get("block_index", -1)),
        "knot_index": int(plan.get("knot_index", -1)),
        "mode": str(plan.get("mode", "setpoint_only")),
        "mode_source": str(plan.get("mode_source", "setpoint_only")),
        "setpoints": plan.get("knot") or plan.get("setpoints") or {},
        "raw_output": plan.get("raw_output"),
    }


def _build_backend(args: argparse.Namespace, tokenizer: Any) -> Any:
    if args.backend == "vllm":
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from llm_setpoint_planner_vllm import VLLMQwen35Backend

        print(
            f"[LOAD] vLLM model={args.model_name_or_path} tp={args.vllm_tp} "
            f"gpu_mem={args.vllm_gpu_mem_util} ckpt={args.checkpoint}",
            flush=True,
        )
        engine = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=int(args.vllm_tp),
            dtype=args.torch_dtype,
            gpu_memory_utilization=float(args.vllm_gpu_mem_util),
            enable_prefix_caching=True,
            max_model_len=int(args.max_model_len),
            enforce_eager=False,
            enable_lora=True,
            max_lora_rank=int(args.max_lora_rank),
            max_loras=1,
        )
        lora_request = LoRARequest("eval_checkpoint", 1, str(args.checkpoint))
        return VLLMQwen35Backend(
            llm_engine=engine,
            tokenizer=tokenizer,
            model_name=args.model_name_or_path,
            max_output_tokens=int(args.max_output_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            repetition_penalty=float(args.repetition_penalty),
            lora_request=lora_request,
            seed=int(args.seed),
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from llm_setpoint_planner_qwen35 import Qwen35TransformersSamplingBackend

    print(
        f"[LOAD] transformers model={args.model_name_or_path} ckpt={args.checkpoint}",
        flush=True,
    )
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = PeftModel.from_pretrained(model, str(args.checkpoint), is_trainable=False)
    model.eval()
    return Qwen35TransformersSamplingBackend(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name_or_path,
        max_output_tokens=int(args.max_output_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        repetition_penalty=float(args.repetition_penalty),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "eval_set_pos",
        nargs="?",
        default=None,
        help="Backward-compatible positional eval set, e.g. aug_lastweek.",
    )
    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help="Backward-compatible checkpoint name/path, e.g. checkpoint-8.",
    )
    parser.add_argument("--eval-set", choices=sorted(EVAL_SETS), default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument(
        "--dates",
        type=str,
        default=None,
        help="Comma-separated dates from the eval set, e.g. 2025-08-26,2025-08-27.",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--building-idf", type=Path, default=None)
    parser.add_argument("--weather-epw", type=Path, default=None)
    parser.add_argument("--model-name-or-path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--vllm-tp", type=int, default=2)
    parser.add_argument("--vllm-gpu-mem-util", type=float, default=0.45)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--baseline-setpoint", type=float, default=23.0)
    parser.add_argument("--no-traces", action="store_true")
    args = parser.parse_args()

    args.eval_set = args.eval_set or args.eval_set_pos or "aug_lastweek"
    if args.eval_set not in EVAL_SETS:
        choices = ", ".join(sorted(EVAL_SETS))
        raise SystemExit(f"unknown eval set {args.eval_set!r}; choices: {choices}")
    checkpoint = args.checkpoint or args.checkpoint_pos or DEFAULT_CHECKPOINT_NAME
    args.checkpoint = _resolve_checkpoint(checkpoint, args.run_dir.expanduser())
    args.output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output(args.eval_set, args.checkpoint)
    )
    return args


def main() -> None:
    _set_env_defaults()
    _append_legacy_site_packages()
    args = parse_args()
    eval_set = EVAL_SETS[args.eval_set]
    if args.date is not None and args.dates is not None:
        raise ValueError("use either --date or --dates, not both")
    if eval_set.get("forecast_csv"):
        os.environ["RL_FORECAST_CSV"] = str(eval_set["forecast_csv"])

    from transformers import AutoTokenizer
    from grpo_miami_bandit import MiamiGRPOBandit
    from llm_setpoint_planner import PlannerConstraints
    from llm_setpoint_planner_unified import UnifiedBlockPlanner

    MiamiGRPOBandit.STEP_MINUTES = 10
    MiamiGRPOBandit.KNOT_ENV_STEPS = 1

    days = list(eval_set["days"])
    if args.dates is not None:
        selected_dates = [date.strip() for date in args.dates.split(",") if date.strip()]
        available_dates = {day["date"] for day in days}
        missing_dates = [date for date in selected_dates if date not in available_dates]
        if missing_dates:
            raise ValueError(
                f"dates {missing_dates} are not in eval set {args.eval_set}"
            )
        selected = set(selected_dates)
        days = [day for day in days if day["date"] in selected]
    elif args.date is not None:
        days = [day for day in days if day["date"] == args.date]
        if not days:
            raise ValueError(f"date {args.date} is not in eval set {args.eval_set}")
    if args.max_days is not None:
        days = days[: max(0, int(args.max_days))]
    if not days:
        raise ValueError("no eval days selected")

    building_idf = (args.building_idf or eval_set["idf"]).expanduser().resolve()
    weather_epw = (args.weather_epw or eval_set["epw"]).expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[EVAL] checkpoint={args.checkpoint}", flush=True)
    print(f"[EVAL] eval_set={args.eval_set} days={[d['date'] for d in days]}", flush=True)
    print("[EVAL] control_window=07:00-19:00 step_minutes=10 expected_steps=72", flush=True)
    print(f"[EVAL] building_idf={building_idf}", flush=True)
    print(f"[EVAL] weather_epw={weather_epw}", flush=True)
    print(f"[EVAL] forecast_csv={os.environ.get('RL_FORECAST_CSV')}", flush=True)
    print(f"[EVAL] output={args.output}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backend = _build_backend(args, tokenizer)

    bandit = MiamiGRPOBandit(
        include_forecast=True,
        control_window_start="07:00",
        control_window_end="19:00",
        weekday_only=True,
        request_mode="step_action",
        building_path=building_idf,
        weather_path=weather_epw,
        fallback_setpoint_low_occ_c=30.0,
        fallback_setpoint_high_occ_c=23.5,
        fallback_occ_low_threshold=0.15,
        fallback_occ_high_threshold=0.5,
    )
    planner = UnifiedBlockPlanner(
        backend,
        constraints=PlannerConstraints(
            min_setpoint_c=20.0,
            max_setpoint_c=30.0,
            max_delta_per_step_c=2.0,
            fallback_setpoint_c=30.0,
            quantization_c=0.1,
            fallback_setpoint_low_occ_c=30.0,
            fallback_setpoint_high_occ_c=23.5,
            fallback_occ_low_threshold=0.15,
            fallback_occ_high_threshold=0.5,
        ),
        zone_ids=bandit.zone_ids,
        max_generation_attempts=2,
    )
    proxy = SetpointOnlyProxy(planner)
    baseline_action = {
        zone_id: {"thermostat": float(args.baseline_setpoint)}
        for zone_id in bandit.zone_ids
    }

    trace_dir = args.output.with_suffix("")
    if not args.no_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for day in days:
        planner.clear_block_results()
        planner.clear_knot_results()
        t0 = time.time()
        print(f"[DAY] {day['date']} skip={day['skip_valid_steps']} starting baseline", flush=True)
        baseline = bandit._rollout_baseline_full_day_blocks(
            skip_valid_steps=int(day["skip_valid_steps"]),
            baseline_action=baseline_action,
        )
        print(f"[DAY] {day['date']} starting checkpoint rollout", flush=True)
        rollout = bandit._rollout_workday_with_knot_planner(
            skip_valid_steps=int(day["skip_valid_steps"]),
            planner=proxy,
            mode="setpoint_only",
        )
        elapsed = time.time() - t0
        cand_trace = rollout.get("reward_trace", [])
        bl_trace = baseline.get("reward_trace", [])
        day_return = float(rollout["total_reward"])
        baseline_return = float(baseline["total_reward"])
        row = {
            "method": "qwen35_v15_setpoint_only",
            "checkpoint": str(args.checkpoint),
            "date": day["date"],
            "skip_valid_steps": int(day["skip_valid_steps"]),
            "target_date": rollout.get("target_date"),
            "day_return": day_return,
            "baseline_day_return": baseline_return,
            "relative_day_return": day_return - baseline_return,
            "elapsed_s": round(elapsed, 1),
            "control_steps": int(rollout.get("control_steps_applied", 0)),
            "n_knots": len(rollout.get("knot_plans", [])),
            "facility_kwh": round(_sum_phys(cand_trace, "facility_kwh"), 2),
            "hvac_kwh": round(_sum_phys(cand_trace, "hvac_kwh"), 2),
            "pv_kwh": round(_sum_phys(cand_trace, "pv_kwh"), 2),
            "net_grid_kwh": round(_sum_phys(cand_trace, "net_grid_kwh"), 2),
            "pmv_violation_sum": round(_sum_phys(cand_trace, "total_pmv_violation"), 4),
            "baseline_hvac_kwh": round(_sum_phys(bl_trace, "hvac_kwh"), 2),
            "baseline_pmv_violation_sum": round(
                _sum_phys(bl_trace, "total_pmv_violation"), 4
            ),
        }
        rows.append(row)

        if not args.no_traces:
            trace_path = trace_dir / f"{day['date']}.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "row": row,
                        "baseline_reward_trace": bl_trace,
                        "candidate_reward_trace": cand_trace,
                        "candidate_action_trace": rollout.get("action_trace", []),
                        "knot_plans": [
                            _compact_knot_plan(p)
                            for p in rollout.get("knot_plans", [])
                        ],
                    },
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            row["trace_path"] = str(trace_path)

        total = sum(r["relative_day_return"] for r in rows)
        summary = {
            "method": "qwen35_v15_setpoint_only",
            "eval_set": args.eval_set,
            "checkpoint": str(args.checkpoint),
            "control_window": "07:00-19:00",
            "step_minutes": 10,
            "expected_control_steps_per_day": 72,
            "building_idf": str(building_idf),
            "weather_epw": str(weather_epw),
            "forecast_csv": os.environ.get("RL_FORECAST_CSV"),
            "baseline_setpoint": float(args.baseline_setpoint),
            "rows": rows,
            "total_relative": total,
            "mean_relative": total / len(rows),
        }
        args.output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(
            f"[RESULT] {day['date']} rel={row['relative_day_return']:+.4f} "
            f"day={row['day_return']:+.4f} baseline={row['baseline_day_return']:+.4f} "
            f"steps={row['control_steps']} knots={row['n_knots']} "
            f"hvac={row['hvac_kwh']:.1f} bl_hvac={row['baseline_hvac_kwh']:.1f} "
            f"pmv={row['pmv_violation_sum']:.3f} elapsed={elapsed:.0f}s",
            flush=True,
        )

    print(
        f"[DONE] total={sum(r['relative_day_return'] for r in rows):+.4f} "
        f"mean={sum(r['relative_day_return'] for r in rows) / len(rows):+.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
