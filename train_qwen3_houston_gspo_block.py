#!/usr/bin/env python3
"""Block-based 3h GSPO trainer for Houston Qwen step-action control.

Each workday is split into 4 forecast-aligned 3h blocks (07-10, 10-13, 13-16, 16-19).
For each block, 3 candidates (comfort / balanced / energy_saving) are generated.
Block-level grouped reward drives GSPO-style policy gradient updates.
Winner actions are replayed to establish correct initial state for subsequent blocks.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/home/AD/user/lab/asim")
SHARED_SITE_PACKAGES = PROJECT_ROOT / ".venv" / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if SHARED_SITE_PACKAGES.exists() and str(SHARED_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(SHARED_SITE_PACKAGES))

import numpy as np

from gspo_houston_bandit import RESULT_DIR, HoustonGSPOBandit, _plainify
from llm_setpoint_planner import (
    BlockPlanner,
    PlannerConstraints,
    PlannerRequest,
    TransformersSamplingBackend,
)


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
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--kl-beta", type=float, default=0.1,
                        help="KL penalty coefficient for GRPO (0 to disable)")
    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a PEFT adapter checkpoint to resume training from")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--wandb-project", type=str, default="asim-houston-grpo")
    parser.add_argument("--wandb-group", type=str, default="block-rolling-grpo")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="WandB run name. Defaults to output-dir basename.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--building-idf", type=str, default=None,
                        help="Path to IDF file (default: houston.idf)")
    parser.add_argument("--weather-epw", type=str, default=None,
                        help="Path to EPW file (default: houston EPW)")
    return parser.parse_args()


from llm_setpoint_planner import STRATEGY_TIERS, ALL_CANDIDATE_MODES
CANDIDATE_MODES = ALL_CANDIDATE_MODES  # 3 modes: comfort, balanced, energy_saving


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print(f"Resumed PEFT adapter from {args.resume_from}")
    elif args.use_peft:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.learning_rate),
    )

    bandit = HoustonGSPOBandit(
        include_forecast=True,
        control_window_start="06:30",
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
    # Save SFT adapter state as KL reference (frozen snapshot)
    import copy
    _sft_adapter_state = copy.deepcopy({
        k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad
    })
    print(f"Saved SFT adapter snapshot ({len(_sft_adapter_state)} params) as KL reference", flush=True)

    block_planner = BlockPlanner(
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
    baseline_action = {zone_id: {"thermostat": 24.0} for zone_id in bandit.zone_ids}

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

    with (
        metrics_path.open("w", encoding="utf-8") as metrics_handle,
        trajectory_path.open("w", encoding="utf-8") as trajectory_handle,
        phase_trace_path.open("w", encoding="utf-8") as phase_handle,
    ):
        for step_index in range(1, int(args.max_steps) + 1):
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

            model.eval()

            for block_index, (block_start_time, block_end_time) in enumerate(bandit.BLOCK_DEFINITIONS):
                replay_actions = list(winner_actions_history)
                block_candidates: list[dict[str, Any]] = []
                block_rewards: list[float] = []
                block_knot_plans: list[list[dict[str, Any]]] = []  # per-candidate list of knot plans

                _write_phase(phase_handle, step_index=step_index, phase="block_start",
                             block_index=block_index, block_start=str(block_start_time))

                # --- Parallel rollout of all candidate modes ---
                def _rollout_mode(mode: str) -> tuple[str, dict]:
                    for attempt in range(2):
                        try:
                            return mode, bandit._rollout_block_rolling(
                                skip_valid_steps=skip_valid_steps,
                                replay_actions=replay_actions,
                                baseline_action=baseline_action,
                                planner=block_planner,
                                block_index=block_index,
                                block_start=block_start_time,
                                block_end=block_end_time,
                                mode=mode,
                            )
                        except Exception as exc:
                            if attempt == 0:
                                print(f"  [retry] block {block_index} mode={mode} failed: {exc}, retrying...", flush=True)
                                time.sleep(2)
                            else:
                                raise

                mode_results: dict[str, dict] = {}
                # Use 4 workers for early blocks, 2 for late blocks (longer replay)
                max_parallel = min(len(CANDIDATE_MODES), 3)  # max 3 parallel EP instances
                with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                    futures = {executor.submit(_rollout_mode, m): m for m in CANDIDATE_MODES}
                    for future in as_completed(futures):
                        mode, candidate_result = future.result()
                        mode_results[mode] = candidate_result

                # --- Collect results in original mode order ---
                for mode in CANDIDATE_MODES:
                    candidate_result = mode_results[mode]
                    candidate_block_reward = candidate_result["block_reward"]
                    baseline_block_reward = baseline_block_rewards[block_index]
                    relative_block_reward = float(args.reward_scale) * (candidate_block_reward - baseline_block_reward)

                    knot_plans = candidate_result.get("knot_plans", [])
                    block_rewards.append(relative_block_reward)
                    block_knot_plans.append(knot_plans)
                    block_candidates.append({
                        "mode": mode,
                        "block_reward": candidate_block_reward,
                        "baseline_block_reward": baseline_block_reward,
                        "relative_block_reward": relative_block_reward,
                        "block_action_trace": candidate_result.get("block_action_trace", []),
                        "block_reward_trace": candidate_result.get("block_reward_trace", []),
                        "control_steps_applied": candidate_result.get("control_steps_applied", 0),
                        "target_date": candidate_result.get("target_date"),
                        "knot_count": len(knot_plans),
                    })

                    _write_phase(phase_handle, step_index=step_index, phase="block_candidate_done",
                                 block_index=block_index, mode=mode,
                                 relative_block_reward=relative_block_reward,
                                 knot_count=len(knot_plans))

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

                # --- Hierarchical two-level advantage ---
                # Build mode->reward and mode->knot_plans mapping
                mode_reward = {CANDIDATE_MODES[i]: block_rewards[i] for i in range(len(CANDIDATE_MODES))}
                mode_knot_plans = {CANDIDATE_MODES[i]: block_knot_plans[i] for i in range(len(CANDIDATE_MODES))}

                # Level 1: strategy-tier advantage (comfort vs balanced vs energy_saving)
                tier_best_reward = {}
                tier_best_mode = {}
                for tier_name, tier_modes in STRATEGY_TIERS.items():
                    tier_rewards = [mode_reward[m] for m in tier_modes]
                    best_idx = int(max(range(len(tier_rewards)), key=lambda i: tier_rewards[i]))
                    tier_best_reward[tier_name] = tier_rewards[best_idx]
                    tier_best_mode[tier_name] = tier_modes[best_idx]

                tier_rewards_list = [tier_best_reward[t] for t in STRATEGY_TIERS]
                tier_rewards_tensor = torch.tensor(tier_rewards_list, dtype=torch.float32, device=device)
                block_reward_std = float(tier_rewards_tensor.std(unbiased=False).item()) if len(tier_rewards_list) > 1 else 0.0
                # Baseline-anchored advantage: use 0 as anchor (rewards are already relative to baseline)
                # If all modes are worse than baseline (all negative), all get negative advantage
                tier_std = tier_rewards_tensor.std(unbiased=False)
                if tier_std > 1e-6:
                    tier_advantages = tier_rewards_tensor / (tier_std + 1e-4)
                else:
                    tier_advantages = torch.zeros_like(tier_rewards_tensor)

                # Level 2: within-tier sub-mode advantage (still group-normalized within tier)
                sub_advantages: dict[str, float] = {}
                for tier_name, tier_modes in STRATEGY_TIERS.items():
                    tier_rewards = torch.tensor([mode_reward[m] for m in tier_modes], dtype=torch.float32, device=device)
                    sub_std = float(tier_rewards.std(unbiased=False).item())
                    if sub_std > 1e-6:
                        sub_adv = (tier_rewards - tier_rewards.mean()) / (tier_rewards.std(unbiased=False) + 1e-4)
                        for m, a in zip(tier_modes, sub_adv):
                            sub_advantages[m] = float(a.item())
                    else:
                        for m in tier_modes:
                            sub_advantages[m] = 0.0

                # --- Backward pass: two-level gradient ---
                block_token_counts = []
                block_loss_value = 0.0
                block_kl_value = 0.0
                has_signal = (float(torch.max(torch.abs(tier_advantages)).item()) >= 1e-8 or
                              any(abs(v) >= 1e-8 for v in sub_advantages.values()))

                # Build per-mode step reward traces for per-knot partial return
                mode_step_rewards: dict[str, list[float]] = {}
                for cand in block_candidates:
                    trace = cand.get("block_reward_trace", [])
                    mode_step_rewards[cand["mode"]] = [float(e["reward"]) for e in trace]

                if has_signal:
                    optimizer.zero_grad(set_to_none=True)
                    model.train()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    total_modes = len(CANDIDATE_MODES)
                    for tier_idx, (tier_name, tier_modes) in enumerate(STRATEGY_TIERS.items()):
                        tier_adv = float(tier_advantages[tier_idx].item())
                        for mode in tier_modes:
                            sub_adv = sub_advantages.get(mode, 0.0)
                            combined_advantage = tier_adv + 0.5 * sub_adv
                            knot_plans = mode_knot_plans[mode]
                            step_rewards = mode_step_rewards.get(mode, [])
                            n_knots = len(knot_plans)
                            divisor = max(total_modes * max(n_knots, 1), 1)
                            cand_tc = 0

                            # Compute per-knot partial return (Monte Carlo within block)
                            knot_partial_returns = []
                            for k in range(n_knots):
                                # Sum of step rewards from knot k to end of block
                                partial = sum(step_rewards[k:]) if k < len(step_rewards) else 0.0
                                knot_partial_returns.append(partial)

                            # Normalize partial returns within this mode
                            if knot_partial_returns:
                                pr_mean = sum(knot_partial_returns) / len(knot_partial_returns)
                                pr_std = (sum((r - pr_mean)**2 for r in knot_partial_returns) / len(knot_partial_returns)) ** 0.5
                                if pr_std > 1e-6:
                                    knot_partial_advantages = [(r - pr_mean) / (pr_std + 1e-4) for r in knot_partial_returns]
                                else:
                                    knot_partial_advantages = [0.0] * len(knot_partial_returns)
                            else:
                                knot_partial_advantages = [0.0] * n_knots

                            for knot_idx, knot_plan in enumerate(knot_plans):
                                # Three-level advantage: tier + sub + per-knot
                                knot_adv = knot_partial_advantages[knot_idx] if knot_idx < len(knot_partial_advantages) else 0.0
                                total_advantage = combined_advantage + 0.3 * knot_adv
                                logprob, tc, kl_val = _accumulate_block_gradient(
                                    model=model,
                                    tokenizer=tokenizer,
                                    backend=backend,
                                    block_plan=knot_plan,
                                    advantage=total_advantage,
                                    block_divisor=divisor,
                                    kl_beta=float(args.kl_beta),
                                    sft_adapter_state=_sft_adapter_state,
                                )
                                cand_tc += tc
                                block_kl_value += kl_val
                            block_token_counts.append(cand_tc)

                    import torch
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    block_grad_norm = _grad_norm(model)
                    optimizer.step()
                    total_block_grad_updates += 1
                    model.eval()

                    _write_phase(phase_handle, step_index=step_index, phase="block_optimizer_step",
                                 block_index=block_index, grad_norm=block_grad_norm,
                                 block_reward_std=block_reward_std, block_loss=block_loss_value,
                                 block_kl=block_kl_value,
                                 tier_best_modes={t: tier_best_mode[t] for t in STRATEGY_TIERS})
                else:
                    block_grad_norm = 0.0
                    block_kl_value = 0.0
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

                day_block_results.append({
                    "block_index": block_index,
                    "block_start": str(block_start_time),
                    "block_end": str(block_end_time),
                    "winner_index": winner_idx,
                    "winner_mode": CANDIDATE_MODES[winner_idx],
                    "block_rewards": block_rewards,
                    "block_reward_std": block_reward_std,
                    "block_grad_norm": block_grad_norm,
                    "block_loss": block_loss_value,
                    "block_kl": block_kl_value,
                    "candidate_modes": CANDIDATE_MODES,
                    "token_counts": block_token_counts,
                    "knots_per_candidate": [len(kp) for kp in block_knot_plans],
                })

                _write_phase(phase_handle, step_index=step_index, phase="block_done",
                             block_index=block_index, winner_mode=CANDIDATE_MODES[winner_idx],
                             winner_reward=block_rewards[winner_idx])

                # --- Per-block reflection with zone PMV ---
                try:
                    # Extract zone PMV violations from winner candidate's knot plans
                    zone_pmv_lines = []
                    winner_cand = block_candidates[winner_idx] if winner_idx < len(block_candidates) else {}
                    winner_kp = block_knot_plans[winner_idx] if winner_idx < len(block_knot_plans) else []
                    # Get last knot's observation for end-of-block zone states
                    if winner_kp:
                        last_knot = winner_kp[-1]
                        last_user_prompt = last_knot.get("user_prompt", "")
                        # Parse zone PMV from knot user prompt (contains zone states)
                        import re as _re
                        for zone_id in bandit.zone_ids:
                            # Look for patterns like "temp=26.3C" and "PMV=+0.45"
                            zone_pattern = _re.search(
                                rf'{zone_id}.*?temp=([0-9.]+).*?PMV=([+-]?[0-9.]+)',
                                last_user_prompt
                            )
                            if zone_pattern:
                                temp = float(zone_pattern.group(1))
                                pmv = float(zone_pattern.group(2))
                                mode_target = CANDIDATE_MODES[winner_idx]
                                # Check if PMV is outside mode's target range
                                if mode_target == "comfort" and abs(pmv) > 0.2:
                                    zone_pmv_lines.append(f"  {zone_id}: PMV={pmv:+.2f} EXCEEDED comfort range (±0.2), temp={temp:.1f}°C")
                                elif mode_target == "balanced" and (pmv < -0.1 or pmv > 0.35):
                                    zone_pmv_lines.append(f"  {zone_id}: PMV={pmv:+.2f} EXCEEDED balanced range (0~+0.3), temp={temp:.1f}°C")
                                elif mode_target == "energy_saving" and pmv > 0.55:
                                    zone_pmv_lines.append(f"  {zone_id}: PMV={pmv:+.2f} EXCEEDED energy_saving limit (+0.5), temp={temp:.1f}°C")

                    zone_pmv_summary = "\n".join(zone_pmv_lines) if zone_pmv_lines else ""

                    block_planner.generate_block_reflection(
                        date=str(baseline_result.get("target_date", f"step{step_index}")),
                        block_index=block_index,
                        block_start=str(block_start_time),
                        block_end=str(block_end_time),
                        all_mode_rewards=dict(zip(CANDIDATE_MODES, block_rewards)),
                        winner_mode=CANDIDATE_MODES[winner_idx],
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

            # --- Aggregate metrics for this step ---
            total_winner_reward = sum(
                br["block_rewards"][br["winner_index"]] for br in day_block_results
            )

            # --- Day-level gradient pass on all winning knots ---
            day_grad_norm = 0.0
            day_kl_value = 0.0
            DAY_ADVANTAGE_SCALE = 0.3  # relative weight of day-level vs block-level signal
            if abs(total_winner_reward) > 1e-6 and day_winner_knot_plans:
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
                        kl_beta=float(args.kl_beta),
                        sft_adapter_state=_sft_adapter_state,
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                day_grad_norm = _grad_norm(model)
                optimizer.step()
                model.eval()
                _write_phase(phase_handle, step_index=step_index, phase="day_level_gradient",
                             day_advantage=day_advantage, day_grad_norm=day_grad_norm,
                             num_winning_knots=len(day_winner_knot_plans))

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
                "elapsed_s": time.time() - started_at,
            }
            metrics.append(metric_row)
            metrics_handle.write(json.dumps(metric_row, ensure_ascii=False) + "\n")
            metrics_handle.flush()

            _write_phase(phase_handle, step_index=step_index, phase="step_done",
                         total_winner_reward=total_winner_reward,
                         avg_block_reward_std=avg_block_reward_std)

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
            if int(args.save_steps) > 0 and step_index % int(args.save_steps) == 0:
                checkpoint_dir = args.output_dir / f"checkpoint-{step_index}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                _write_phase(phase_handle, step_index=step_index, phase="checkpoint_saved",
                             checkpoint_dir=str(checkpoint_dir))

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
        "learning_rate": float(args.learning_rate),
        "kl_beta": float(args.kl_beta),
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
