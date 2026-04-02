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
    return parser.parse_args()


CANDIDATE_MODES = ["comfort", "balanced", "energy_saving", "precooling"]


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

    # --- Reference model log-probs (KL anchor) ---
    ref_logprobs = None
    has_adapters = hasattr(model, "disable_adapter_layers")
    if kl_beta > 0 and has_adapters:
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

    # --- Policy gradient loss ---
    trajectory_logprob = completion_logprobs.sum()
    pg_loss = -(advantage_value * trajectory_logprob) / normalizer

    # --- KL penalty (Schulman approximation: exp(r) - r - 1, always >= 0) ---
    kl_value = 0.0
    if ref_logprobs is not None:
        log_ratio = completion_logprobs - ref_logprobs
        kl_per_token = torch.exp(log_ratio) - log_ratio - 1.0
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

            # --- Get baseline for the full day (cached) ---
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
                max_parallel = 2 if block_index >= 10 else len(CANDIDATE_MODES)
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

                # --- Compute block-level advantages ---
                rewards_tensor = torch.tensor(block_rewards, dtype=torch.float32, device=device)
                block_reward_std = float(rewards_tensor.std(unbiased=False).item()) if len(block_rewards) > 1 else 0.0
                block_advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std(unbiased=False) + 1e-4)

                # --- Backward pass: accumulate gradient over all knots of all candidates ---
                block_token_counts = []
                block_loss_value = 0.0
                total_knots_in_block = sum(len(kp) for kp in block_knot_plans)

                if float(torch.max(torch.abs(block_advantages)).item()) >= 1e-8:
                    optimizer.zero_grad(set_to_none=True)
                    model.train()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    block_kl_value = 0.0
                    for cand_idx, (knot_plans, advantage_val) in enumerate(zip(block_knot_plans, block_advantages)):
                        cand_tc = 0
                        cand_logprob = 0.0
                        cand_kl = 0.0
                        divisor = max(len(CANDIDATE_MODES) * max(len(knot_plans), 1), 1)
                        for knot_plan in knot_plans:
                            logprob, tc, kl_val = _accumulate_block_gradient(
                                model=model,
                                tokenizer=tokenizer,
                                backend=backend,
                                block_plan=knot_plan,
                                advantage=float(advantage_val.item()),
                                block_divisor=divisor,
                                kl_beta=float(args.kl_beta),
                            )
                            cand_tc += tc
                            cand_logprob += logprob
                            cand_kl += kl_val
                        block_token_counts.append(cand_tc)
                        block_loss_value += float((-float(advantage_val.item()) * cand_logprob) / max(divisor, 1))
                        block_kl_value += cand_kl

                    block_grad_norm = _grad_norm(model)
                    optimizer.step()
                    total_block_grad_updates += 1
                    model.eval()

                    _write_phase(phase_handle, step_index=step_index, phase="block_optimizer_step",
                                 block_index=block_index, grad_norm=block_grad_norm,
                                 block_reward_std=block_reward_std, block_loss=block_loss_value,
                                 block_kl=block_kl_value)
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
                    )
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
                    }
                    for br in day_block_results
                ]
                reflection = block_planner.generate_day_reflection(
                    date=str(baseline_result.get("target_date", f"step{step_index}")),
                    block_results=reflection_block_results,
                    total_reward=total_winner_reward + sum(baseline_block_rewards),
                    baseline_reward=sum(baseline_block_rewards),
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

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
