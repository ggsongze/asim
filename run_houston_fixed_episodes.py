#!/usr/bin/env python3

import csv
import json
import math
import os
import warnings
from numbers import Number
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig


warnings.filterwarnings("error", message=".*Casting input x to numpy array.*")

PROJECT_ROOT = Path(__file__).resolve().parent
TMP_CELL_PREFIX = os.getenv("RL_TMP_CELL_PREFIX", ".tmp_todo_random_start")


def load_notebook_namespace() -> dict:
    namespace: dict = {}
    for path in (
        PROJECT_ROOT / f"{TMP_CELL_PREFIX}_cell0.py",
        PROJECT_ROOT / f"{TMP_CELL_PREFIX}_cell1.py",
    ):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


def detect_wandb_login() -> bool:
    if os.getenv("WANDB_API_KEY"):
        return True
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return False
    try:
        text = netrc_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "api.wandb.ai" in text or "machine wandb.ai" in text


def build_run_name(prefix: str, variant: str, episode_steps: int, target_episodes: int) -> str:
    return f"{prefix}_ep{episode_steps}_x{target_episodes}_{variant}_manual"


def write_history_csv(path: Path, history: list[dict]) -> None:
    fieldnames = [
        "iteration",
        "episodes_this_iter",
        "episodes_completed",
        "episode_len_mean",
        "episode_reward_mean",
        "num_env_steps_sampled_lifetime",
        "time_total_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_episode_history_csv(path: Path, episode_history: list[dict]) -> None:
    fieldnames = [
        "episode_index",
        "training_iteration",
        "episode_reward",
        "episode_length",
        "episodes_this_iter",
        "episodes_completed",
        "num_env_steps_sampled_lifetime",
        "time_total_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_history)


def flatten_numeric_metrics(data, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            metrics.update(flatten_numeric_metrics(value, child_prefix))
        return metrics

    if isinstance(data, Number) and not isinstance(data, bool):
        value = float(data)
        if math.isfinite(value):
            metrics[prefix] = value
    return metrics


def main() -> int:
    episode_steps = max(int(os.getenv("RL_EPISODE_STEPS", "5000")), 1)
    target_episodes = max(int(os.getenv("RL_TRAIN_EPISODES", "10")), 1)
    variant = os.getenv("RL_VARIANT", "forecast_window").strip() or "forecast_window"
    num_gpus = max(float(os.getenv("RL_NUM_GPUS", "0")), 0)
    minibatch_size = max(int(os.getenv("RL_MINIBATCH_SIZE", "1000")), 1)
    num_epochs = max(int(os.getenv("RL_NUM_EPOCHS", "1")), 1)
    learning_rate = float(os.getenv("RL_LR", "2e-5"))
    wandb_project = os.getenv("WANDB_PROJECT", "asim-houston-forecast-rl")
    wandb_group = os.getenv("WANDB_GROUP", "fixed-episode-train")
    wandb_entity = os.getenv("WANDB_ENTITY", "").strip() or None
    wandb_run_prefix = os.getenv("WANDB_RUN_PREFIX", "houston_aug2025")
    run_name = build_run_name(wandb_run_prefix, variant, episode_steps, target_episodes)

    os.environ["RL_EPISODE_STEPS"] = str(episode_steps)
    os.environ["RL_TRAIN_EPISODES"] = str(target_episodes)
    os.environ["RL_NUM_ENV_RUNNERS"] = "0"
    os.environ["RL_NUM_GPUS"] = str(num_gpus)
    os.environ["RL_REPORT_SUBDIR"] = f"{run_name}_energyplus"

    result_dir = PROJECT_ROOT / "result" / "manual_train" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.json"
    history_json_path = result_dir / "history.json"
    history_csv_path = result_dir / "history.csv"
    episode_history_json_path = result_dir / "episode_history.json"
    episode_history_csv_path = result_dir / "episode_history.csv"

    ns = load_notebook_namespace()
    env_variants = ns["ENV_VARIANTS"]
    if variant not in env_variants:
        raise ValueError(f"Unknown RL_VARIANT: {variant}")
    notebook_get_config = ns.get("get_config")

    wandb_run = None
    wandb_enabled = False
    if detect_wandb_login():
        try:
            import wandb

            wandb_kwargs = {
                "project": wandb_project,
                "group": wandb_group,
                "name": run_name,
                "job_type": "train",
                "config": {
                    "variant": variant,
                    "episode_steps": episode_steps,
                    "target_episodes": target_episodes,
                    "num_env_runners": 0,
                    "batch_mode": "complete_episodes",
                    "rollout_fragment_length": episode_steps,
                    "train_batch_size": episode_steps,
                    "minibatch_size": min(minibatch_size, episode_steps),
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                },
                "reinit": "finish_previous",
                "settings": wandb.Settings(
                    x_disable_stats=True,
                    x_disable_machine_info=True,
                ),
            }
            if wandb_entity is not None:
                wandb_kwargs["entity"] = wandb_entity
            wandb_run = wandb.init(**wandb_kwargs)
            wandb_run.define_metric("episode_index")
            for metric_name in (
                "episode_reward",
                "episode_length",
                "episode_env_steps_lifetime",
                "episode_time_total_s",
            ):
                wandb_run.define_metric(metric_name, step_metric="episode_index")
            wandb_run.define_metric("training_iteration")
            wandb_run.define_metric("iteration_episode_reward_mean", step_metric="training_iteration")
            wandb_run.define_metric("info/*", step_metric="training_iteration")
            wandb_run.define_metric("env_runners/*", step_metric="training_iteration")
            wandb_enabled = True
        except Exception:
            wandb_run = None
            wandb_enabled = False

    summary = {
        "status": "started",
        "tmp_cell_prefix": TMP_CELL_PREFIX,
        "variant": variant,
        "episode_steps": episode_steps,
        "target_episodes": target_episodes,
        "inherits_notebook_training_config": callable(notebook_get_config),
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
        "wandb_group": wandb_group,
        "wandb_run_name": run_name,
        "report_subdir": os.environ["RL_REPORT_SUBDIR"],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    history: list[dict] = []
    episode_history: list[dict] = []
    checkpoint_path = None
    algo = None
    wandb_log_step = 0
    try:
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2)
        if callable(notebook_get_config):
            base_config = PPOConfig().update_from_dict(notebook_get_config(variant))
        else:
            base_config = PPOConfig().environment(env_variants[variant]["env_cls"])

        config = (
            base_config
            .environment(env_variants[variant]["env_cls"])
            .env_runners(
                enable_connectors=False,
                num_env_runners=0,
                create_env_on_local_worker=True,
                rollout_fragment_length=episode_steps,
                batch_mode="complete_episodes",
                sample_timeout_s=300,
            )
            .training(
                lr=learning_rate,
                train_batch_size=episode_steps,
                minibatch_size=min(minibatch_size, episode_steps),
                num_epochs=num_epochs,
            )
            .resources(num_gpus=num_gpus)
            .debugging(seed=1229)
        )
        algo = config.build()

        episodes_completed = 0
        iteration = 0
        best_reward = float("-inf")
        best_checkpoint_dir = result_dir / "checkpoint_best"
        while episodes_completed < target_episodes:
            iteration += 1
            result = algo.train()
            env_runners = result.get("env_runners", {})
            episodes_this_iter = int(env_runners.get("episodes_this_iter") or 0)
            episodes_completed += episodes_this_iter
            hist_stats = env_runners.get("hist_stats", {})
            episode_rewards = list(hist_stats.get("episode_reward") or [])
            episode_lengths = list(hist_stats.get("episode_lengths") or [])

            record = {
                "iteration": iteration,
                "episodes_this_iter": episodes_this_iter,
                "episodes_completed": episodes_completed,
                "episode_len_mean": env_runners.get("episode_len_mean"),
                "episode_reward_mean": env_runners.get("episode_reward_mean"),
                "num_env_steps_sampled_lifetime": result.get("num_env_steps_sampled_lifetime"),
                "time_total_s": result.get("time_total_s"),
            }
            history.append(record)

            history_json_path.write_text(
                json.dumps(history, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            write_history_csv(history_csv_path, history)

            new_episode_rewards = episode_rewards[-episodes_this_iter:] if episodes_this_iter > 0 else []
            new_episode_lengths = episode_lengths[-episodes_this_iter:] if episodes_this_iter > 0 else []
            first_episode_index = episodes_completed - episodes_this_iter + 1
            for offset in range(episodes_this_iter):
                episode_record = {
                    "episode_index": first_episode_index + offset,
                    "training_iteration": iteration,
                    "episode_reward": (
                        float(new_episode_rewards[offset])
                        if offset < len(new_episode_rewards)
                        else None
                    ),
                    "episode_length": (
                        int(new_episode_lengths[offset])
                        if offset < len(new_episode_lengths)
                        else None
                    ),
                    "episodes_this_iter": episodes_this_iter,
                    "episodes_completed": first_episode_index + offset,
                    "num_env_steps_sampled_lifetime": result.get("num_env_steps_sampled_lifetime"),
                    "time_total_s": result.get("time_total_s"),
                }
                episode_history.append(episode_record)
                if wandb_run is not None:
                    wandb_log_step += 1
                    wandb_run.log(
                        {
                            "episode_index": episode_record["episode_index"],
                            "episode_reward": episode_record["episode_reward"],
                            "episode_length": episode_record["episode_length"],
                            "episode_env_steps_lifetime": episode_record["num_env_steps_sampled_lifetime"],
                            "episode_time_total_s": episode_record["time_total_s"],
                            "training_iteration": iteration,
                            "iteration_episode_reward_mean": record["episode_reward_mean"],
                        },
                        step=wandb_log_step,
                    )

            if wandb_run is not None:
                iteration_log = {
                    "training_iteration": iteration,
                    "iteration_episode_reward_mean": record["episode_reward_mean"],
                    "num_env_steps_sampled_lifetime": result.get("num_env_steps_sampled_lifetime"),
                    "env_runners/episode_reward_mean": env_runners.get("episode_reward_mean"),
                    "env_runners/episode_len_mean": env_runners.get("episode_len_mean"),
                }
                iteration_log.update(flatten_numeric_metrics(result.get("info", {}), prefix="info"))
                wandb_log_step += 1
                wandb_run.log(iteration_log, step=wandb_log_step)

            episode_history_json_path.write_text(
                json.dumps(episode_history, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            write_episode_history_csv(episode_history_csv_path, episode_history)

            # Save best checkpoint by episode_reward_mean
            current_reward = record.get("episode_reward_mean")
            if current_reward is not None and current_reward > best_reward:
                best_reward = current_reward
                best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                algo.save(best_checkpoint_dir)
                summary["best_reward"] = best_reward
                summary["best_checkpoint_episode"] = episodes_completed
                summary["best_checkpoint_path"] = str(best_checkpoint_dir)

            summary.update(
                {
                    "status": "running",
                    "iterations_completed": iteration,
                    "episodes_completed": episodes_completed,
                    "last_record": record,
                }
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        checkpoint_dir = result_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_result = algo.save(checkpoint_dir)
        checkpoint = getattr(checkpoint_result, "checkpoint", None)
        checkpoint_path = getattr(checkpoint, "path", None) or str(checkpoint_dir)
        summary.update(
            {
                "status": "ok",
                "iterations_completed": iteration,
                "episodes_completed": episodes_completed,
                "checkpoint_path": checkpoint_path,
                "history_json": str(history_json_path),
                "history_csv": str(history_csv_path),
                "episode_history_json": str(episode_history_json_path),
                "episode_history_csv": str(episode_history_csv_path),
            }
        )
    except Exception as exc:
        summary.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "history_json": str(history_json_path),
                "history_csv": str(history_csv_path),
                "episode_history_json": str(episode_history_json_path),
                "episode_history_csv": str(episode_history_csv_path),
            }
        )
        raise
    finally:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        if wandb_run is not None:
            if checkpoint_path is not None:
                wandb_run.summary["checkpoint_path"] = checkpoint_path
            wandb_run.summary["status"] = summary["status"]
            wandb_run.finish()
        if algo is not None:
            algo.stop()
        if ray.is_initialized():
            ray.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
