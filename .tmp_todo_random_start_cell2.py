from ray import air, tune
from ray.air import CheckpointConfig
from ray.air.integrations.wandb import WandbLoggerCallback
import os
from pathlib import Path
import ray

try:
    import wandb  # noqa: F401
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


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


WANDB_PROJECT = os.getenv("WANDB_PROJECT", "asim-houston-forecast-rl")
WANDB_GROUP = os.getenv("WANDB_GROUP", "forecast-vs-no-forecast")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "").strip() or None
WANDB_RUN_PREFIX = os.getenv("WANDB_RUN_PREFIX", "houston_aug2025")
WANDB_LOGIN_DETECTED = detect_wandb_login()
WANDB_TAGS_BASE = [
    "rllib",
    "ppo",
    "energyplus",
    "houston",
    "forecast-ablation",
]


if not ray.is_initialized():
    ray.init(runtime_env={"env_vars": {
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
        "WANDB_START_METHOD": "thread",
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_GROUP": WANDB_GROUP,
        "WANDB_RUN_PREFIX": WANDB_RUN_PREFIX,
        "WANDB_ENTITY": WANDB_ENTITY or "",
        "PYTHONWARNINGS": "ignore:.*CUBLAS_WORKSPACE_CONFIG.*:UserWarning",
    }})


def build_training_run_name(variant: str = "forecast_window") -> str:
    return f"{WANDB_RUN_PREFIX}_{variant}"


def build_wandb_callback(variant: str = "forecast_window") -> WandbLoggerCallback:
    kwargs = {
        "project": WANDB_PROJECT,
        "group": WANDB_GROUP,
        "name": build_training_run_name(variant),
        "job_type": "train",
        "tags": [*WANDB_TAGS_BASE, variant],
        "log_config": True,
        "reinit": "finish_previous",
        "settings": {
            "x_disable_stats": True,
            "x_disable_machine_info": True,
        },
    }
    if WANDB_ENTITY is not None:
        kwargs["entity"] = WANDB_ENTITY
    return WandbLoggerCallback(**kwargs)


def build_run_callbacks(variant: str = "forecast_window") -> list:
    if not WANDB_AVAILABLE:
        return []
    return [build_wandb_callback(variant)]


def build_tuner(
    variant: str = "forecast_window",
    stop_total_env_steps: int = TRAIN_TOTAL_ENV_STEPS_TARGET,
):
    stop_config = {"num_env_steps_sampled_lifetime": int(stop_total_env_steps)}
    return tune.Tuner(
        "PPO",
        param_space=get_config(variant=variant),
        tune_config=tune.TuneConfig(
            num_samples=1,
            metric="env_runners/episode_reward_mean",
            mode="max",
        ),
        run_config=air.RunConfig(
            name=build_training_run_name(variant),
            storage_path=str((RESULT_DIR / "ray_tune").resolve()),
            stop=stop_config,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_frequency=1,
                checkpoint_at_end=True,
                checkpoint_score_attribute="env_runners/episode_reward_mean",
                checkpoint_score_order="max",
            ),
            verbose=2,
            callbacks=build_run_callbacks(variant),
        ),
    )


WANDB_CONFIG_SUMMARY = {
    "available": WANDB_AVAILABLE,
    "login_detected": WANDB_LOGIN_DETECTED,
    "callbacks_enabled": WANDB_AVAILABLE,
    "project": WANDB_PROJECT,
    "group": WANDB_GROUP,
    "entity": WANDB_ENTITY,
    "run_prefix": WANDB_RUN_PREFIX,
    "episode_steps": EPISODE_STEPS,
    "target_episodes": TRAIN_EPISODES_TARGET,
    "target_total_env_steps": TRAIN_TOTAL_ENV_STEPS_TARGET,
}


FORECAST_TUNER = build_tuner("forecast_window")
NO_FORECAST_TUNER = build_tuner("no_forecast_window")
WANDB_CONFIG_SUMMARY
