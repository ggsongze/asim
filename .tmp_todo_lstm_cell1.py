from ray.rllib.algorithms.ppo import PPOConfig
import os


REWARD_DESCRIPTION = {
    "type": "negative penalty",
    "formula": (
        "reward = -0.01 * (1.0 * net_building_energy_kwh + 50.0 * pmv_violation * occupancy)"
    ),
    "terms": [
        "net building electricity from Electricity:Facility - ElectricityProduced:Facility, clipped at zero and converted from J to kWh",
        "occupied PMV violation beyond +/-0.5",
    ],
}

FORECAST_WINDOW_DESCRIPTION = {
    "forecast_available": "whether a valid forecast issue time exists before the current wallclock",
    "forecast_temperature_6h": "next 1-6 hour outdoor dry-bulb temperature forecast",
    "forecast_humidity_6h": "next 1-6 hour outdoor relative humidity forecast",
    "forecast_precip_prob_6h": "next 1-6 hour precipitation probability forecast",
    "forecast_precip_6h": "next 1-6 hour precipitation forecast",
}

LSTM_MODEL_CONFIG = {
    "use_attention": False,
    "use_lstm": True,
    "max_seq_len": 64,
}


def describe_training_variant(variant: str = "forecast_window") -> dict[str, object]:
    spec = ENV_VARIANTS[variant]
    return {
        "variant": variant,
        "env_cls": spec["env_cls"].__name__,
        "reward": REWARD_DESCRIPTION,
        "observation_keys": list(spec["observation_keys"]),
        "forecast_observation": (
            FORECAST_WINDOW_DESCRIPTION if variant == "forecast_window" else {}
        ),
        "model": dict(LSTM_MODEL_CONFIG),
    }


def detect_num_gpus() -> int:
    override = os.getenv("RL_NUM_GPUS")
    if override is not None:
        try:
            return max(int(override), 0)
        except ValueError:
            return 0
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return max(int(torch.cuda.device_count()), 0)
    except Exception:
        return 0


NUM_GPUS = detect_num_gpus()
NUM_ENV_RUNNERS = max(int(os.getenv("RL_NUM_ENV_RUNNERS", "3")), 0)
ROLLOUT_FRAGMENT_LENGTH = max(int(os.getenv("RL_ROLLOUT_FRAGMENT_LENGTH", "256")), 1)
TRAIN_BATCH_SIZE = max(int(os.getenv("RL_TRAIN_BATCH_SIZE", str(EPISODE_STEPS))), 1)
MINIBATCH_SIZE = max(int(os.getenv("RL_MINIBATCH_SIZE", "1024")), 1)
NUM_EPOCHS = max(int(os.getenv("RL_NUM_EPOCHS", "2")), 1)
TRAIN_EPISODES_TARGET = max(int(os.getenv("RL_TRAIN_EPISODES", "10")), 1)
TRAIN_TOTAL_ENV_STEPS_TARGET = EPISODE_STEPS * TRAIN_EPISODES_TARGET


def get_config(variant: str = "forecast_window") -> dict:
    env_cls = ENV_VARIANTS[variant]["env_cls"]
    return (
        PPOConfig()
        .environment(env_cls)
        .env_runners(
            enable_connectors=False,
            batch_mode="truncate_episodes",
            sample_timeout_s=120,
            rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,
            num_env_runners=NUM_ENV_RUNNERS,
            create_env_on_local_worker=True,
        )
        .training(
            lr=2e-5,
            model=dict(LSTM_MODEL_CONFIG),
            train_batch_size=TRAIN_BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            minibatch_size=min(MINIBATCH_SIZE, TRAIN_BATCH_SIZE),
        )
        .resources(num_gpus=NUM_GPUS)
        .debugging(seed=1229)
        .to_dict()
    )


TRAINING_VARIANT_SUMMARY = {
    variant: describe_training_variant(variant)
    for variant in ENV_VARIANTS.keys()
}
TRAINING_VARIANT_SUMMARY["resource_config"] = {
    "num_gpus": NUM_GPUS,
    "rl_num_gpus_override": os.getenv("RL_NUM_GPUS"),
}
TRAINING_VARIANT_SUMMARY["sampling_config"] = {
    "episode_steps": EPISODE_STEPS,
    "target_episodes": TRAIN_EPISODES_TARGET,
    "target_total_env_steps": TRAIN_TOTAL_ENV_STEPS_TARGET,
    "num_env_runners": NUM_ENV_RUNNERS,
    "rollout_fragment_length": ROLLOUT_FRAGMENT_LENGTH,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "minibatch_size": min(MINIBATCH_SIZE, TRAIN_BATCH_SIZE),
    "num_epochs": NUM_EPOCHS,
}
TRAINING_VARIANT_SUMMARY
