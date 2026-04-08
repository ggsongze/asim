import os
from pathlib import Path
from typing import TypedDict

try:
    import ipynbname
except Exception:
    ipynbname = None

import numpy as _numpy_
import pandas as pd
import pythermalcomfort as pytc
import gymnasium.spaces as _gym_spaces_
from controllables.core import TemporaryUnavailableError
from controllables.core.tools.gymnasium import BoxSpace, DictSpace
from controllables.core.tools.gymnasium.spaces import (
    CompositeSpaceMapper as _GymCompositeSpaceMapper,
    MutableSpaceVariable as _GymMutableSpaceVariable,
    SpaceVariable as _GymSpaceVariable,
)
from controllables.core.tools.rllib import Env
from controllables.core.variables import MutableVariable
from controllables.energyplus import Actuator, OutputMeter, OutputVariable, System


def _coerce_box_value(space, value):
    if isinstance(space, BoxSpace):
        return _numpy_.asarray(value, dtype=space.dtype).reshape(space.shape)
    return value


def _coerce_space_value(space, value):
    if isinstance(space, BoxSpace):
        return _numpy_.asarray(value, dtype=space.dtype).reshape(space.shape)

    child_spaces = getattr(space, "spaces", None)
    if isinstance(child_spaces, dict):
        return {
            key: _coerce_space_value(child_space, value[key])
            for key, child_space in child_spaces.items()
        }
    if isinstance(child_spaces, tuple):
        return tuple(
            _coerce_space_value(child_space, child_value)
            for child_space, child_value in zip(child_spaces, value)
        )
    if isinstance(child_spaces, list):
        return [
            _coerce_space_value(child_space, child_value)
            for child_space, child_value in zip(child_spaces, value)
        ]
    return value


def _patched_box_contains(self, x):
    if not isinstance(x, _numpy_.ndarray):
        try:
            x = _numpy_.asarray(x, dtype=self.dtype).reshape(self.shape)
        except (ValueError, TypeError):
            return False

    return bool(
        _numpy_.can_cast(x.dtype, self.dtype)
        and x.shape == self.shape
        and _numpy_.all(x >= self.low)
        and _numpy_.all(x <= self.high)
    )


_GYM_MUTABLE_SPACE_SETTER = _GymMutableSpaceVariable.value.fset


def _patched_space_variable_getter(self):
    def getter(space):
        value = space.deref(self.__parent__).value
        return _coerce_box_value(space, value)

    return _GymCompositeSpaceMapper(getter)(self.space)


_SPACE_PATCHES_INSTALLED = False


def install_space_patches() -> None:
    global _SPACE_PATCHES_INSTALLED
    if _SPACE_PATCHES_INSTALLED:
        return

    _gym_spaces_.Box.contains = _patched_box_contains
    BoxSpace.contains = _patched_box_contains
    _GymSpaceVariable.value = property(_patched_space_variable_getter)
    _GymMutableSpaceVariable.value = _GymSpaceVariable.value.setter(_GYM_MUTABLE_SPACE_SETTER)
    _SPACE_PATCHES_INSTALLED = True


install_space_patches()


try:
    RUN_NAME = ipynbname.name() if ipynbname is not None else "TODO_compare_single_agent"
except Exception:
    RUN_NAME = "TODO_compare_single_agent"
PROJECT_ROOT = Path.cwd().resolve()
WEATHER_DIR = PROJECT_ROOT / "weather"
RESULT_DIR = PROJECT_ROOT / "result"
REPORTS_DIR = RESULT_DIR / "reports"
REPORT_SUBDIR = os.getenv("RL_REPORT_SUBDIR", "tmp_energyplus")
DRIVER_RESULT_DIR = str(RESULT_DIR.resolve())
for path in (WEATHER_DIR, RESULT_DIR, REPORTS_DIR):
    path.mkdir(parents=True, exist_ok=True)
FORECAST_CSV_PATH = WEATHER_DIR / os.getenv("RL_FORECAST_CSV", "houston_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6.csv")
FORECAST_HORIZON_HOURS = 6
EPISODE_STEPS = max(int(os.getenv("RL_EPISODE_STEPS", "5000")), 1)


ZONE_MAP = {
    "1FNW": "1FNW:OPENOFFICE",
    "1FNE": "1FNE:OPENOFFICE",
    "0FNW": "0FNW:OPENOFFICE",
    "0FNE": "0FNE:OPENOFFICE",
    "1FSW": "1FSW:OPENOFFICE",
    "1FSE": "1FSE:OPENOFFICE",
    "0FSW": "0FSW:OPENOFFICE",
    "0FSE": "0FSE:OPENOFFICE",
}


def joules_to_kwh(value) -> float:
    try:
        return float(value) / 3.6e6
    except Exception:
        return 0.0


def float32_array(values) -> _numpy_.ndarray:
    return _numpy_.asarray(values, dtype=_numpy_.float32)


class ForecastBundleReader:
    def __init__(self, csv_path: str | Path, horizon_hours: int = FORECAST_HORIZON_HOURS):
        self.csv_path = Path(csv_path)
        self.horizon_hours = int(horizon_hours)
        self._zero = float32_array([0.0] * self.horizon_hours)

        df = pd.read_csv(self.csv_path, parse_dates=["run_time"])
        if df.empty:
            self._times_ns = _numpy_.asarray([], dtype=_numpy_.int64)
            self._temperature = _numpy_.zeros((0, self.horizon_hours), dtype=_numpy_.float32)
            self._humidity = _numpy_.zeros((0, self.horizon_hours), dtype=_numpy_.float32)
            self._cloudcover = _numpy_.zeros((0, self.horizon_hours), dtype=_numpy_.float32)
            self._precip_prob = _numpy_.zeros((0, self.horizon_hours), dtype=_numpy_.float32)
            self._precip = _numpy_.zeros((0, self.horizon_hours), dtype=_numpy_.float32)
            self.missing_summary = {
                "rows_with_missing_used_fields_before": 0,
                "missing_used_cells_before": 0,
                "rows_with_missing_used_fields_after": 0,
                "missing_used_cells_after": 0,
            }
            return

        df["run_time"] = pd.to_datetime(df["run_time"], errors="coerce")
        df = df.dropna(subset=["run_time"]).sort_values("run_time").reset_index(drop=True)
        # Force nanosecond resolution so searchsorted uses the same unit as Timestamp.value
        # across pandas 1.x (datetime64[ns]) and pandas 3.x (which may yield datetime64[us]).
        self._times_ns = df["run_time"].to_numpy(dtype="datetime64[ns]").astype(_numpy_.int64)
        self._years = set(int(year) for year in df["run_time"].dt.year.unique().tolist())
        temperature_cols = [f"temperature_2m_t_plus_{lead}h" for lead in range(1, self.horizon_hours + 1)]
        humidity_cols = [f"relative_humidity_2m_t_plus_{lead}h" for lead in range(1, self.horizon_hours + 1)]
        cloudcover_cols = [f"cloudcover_t_plus_{lead}h" for lead in range(1, self.horizon_hours + 1)]
        precip_prob_cols = [f"precipitation_probability_t_plus_{lead}h" for lead in range(1, self.horizon_hours + 1)]
        precip_cols = [f"precipitation_t_plus_{lead}h" for lead in range(1, self.horizon_hours + 1)]

        used_cols = temperature_cols + humidity_cols + cloudcover_cols + precip_prob_cols + precip_cols
        used = df.loc[:, used_cols].apply(pd.to_numeric, errors="coerce")
        self.missing_summary = {
            "rows_with_missing_used_fields_before": int(used.isna().any(axis=1).sum()),
            "missing_used_cells_before": int(used.isna().sum().sum()),
        }
        if self.missing_summary["missing_used_cells_before"] > 0:
            # Fill sparse forecast gaps from neighboring issue times before exposing values to RL.
            used = used.ffill().bfill().fillna(0.0)
        self.missing_summary["rows_with_missing_used_fields_after"] = int(used.isna().any(axis=1).sum())
        self.missing_summary["missing_used_cells_after"] = int(used.isna().sum().sum())

        self._temperature = used.loc[:, temperature_cols].to_numpy(dtype=_numpy_.float32)
        self._humidity = used.loc[:, humidity_cols].to_numpy(dtype=_numpy_.float32)
        self._cloudcover = used.loc[:, cloudcover_cols].to_numpy(dtype=_numpy_.float32)
        self._precip_prob = used.loc[:, precip_prob_cols].to_numpy(dtype=_numpy_.float32)
        self._precip = used.loc[:, precip_cols].to_numpy(dtype=_numpy_.float32)

    def _to_local_naive_ns(self, value) -> int:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None or ts.tz is not None:
            ts = ts.tz_localize(None)
        return int(ts.value)

    def _zero_bundle(self) -> dict[str, _numpy_.ndarray | _numpy_.float32]:
        return {
            "available": _numpy_.float32(0.0),
            "temperature": self._zero.copy(),
            "humidity": self._zero.copy(),
            "cloudcover": self._zero.copy(),
            "precip_prob": self._zero.copy(),
            "precip": self._zero.copy(),
        }

    def get_bundle(self, value) -> dict[str, _numpy_.ndarray | _numpy_.float32]:
        if len(self._times_ns) == 0:
            return self._zero_bundle()

        ts = pd.Timestamp(value)
        if ts.tzinfo is not None or ts.tz is not None:
            ts = ts.tz_localize(None)
        if int(ts.year) not in self._years:
            return self._zero_bundle()

        t_ns = int(ts.value)
        idx = int(_numpy_.searchsorted(self._times_ns, t_ns, side="right") - 1)
        if idx < 0:
            return self._zero_bundle()

        return {
            "available": _numpy_.float32(1.0),
            "temperature": self._temperature[idx].copy(),
            "humidity": self._humidity[idx].copy(),
            "cloudcover": self._cloudcover[idx].copy(),
            "precip_prob": self._precip_prob[idx].copy(),
            "precip": self._precip[idx].copy(),
        }


try:
    forecast_reader = ForecastBundleReader(FORECAST_CSV_PATH)
except Exception as _forecast_load_err:
    print(f"WARNING: Failed to load forecast CSV: {_forecast_load_err}. Forecast will be unavailable.")
    forecast_reader = None
forecast_available = MutableVariable(_numpy_.float32(0.0))
forecast_temperature_6h = MutableVariable(float32_array([0.0] * FORECAST_HORIZON_HOURS))
forecast_humidity_6h = MutableVariable(float32_array([0.0] * FORECAST_HORIZON_HOURS))
forecast_cloudcover_6h = MutableVariable(float32_array([0.0] * FORECAST_HORIZON_HOURS))
forecast_precip_prob_6h = MutableVariable(float32_array([0.0] * FORECAST_HORIZON_HOURS))
forecast_precip_6h = MutableVariable(float32_array([0.0] * FORECAST_HORIZON_HOURS))


def update_forecast_observation(clock_value) -> None:
    if forecast_reader is None:
        return
    bundle = forecast_reader.get_bundle(clock_value)
    forecast_available.value = _numpy_.float32(bundle["available"])
    forecast_temperature_6h.value = float32_array(bundle["temperature"])
    forecast_humidity_6h.value = float32_array(bundle["humidity"])
    forecast_cloudcover_6h.value = float32_array(bundle["cloudcover"])
    forecast_precip_prob_6h.value = float32_array(bundle["precip_prob"])
    forecast_precip_6h.value = float32_array(bundle["precip"])


def prime_space_bindings(space, manager) -> None:
    ref = getattr(space, "__ref__", None)
    if ref is not None:
        space.deref(manager)

    child_spaces = getattr(space, "spaces", None)
    if isinstance(child_spaces, dict):
        for child in child_spaces.values():
            prime_space_bindings(child, manager)
    elif child_spaces is not None:
        for child in child_spaces:
            prime_space_bindings(child, manager)


def prime_reward_variables(system) -> None:
    for zone_key in ZONE_MAP.values():
        system[OutputVariable.Ref(
            type="Zone Mean Radiant Temperature",
            key=zone_key,
        )]
        system[OutputVariable.Ref(
            type="Zone Air Relative Humidity",
            key=zone_key,
        )]


def prime_env_bindings(env, system) -> None:
    # Request all bound variables before EnergyPlus starts so the first run can serve RL observations.
    prime_space_bindings(env.agent.action_space, system)
    prime_space_bindings(env.agent.observation_space, system)
    prime_reward_variables(system)


class RewardFunction:
    def __init__(
        self,
        zone_key: str,
        metab_rate=1.0,
        clothing=0.5,
        pmv_limit=0.5,
        w_comfort: float = 50.0,
    ):
        self.zone_key = zone_key
        self._metab_rate = _numpy_.asarray(metab_rate)
        self._clothing = _numpy_.asarray(clothing)
        self._pmv_limit = _numpy_.asarray(pmv_limit)
        self.w_comfort = float(w_comfort)

    class Inputs(TypedDict):
        office_occupancy: float
        temperature_drybulb: float
        airspeed: float

    def __call__(self, agent, inputs: Inputs) -> float:
        system = agent.system
        occ = float(inputs["office_occupancy"])

        tr = system[OutputVariable.Ref(
            type="Zone Mean Radiant Temperature",
            key=self.zone_key,
        )].value
        rh = system[OutputVariable.Ref(
            type="Zone Air Relative Humidity",
            key=self.zone_key,
        )].value

        pmv = pytc.models.pmv_ppd(
            float(inputs["temperature_drybulb"]),
            tr=tr,
            vr=pytc.utilities.v_relative(
                v=float(inputs.get("airspeed", 0.1)),
                met=self._metab_rate,
            ),
            rh=rh,
            met=self._metab_rate,
            clo=pytc.utilities.clo_dynamic(
                clo=self._clothing,
                met=self._metab_rate,
            ),
            limit_inputs=False,
        )["pmv"]
        pmv_violation = max(abs(pmv) - float(self._pmv_limit), 0.0)

        comfort_penalty = self.w_comfort * pmv_violation * occ
        if not _numpy_.isfinite(comfort_penalty):
            comfort_penalty = 0.0
        return float(comfort_penalty)


BASE_OBSERVATION_KEYS = (
    "temperature_drybulb",
    "temperature:radiant",
    "humidity",
    "energy_consumption",
    "energy_building",
    "occupancy",
    "PV",
    "outdoor_temp",
    "cloud_cover",
)

FORECAST_OBSERVATION_KEYS = (
    "forecast_available",
    "forecast_temperature_6h",
    "forecast_humidity_6h",
    "forecast_cloudcover_6h",
    "forecast_precip_prob_6h",
    "forecast_precip_6h",
)


def build_action_space(room_agent_ids: dict[str, str]) -> DictSpace:
    return DictSpace({
        zone_id: DictSpace({
            "thermostat": BoxSpace(
                low=20.0,
                high=30.0,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(
                Actuator.Ref(
                    type="Zone Temperature Control",
                    control_type="Cooling Setpoint",
                    key=zone_key,
                )
            ),
        })
        for zone_id, zone_key in room_agent_ids.items()
    })


def build_zone_observation_fields(zone_key: str, include_forecast: bool) -> dict[str, object]:
    fields = {
        "temperature_drybulb": BoxSpace(
            low=-_numpy_.inf,
            high=+_numpy_.inf,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(OutputVariable.Ref(
            type="Zone Mean Air Temperature",
            key=zone_key,
        )),
        "temperature:radiant": BoxSpace(
            low=-_numpy_.inf,
            high=+_numpy_.inf,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(OutputVariable.Ref(
            type="Zone Mean Radiant Temperature",
            key=zone_key,
        )),
        "humidity": BoxSpace(
            low=-_numpy_.inf,
            high=+_numpy_.inf,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(OutputVariable.Ref(
            type="Zone Air Relative Humidity",
            key=zone_key,
        )),
        "energy_consumption": BoxSpace(
            low=-_numpy_.inf,
            high=+_numpy_.inf,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(OutputMeter.Ref(type="Electricity:Facility")),
        "energy_building": BoxSpace(
            low=-_numpy_.inf,
            high=+_numpy_.inf,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(OutputMeter.Ref(type="Electricity:Building")),
        "occupancy": BoxSpace(
            low=-_numpy_.inf,
            high=+_numpy_.inf,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(OutputVariable.Ref(
            type="Schedule Value",
            key="Office_OpenOff_Occ",
        )),
        # Keep the raw meter value in J so RL and baseline read the same source.
        "PV": BoxSpace(
            low=-_numpy_.inf,
            high=+_numpy_.inf,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(OutputMeter.Ref("ElectricityProduced:Facility")),
        "outdoor_temp": BoxSpace(
            low=-_numpy_.inf,
            high=+_numpy_.inf,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(OutputVariable.Ref(
            type="Site Outdoor Air Drybulb Temperature",
            key="Environment",
        )),
        "cloud_cover": BoxSpace(
            low=0.0,
            high=10.0,
            dtype=_numpy_.float32,
            shape=(),
        ).bind(OutputVariable.Ref(
            type="Site Total Sky Cover",
            key="Environment",
        )),
    }
    if include_forecast:
        fields.update({
            "forecast_available": BoxSpace(
                low=0.0,
                high=1.0,
                dtype=_numpy_.float32,
                shape=(),
            ).bind(forecast_available),
            "forecast_temperature_6h": BoxSpace(
                low=-_numpy_.inf,
                high=+_numpy_.inf,
                dtype=_numpy_.float32,
                shape=(FORECAST_HORIZON_HOURS,),
            ).bind(forecast_temperature_6h),
            "forecast_humidity_6h": BoxSpace(
                low=-_numpy_.inf,
                high=+_numpy_.inf,
                dtype=_numpy_.float32,
                shape=(FORECAST_HORIZON_HOURS,),
            ).bind(forecast_humidity_6h),
            "forecast_cloudcover_6h": BoxSpace(
                low=0.0,
                high=100.0,
                dtype=_numpy_.float32,
                shape=(FORECAST_HORIZON_HOURS,),
            ).bind(forecast_cloudcover_6h),
            "forecast_precip_prob_6h": BoxSpace(
                low=0.0,
                high=100.0,
                dtype=_numpy_.float32,
                shape=(FORECAST_HORIZON_HOURS,),
            ).bind(forecast_precip_prob_6h),
            "forecast_precip_6h": BoxSpace(
                low=0.0,
                high=+_numpy_.inf,
                dtype=_numpy_.float32,
                shape=(FORECAST_HORIZON_HOURS,),
            ).bind(forecast_precip_6h),
        })
    return fields


def build_observation_space(
    room_agent_ids: dict[str, str],
    include_forecast: bool,
) -> DictSpace:
    return DictSpace({
        zone_id: DictSpace(
            build_zone_observation_fields(zone_key, include_forecast=include_forecast)
        )
        for zone_id, zone_key in room_agent_ids.items()
    })


class EnvRewardFunction:
    def __init__(
        self,
        w_building_energy: float = 1.0,
        reward_scale: float = 0.01,
    ):
        self._reward_fns = {
            zone_id: RewardFunction(zone_key=zone_key)
            for zone_id, zone_key in ZONE_MAP.items()
        }
        self.w_building_energy = float(w_building_energy)
        self.reward_scale = float(reward_scale)

    def __call__(self, agent):
        try:
            zone_observations = agent.observation.value
            first_zone_id = next(iter(self._reward_fns))
            first_zone_obs = zone_observations[first_zone_id]
            # HVAC-only energy = Facility - Building (excludes lights + equipment)
            hvac_energy_j = first_zone_obs["energy_consumption"] - first_zone_obs.get("energy_building", 0.0)
            # Allow negative net_grid (PV surplus) so midday blocks still have reward signal
            net_building_energy_kwh = joules_to_kwh(hvac_energy_j - first_zone_obs["PV"])
            comfort_penalties = []
            for zone_id in self._reward_fns.keys():
                zone_obs = zone_observations[zone_id]
                comfort_penalties.append(
                    self._reward_fns[zone_id](
                        agent,
                        {
                            "office_occupancy": zone_obs["occupancy"],
                            "temperature_drybulb": zone_obs["temperature_drybulb"],
                            "airspeed": 0.1,
                        },
                    )
                )
            penalty = (
                self.w_building_energy * net_building_energy_kwh
                + float(_numpy_.sum(comfort_penalties))
            )
            reward = -float(penalty) * self.reward_scale
            if not _numpy_.isfinite(reward):
                reward = 0.0
            return float(reward)
        except TemporaryUnavailableError:
            return 0.0


def make_env_config(include_forecast: bool) -> Env.Config:
    return {
        "action_space": build_action_space(ZONE_MAP),
        "observation_space": build_observation_space(
            ZONE_MAP,
            include_forecast=include_forecast,
        ),
        "reward": EnvRewardFunction(w_building_energy=float(os.getenv("RL_W_ENERGY", "1.0"))),
    }


class _BaseUserEnv(Env):
    room_agent_ids = ZONE_MAP
    config: Env.Config = {}

    def __init__(self, config: dict | None = None):
        install_space_patches()
        merged = dict(self.__class__.config)
        if config:
            merged.update(config)
        super().__init__(merged)

    def _coerce_observation(self, observation):
        return _coerce_space_value(self.observation_space, observation)

    def get_action(self, episode_id, observation=None):
        if observation is None:
            observation = self.agent.observation.value
        return super().get_action(
            episode_id,
            observation=self._coerce_observation(observation),
        )

    def log_action(self, episode_id, observation=None, action=None):
        if observation is None:
            observation = self.agent.observation.value
        return super().log_action(
            episode_id,
            observation=self._coerce_observation(observation),
            action=action if action is not None else self.agent.action.value,
        )

    def _get_latest_observation(self, episode_id):
        return self._coerce_observation(super()._get_latest_observation(episode_id))

    def end_episode(self, episode_id, observation=None):
        if observation is None:
            observation = self._get_latest_observation(episode_id)
        return super().end_episode(
            episode_id,
            observation=self._coerce_observation(observation),
        )

    def run(self):
        building_path = str((PROJECT_ROOT / os.getenv("RL_IDF", "houston.idf")).resolve())
        weather_path = str((WEATHER_DIR / os.getenv("RL_EPW", "houston_2025_06_01_2025_09_30_historical_weather_api.epw")).resolve())
        report_dir = str((REPORTS_DIR / RUN_NAME / REPORT_SUBDIR).resolve())

        system = System(
            building=building_path,
            weather=weather_path,
            report=report_dir,
            repeat=True,
            verbose=2,
        )
        self.__attach__(system)
        prime_env_bindings(self, system)

        rng = _numpy_.random.default_rng(1229)
        action_repeat = max(int(os.getenv("RL_ACTION_REPEAT", "1")), 1)
        state = {"skip": 0, "steps_left": 0, "episode_id": None, "repeat_count": 0}

        def _reset_episode_window():
            state["skip"] = int(rng.integers(0, EPISODE_STEPS))
            state["steps_left"] = EPISODE_STEPS
            state["episode_id"] = None
            state["repeat_count"] = 0

        _reset_episode_window()

        @system.events["timestep"].on
        def _on_timestep(_):
            update_forecast_observation(system["wallclock"].value)

            if state["skip"] > 0:
                state["skip"] -= 1
                return

            try:
                if state["episode_id"] is None:
                    state["episode_id"] = self.start_episode()

                # Action repeat: only call step_episode every N timesteps
                # In between, the previous action is held (EnergyPlus keeps last setpoint)
                if state["repeat_count"] == 0:
                    self.step_episode(state["episode_id"])
                state["repeat_count"] = (state["repeat_count"] + 1) % action_repeat
            except TemporaryUnavailableError:
                return

            state["steps_left"] -= 1

            if state["steps_left"] == 0:
                self.end_episode(state["episode_id"])
                _reset_episode_window()

        system.add("logging:progress").start().wait()


class UserEnv(_BaseUserEnv):
    config: Env.Config = make_env_config(include_forecast=True)


class UserEnvNoForecast(_BaseUserEnv):
    config: Env.Config = make_env_config(include_forecast=False)


ENV_VARIANTS = {
    "forecast_window": {
        "env_cls": UserEnv,
        "observation_keys": BASE_OBSERVATION_KEYS + FORECAST_OBSERVATION_KEYS,
    },
    "no_forecast_window": {
        "env_cls": UserEnvNoForecast,
        "observation_keys": BASE_OBSERVATION_KEYS,
    },
}
