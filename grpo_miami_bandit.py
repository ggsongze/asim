from __future__ import annotations

import importlib.util
import json
import os
import re
import uuid
from copy import deepcopy
from datetime import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from llm_setpoint_planner import HeuristicPlannerBackend, LLMSetpointPlanner, PlannerConstraints


def _extract_step_physics(observation, zone_ids):
    """Extract per-step physical quantities from observation for analysis."""
    try:
        from llm_setpoint_planner import estimate_zone_pmv
    except ImportError:
        estimate_zone_pmv = None

    first_zone = observation.get(zone_ids[0], {})
    pv_kwh = float(first_zone.get("PV", 0)) / 3.6e6
    facility_kwh = float(first_zone.get("energy_consumption", 0)) / 3.6e6
    building_kwh = float(first_zone.get("energy_building", 0)) / 3.6e6
    hvac_kwh = facility_kwh - building_kwh
    net_grid_kwh = hvac_kwh - pv_kwh  # can be negative (PV surplus)

    total_pmv_violation = 0.0
    for zid in zone_ids:
        zobs = observation.get(zid, {})
        temp = float(zobs.get("temperature_drybulb", 24.0))
        tr = float(zobs.get("temperature:radiant", zobs.get("temperature_radiant", temp)))
        hum = float(zobs.get("humidity", 50.0))
        occ = float(zobs.get("occupancy", 0.0))
        pmv = 0.0
        if estimate_zone_pmv is not None:
            try:
                pmv = estimate_zone_pmv(temperature_drybulb=temp, temperature_radiant=tr, humidity=hum)
            except Exception:
                pass
        total_pmv_violation += occ * max(abs(pmv) - 0.5, 0.0)

    return {
        "facility_kwh": round(facility_kwh, 4),
        "hvac_kwh": round(hvac_kwh, 4),
        "pv_kwh": round(pv_kwh, 4),
        "net_grid_kwh": round(net_grid_kwh, 4),
        "total_pmv_violation": round(total_pmv_violation, 4),
    }


PROJECT_ROOT = Path("/home/AD/user/lab/asim")
WEATHER_PATH = PROJECT_ROOT / "weather" / "miami_2025_06_01_2025_09_30_historical_weather_api.epw"
BUILDING_PATH = PROJECT_ROOT / "miami_3week.idf"
RESULT_DIR = PROJECT_ROOT / "result" / "gspo"
REPORTS_DIR = PROJECT_ROOT / "result" / "reports"


def _plainify(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _plainify(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_plainify(child) for child in value]
    if isinstance(value, tuple):
        return [_plainify(child) for child in value]
    if hasattr(value, "tolist"):
        try:
            return _plainify(value.tolist())
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return value


def load_env_module(tmp_module_name: str = ".tmp_todo_random_start_cell0.py") -> ModuleType:
    module_path = PROJECT_ROOT / tmp_module_name
    spec = importlib.util.spec_from_file_location("todo_env_cell0", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load env module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MiamiGRPOBandit:
    def __init__(
        self,
        *,
        tmp_module_name: str = ".tmp_todo_random_start_cell0.py",
        include_forecast: bool = True,
        default_setpoint_c: float = 24.0,
        control_window_start: str | None = None,
        control_window_end: str | None = None,
        weekday_only: bool = False,
        request_mode: str = "step_action",
        building_path: str | Path | None = None,
        weather_path: str | Path | None = None,
    ):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._building_path = Path(building_path) if building_path else BUILDING_PATH
        self._weather_path = Path(weather_path) if weather_path else WEATHER_PATH
        self.env_mod = load_env_module(tmp_module_name)
        self.zone_ids = tuple(self.env_mod.ZONE_MAP.keys())
        self.include_forecast = bool(include_forecast)
        self.default_setpoint_c = float(default_setpoint_c)
        self.control_window_start = self._parse_hhmm(control_window_start)
        self.control_window_end = self._parse_hhmm(control_window_end)
        self.weekday_only = bool(weekday_only)
        self.request_mode = str(request_mode)
        self.planner = LLMSetpointPlanner(
            HeuristicPlannerBackend(),
            constraints=PlannerConstraints(
                min_setpoint_c=20.0,
                max_setpoint_c=30.0,
                max_delta_per_step_c=2.0,
                fallback_setpoint_c=self.default_setpoint_c,
                quantization_c=0.1,
            ),
            zone_ids=self.zone_ids,
            step_minutes=10,
            candidate_count=1,
        )
        self._baseline_workday_cache: dict[tuple[int, str, int | None], dict[str, Any]] = {}

    @staticmethod
    def _parse_hhmm(value: str | None) -> time | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        hour_text, minute_text = text.split(":", maxsplit=1)
        return time(hour=int(hour_text), minute=int(minute_text))

    def _wallclock_in_control_window(self, ts: pd.Timestamp) -> bool:
        if self.control_window_start is None or self.control_window_end is None:
            return True
        current = ts.time()
        return self.control_window_start <= current <= self.control_window_end

    def _wallclock_on_allowed_day(self, ts: pd.Timestamp) -> bool:
        if not self.weekday_only:
            return True
        return int(ts.dayofweek) < 5

    def default_action(self) -> dict[str, dict[str, float]]:
        return {
            zone_id: {"thermostat": self.default_setpoint_c}
            for zone_id in self.zone_ids
        }

    @staticmethod
    def _serialize_action(action: dict[str, Any]) -> str:
        return json.dumps(_plainify(action), sort_keys=True, ensure_ascii=True)

    def _get_or_rollout_baseline_workday(
        self,
        *,
        skip_valid_steps: int,
        baseline_action: dict[str, dict[str, float]],
        previous_action: dict[str, Any],
        max_control_steps: int | None = None,
    ) -> tuple[dict[str, Any], bool]:
        cache_key = (
            int(skip_valid_steps),
            self._serialize_action(baseline_action),
            None if max_control_steps is None else int(max_control_steps),
        )
        cached = self._baseline_workday_cache.get(cache_key)
        if cached is not None:
            return deepcopy(cached), True
        baseline = self._rollout_workday(
            skip_valid_steps=skip_valid_steps,
            first_action=baseline_action,
            continuation_action=baseline_action,
            previous_action=previous_action,
            max_control_steps=max_control_steps,
        )
        self._baseline_workday_cache[cache_key] = deepcopy(baseline)
        return baseline, False

    def observation_is_plausible(self, observation: dict[str, dict[str, Any]]) -> bool:
        for zone_id in self.zone_ids:
            zone = observation[zone_id]
            try:
                drybulb = float(zone["temperature_drybulb"])
                humidity = float(zone["humidity"])
            except Exception:
                return False
            if not (10.0 <= drybulb <= 40.0):
                return False
            if not (0.0 <= humidity <= 100.0):
                return False
        return True

    def wallclock_is_eligible(self, wallclock: Any, observation: dict[str, dict[str, Any]]) -> bool:
        ts = pd.Timestamp(wallclock)
        if ts.tzinfo is not None or ts.tz is not None:
            ts = ts.tz_localize(None)

        forecast_reader = getattr(self.env_mod, "forecast_reader", None)
        forecast_years = getattr(forecast_reader, "_years", None)
        if forecast_years and int(ts.year) not in set(int(year) for year in forecast_years):
            return False

        if self.include_forecast:
            first_zone = observation[self.zone_ids[0]]
            try:
                forecast_available = float(first_zone.get("forecast_available", 0.0))
            except Exception:
                return False
            if forecast_available < 0.5:
                return False

        if not self._wallclock_on_allowed_day(ts):
            return False
        if not self._wallclock_in_control_window(ts):
            return False

        return True

    def _compact_forecast_from_observation(self, observation: dict[str, dict[str, Any]]) -> dict[str, Any]:
        first_zone = observation[self.zone_ids[0]]
        return {
            "available": bool(round(float(first_zone.get("forecast_available", 0.0)))),
            "temperature_6h_c": [round(float(x), 2) for x in list(first_zone.get("forecast_temperature_6h", []))],
            "humidity_6h_pct": [round(float(x), 2) for x in list(first_zone.get("forecast_humidity_6h", []))],
            "cloudcover_6h_pct": [round(float(x), 2) for x in list(first_zone.get("forecast_cloudcover_6h", []))],
            "precip_prob_6h_pct": [round(float(x), 2) for x in list(first_zone.get("forecast_precip_prob_6h", []))],
            "precip_6h_mm": [round(float(x), 2) for x in list(first_zone.get("forecast_precip_6h", []))],
        }

    @staticmethod
    def _compact_forecast_from_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
        forecast = dict(payload.get("forecast") or {})
        return {
            "available": bool(forecast.get("available", False)),
            "temperature_6h_c": [round(float(x), 2) for x in list(forecast.get("temperature_6h_c", []))],
            "humidity_6h_pct": [round(float(x), 2) for x in list(forecast.get("humidity_6h_pct", []))],
            "cloudcover_6h_pct": [round(float(x), 2) for x in list(forecast.get("cloudcover_6h_pct", []))],
            "precip_prob_6h_pct": [round(float(x), 2) for x in list(forecast.get("precip_prob_6h_pct", []))],
            "precip_6h_mm": [round(float(x), 2) for x in list(forecast.get("precip_6h_mm", []))],
        }

    def _validate_and_build_planner_step_trace(
        self,
        *,
        trace: dict[str, Any],
        observation: dict[str, dict[str, Any]],
        wallclock: Any,
    ) -> dict[str, Any]:
        request_payload = _plainify(getattr(trace.get("request"), "payload", None))
        if not isinstance(request_payload, dict):
            # No request payload (e.g. RLLib planner) - skip all validation, return minimal trace
            return {
                "wallclock": str(wallclock),
                "mode": "policy_step",
                "setpoints": _plainify(trace.get("setpoints")),
                "raw_setpoints": _plainify(trace.get("raw_setpoints")),
                "signals": _plainify(trace.get("signals")),
            }

        observation_zone_ids = list(self.zone_ids)
        missing_observation_zones = [
            zone_id
            for zone_id in self.zone_ids
            if zone_id not in observation
        ]
        if missing_observation_zones:
            raise RuntimeError(
                f"Observation is missing expected zones at {wallclock}: {missing_observation_zones}"
            )

        request_zone_ids = [
            str(zone.get("zone_id"))
            for zone in list(request_payload.get("zones") or [])
            if isinstance(zone, dict) and zone.get("zone_id") is not None
        ]
        missing_request_zones = [
            zone_id
            for zone_id in self.zone_ids
            if zone_id not in request_zone_ids
        ]
        extra_request_zones = [
            zone_id
            for zone_id in request_zone_ids
            if zone_id not in self.zone_ids
        ]
        if (
            len(request_zone_ids) != len(self.zone_ids)
            or missing_request_zones
            or extra_request_zones
        ):
            raise RuntimeError(
                "Planner request payload zone coverage mismatch at "
                f"{wallclock}: request_zone_ids={request_zone_ids}, "
                f"missing={missing_request_zones}, extra={extra_request_zones}"
            )

        observation_forecast = self._compact_forecast_from_observation(observation)
        request_forecast = self._compact_forecast_from_request_payload(request_payload)
        # Compare only keys present in both (cloudcover may be absent when
        # obs space is built without it for checkpoint compatibility).
        _compare_keys = [
            k for k in ("temperature_6h_c", "humidity_6h_pct", "cloudcover_6h_pct",
                         "precip_prob_6h_pct", "precip_6h_mm")
            if observation_forecast.get(k)  # skip empty lists
        ]
        forecast_horizon_lengths = {
            k: len(request_forecast[k]) for k in _compare_keys
        }
        if any(length != 6 for length in forecast_horizon_lengths.values()):
            raise RuntimeError(
                "Planner request payload forecast horizon mismatch at "
                f"{wallclock}: {forecast_horizon_lengths}"
            )
        obs_subset = {k: observation_forecast[k] for k in _compare_keys if k in observation_forecast}
        req_subset = {k: request_forecast[k] for k in _compare_keys if k in request_forecast}
        if obs_subset != req_subset:
            raise RuntimeError(
                "Planner request payload forecast does not match observation at "
                f"{wallclock}: request={req_subset}, observation={obs_subset}"
            )

        return {
            "wallclock": str(wallclock),
            "mode": "planner_step",
            "setpoints": _plainify(trace["setpoints"]),
            "sanitized_setpoints": _plainify(trace.get("sanitized_setpoints")),
            "candidate_count": int(trace.get("candidate_count", 1)),
            "candidate_summaries": _plainify(trace.get("candidate_summaries", [])),
            "raw_output": _plainify(trace.get("raw_output")),
            "request_payload": request_payload,
            "request_system_prompt": getattr(trace.get("request"), "system_prompt", None),
            "request_user_prompt": getattr(trace.get("request"), "user_prompt", None),
            "observation_zone_ids": observation_zone_ids,
            "observation_zone_count": len(observation_zone_ids),
            "request_zone_ids": request_zone_ids,
            "request_zone_count": len(request_zone_ids),
            "observation_forecast": observation_forecast,
            "request_forecast": request_forecast,
            "forecast_horizon_lengths": forecast_horizon_lengths,
            "forecast_matches_observation": True,
        }

    def build_training_request(
        self,
        observation: dict[str, dict[str, Any]],
        *,
        wallclock: Any,
        previous_action: dict[str, Any] | None = None,
        request_mode: str | None = None,
    ) -> dict[str, Any]:
        request_mode = str(request_mode or self.request_mode)
        planner_input = self.planner.build_input(
            observation,
            wallclock=wallclock,
            previous_action=previous_action,
        )
        payload = planner_input.to_payload()
        compact_payload = {
            "timestamp_utc": payload["timestamp_utc"],
            "step_minutes": payload["step_minutes"],
            "facility_electricity_kwh": round(float(payload["facility_electricity_kwh"]), 4),
            "hvac_electricity_kwh": round(float(payload["hvac_electricity_kwh"]), 4),
            "pv_kwh": round(float(payload["pv_kwh"]), 4),
            "net_grid_kwh": round(float(payload["net_grid_kwh"]), 4),
            "forecast": {
                "available": bool(payload["forecast"]["available"]),
                "temperature_6h_c": [round(float(x), 2) for x in payload["forecast"]["temperature_6h_c"]],
                "humidity_6h_pct": [round(float(x), 2) for x in payload["forecast"]["humidity_6h_pct"]],
                "cloudcover_6h_pct": [round(float(x), 2) for x in payload["forecast"]["cloudcover_6h_pct"]],
                "precip_prob_6h_pct": [round(float(x), 2) for x in payload["forecast"]["precip_prob_6h_pct"]],
                "precip_6h_mm": [round(float(x), 2) for x in payload["forecast"]["precip_6h_mm"]],
            },
            "zones": [
                {
                    "zone_id": zone["zone_id"],
                    "temperature_drybulb_c": round(float(zone["temperature_drybulb_c"]), 3),
                    "temperature_radiant_c": round(float(zone["temperature_radiant_c"]), 3),
                    "humidity_pct": round(float(zone["humidity_pct"]), 3),
                    "occupancy": round(float(zone["occupancy"]), 3),
                    "previous_setpoint_c": round(float(zone["previous_setpoint_c"]), 3),
                    "estimated_pmv": None if zone["estimated_pmv"] is None else round(float(zone["estimated_pmv"]), 3),
                }
                for zone in payload["zones"]
            ],
        }
        if request_mode == "workday_policy":
            system_prompt = (
                "You are designing one compact closed-loop HVAC policy for a full workday. "
                "The workday control window is Monday-Friday 06:00-19:00 with 10-minute control steps. "
                "The environment reward at each step is: "
                "reward = -0.01 * (3.0 * net_grid_kwh + sum_over_zones(50.0 * occupied_pmv_violation)). "
                "Here, net_grid_kwh = hvac_electricity_kwh - pv_kwh (can be negative when PV surplus), where hvac_electricity_kwh excludes lighting and equipment. "
                "For each zone, occupied_pmv_violation = occupancy * max(abs(PMV) - 0.5, 0). "
                "Your completion will not be applied just once. Instead, the same policy will be executed at every 10-minute step of the workday using the current observation at that step. "
                "Choose policy parameters that trade off energy and comfort over the whole day, not only the next step. "
                "Return JSON only."
            )
            user_prompt = (
                "Design a compact workday closed-loop policy using exactly this JSON schema and nothing else:\n"
                "{\n"
                '  "occupied_zone_setpoints_c": [24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0],\n'
                '  "unoccupied_zone_setpoints_c": [27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0],\n'
                '  "temp_gain": 0.6,\n'
                '  "pmv_gain": 0.8,\n'
                '  "hot_forecast_gain": 0.3,\n'
                '  "net_grid_gain": 0.3\n'
                "}\n"
                f"Fixed zone order for both temperature arrays: {list(self.zone_ids)}.\n"
                "Runtime semantics at each 10-minute step:\n"
                "- If occupancy > 0, the runtime uses:\n"
                "  setpoint = occupied_zone_setpoints_c[z] - temp_gain*(temperature_drybulb_c - 24.0) - pmv_gain*PMV - hot_forecast_gain*hot_signal + net_grid_gain*grid_signal\n"
                "- If occupancy <= 0, the runtime uses:\n"
                "  setpoint = unoccupied_zone_setpoints_c[z] + 0.5*net_grid_gain*grid_signal\n"
                "- hot_signal = clip((max(forecast_temperature_6h) - 30.0) / 5.0, 0, 1)\n"
                "- grid_signal = clip(net_grid_kwh / 30.0, 0, 1)\n"
                "- Then a safety layer clamps each zone to 20-30 C, rate-limits to about +/-2 C per step, rounds to 0.1 C, and applies symmetry/monotonicity checks.\n"
                "Output guidance:\n"
                "- each occupied_zone_setpoints_c entry should usually be about 22.0-26.0 C\n"
                "- each unoccupied_zone_setpoints_c entry should usually be about 25.0-30.0 C\n"
                "- gains should usually be in 0.0-2.0\n"
                "- use 0.1 C precision for all temperatures\n"
                "Current day-start observation window JSON:\n"
                f"{json.dumps(compact_payload, ensure_ascii=True)}"
            )
        else:
            system_prompt = (
                "You control 8 HVAC cooling setpoints for the next 10-minute step. "
                "Your objective is to maximize the immediate next-step environment reward. "
                "The reward is computed as: "
                "reward = -0.01 * (3.0 * net_grid_kwh + sum_over_zones(50.0 * occupied_pmv_violation)). "
                "Here, net_grid_kwh = hvac_electricity_kwh - pv_kwh (can be negative when PV surplus), where hvac_electricity_kwh excludes lighting and equipment. "
                "For each zone, occupied_pmv_violation = occupancy * max(abs(PMV) - 0.5, 0). "
                "Lower setpoints usually improve occupied comfort but increase cooling energy. "
                "Higher setpoints usually save energy but can worsen occupied comfort. "
                "Use occupancy, PMV, current temperatures, PV, net grid and the 6-hour forecast to choose the next 10-minute setpoints. "
                "Focus on the immediate next step, not a long-term plan. "
                "Return JSON only with exactly these zone keys: "
                f"{list(self.zone_ids)}. "
                "Each value must be a numeric Celsius cooling setpoint between 20.0 and 30.0."
            )
            user_prompt = (
                "Current observation window JSON:\n"
                f"{json.dumps(compact_payload, ensure_ascii=True)}\n"
                "Return only one JSON object with the zone keys and numeric setpoints for the next 10-minute step."
            )
        return {
            "request_mode": request_mode,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "payload": compact_payload,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    @staticmethod
    def _extract_json_text(raw_output: Any) -> str:
        if isinstance(raw_output, dict):
            return json.dumps(raw_output, ensure_ascii=True)
        text = str(raw_output).strip()
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            return text
        match = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if not match:
            raise ValueError("No JSON payload found in policy response.")
        return match.group(0)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(float(low), min(float(high), float(value)))

    def _sanitize_zone_temperature_vector(
        self,
        value: Any,
        *,
        default_c: float,
    ) -> list[float]:
        if isinstance(value, dict):
            raw = [value.get(zone_id, default_c) for zone_id in self.zone_ids]
        elif isinstance(value, list):
            raw = [
                value[idx] if idx < len(value) else default_c
                for idx, _zone_id in enumerate(self.zone_ids)
            ]
        else:
            raw = [default_c] * len(self.zone_ids)
        out: list[float] = []
        for item in raw:
            temp = self._clamp(float(item), 20.0, 30.0)
            temp = round(temp * 10.0) / 10.0
            out.append(float(temp))
        return out

    def sanitize_workday_policy(self, raw_output: Any) -> dict[str, Any]:
        parsed = json.loads(self._extract_json_text(raw_output))
        if isinstance(parsed, dict) and isinstance(parsed.get("policy"), dict):
            parsed = parsed["policy"]
        if not isinstance(parsed, dict):
            raise TypeError("Workday policy output must be a JSON object.")

        # Backward-compatible: old policy schema used occupied/unoccupied bases plus zone_bias_c.
        # New schema uses direct occupied/unoccupied zone temperatures.
        if "occupied_zone_setpoints_c" in parsed or "unoccupied_zone_setpoints_c" in parsed:
            occupied_zone_setpoints_c = self._sanitize_zone_temperature_vector(
                parsed.get("occupied_zone_setpoints_c"),
                default_c=self.default_setpoint_c,
            )
            unoccupied_zone_setpoints_c = self._sanitize_zone_temperature_vector(
                parsed.get("unoccupied_zone_setpoints_c"),
                default_c=27.0,
            )
        else:
            occupied_base_c = self._clamp(float(parsed.get("occupied_base_c", self.default_setpoint_c)), 20.0, 30.0)
            unoccupied_base_c = self._clamp(float(parsed.get("unoccupied_base_c", 27.0)), 20.0, 30.0)
            zone_bias_value = parsed.get("zone_bias_c", [0.0] * len(self.zone_ids))
            if isinstance(zone_bias_value, dict):
                zone_bias = [
                    self._clamp(float(zone_bias_value.get(zone_id, 0.0)), -2.0, 2.0)
                    for zone_id in self.zone_ids
                ]
            elif isinstance(zone_bias_value, list):
                zone_bias = [
                    self._clamp(float(zone_bias_value[idx]) if idx < len(zone_bias_value) else 0.0, -2.0, 2.0)
                    for idx, _zone_id in enumerate(self.zone_ids)
                ]
            else:
                raise TypeError("zone_bias_c must be a list or object.")
            occupied_zone_setpoints_c = [
                float(round((occupied_base_c + zone_bias[idx]) * 10.0) / 10.0)
                for idx in range(len(self.zone_ids))
            ]
            unoccupied_zone_setpoints_c = [
                float(round((unoccupied_base_c + zone_bias[idx]) * 10.0) / 10.0)
                for idx in range(len(self.zone_ids))
            ]

        return {
            "occupied_zone_setpoints_c": occupied_zone_setpoints_c,
            "unoccupied_zone_setpoints_c": unoccupied_zone_setpoints_c,
            "temp_gain": self._clamp(float(parsed.get("temp_gain", 0.5)), 0.0, 2.0),
            "pmv_gain": self._clamp(float(parsed.get("pmv_gain", 0.6)), 0.0, 2.0),
            "hot_forecast_gain": self._clamp(float(parsed.get("hot_forecast_gain", 0.2)), 0.0, 2.0),
            "net_grid_gain": self._clamp(float(parsed.get("net_grid_gain", 0.2)), 0.0, 2.0),
        }

    def _policy_trace_for_observation(
        self,
        policy: dict[str, Any],
        observation: dict[str, dict[str, Any]],
        *,
        wallclock: Any,
        previous_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = self.planner.build_request(
            observation,
            wallclock=wallclock,
            previous_action=previous_action,
        )
        payload = request.payload
        forecast_temp = payload["forecast"]["temperature_6h_c"]
        hot_signal = 0.0
        if payload["forecast"]["available"] and forecast_temp:
            hot_signal = self._clamp((max(float(x) for x in forecast_temp) - 30.0) / 5.0, 0.0, 1.0)
        grid_signal = self._clamp(float(payload["net_grid_kwh"]) / 30.0, 0.0, 1.0)

        raw_setpoints: dict[str, float] = {}
        for zone_index, zone in enumerate(payload["zones"]):
            zone_id = zone["zone_id"]
            occupancy = float(zone["occupancy"])
            if occupancy > 0.0:
                temp_term = float(policy["temp_gain"]) * (float(zone["temperature_drybulb_c"]) - 24.0)
                pmv_value = 0.0 if zone.get("estimated_pmv") is None else float(zone["estimated_pmv"])
                pmv_term = float(policy["pmv_gain"]) * pmv_value
                raw_value = (
                    float(policy["occupied_zone_setpoints_c"][zone_index])
                    - temp_term
                    - pmv_term
                    - float(policy["hot_forecast_gain"]) * hot_signal
                    + float(policy["net_grid_gain"]) * grid_signal
                )
            else:
                raw_value = (
                    float(policy["unoccupied_zone_setpoints_c"][zone_index])
                    + 0.5 * float(policy["net_grid_gain"]) * grid_signal
                )
            raw_setpoints[zone_id] = self.planner._requantize_bounded_value(
                zone_id=zone_id,
                value=raw_value,
                previous_action=previous_action,
            )

        setpoints = self.planner.post_check_setpoints(
            raw_setpoints,
            request=request,
            previous_action=previous_action,
        )
        action = self.planner.to_env_action(setpoints)
        return {
            "request": request,
            "raw_setpoints": raw_setpoints,
            "setpoints": setpoints,
            "action": action,
            "signals": {
                "hot_signal": float(hot_signal),
                "grid_signal": float(grid_signal),
                "net_grid_kwh": float(payload["net_grid_kwh"]),
            },
        }

    def _rollout_workday_closed_loop(
        self,
        *,
        skip_valid_steps: int,
        policy: dict[str, Any],
        previous_action: dict[str, Any],
    ) -> dict[str, Any]:
        report_subdir = f"gspo_day_closed_{uuid.uuid4().hex[:10]}"
        env, system = self._make_env_and_system(report_subdir)
        state = {
            "skip_remaining": int(skip_valid_steps),
            "phase": "seek_start",
            "last_seen_wallclock": None,
            "target_date": None,
            "pending_reward": False,
            "control_steps_applied": 0,
            "invalid_observation_skips": 0,
            "ineligible_window_skips": 0,
            "total_reward": 0.0,
            "reward_trace": [],
            "action_trace": [],
            "start_observation": None,
            "start_wallclock": None,
            "end_wallclock": None,
            "last_action": previous_action,
        }
        result: dict[str, Any] = {}

        @system.events["timestep"].on
        def _on_timestep(_):
            self.env_mod.update_forecast_observation(system["wallclock"].value)
            wallclock_key = str(system["wallclock"].value)
            if state["last_seen_wallclock"] == wallclock_key:
                return
            state["last_seen_wallclock"] = wallclock_key

            ts = pd.Timestamp(system["wallclock"].value)
            if ts.tzinfo is not None or ts.tz is not None:
                ts = ts.tz_localize(None)

            try:
                observation = deepcopy(env._coerce_observation(env.agent.observation.value))
                observation_plausible = self.observation_is_plausible(observation)
            except self.env_mod.TemporaryUnavailableError:
                return
            except Exception:
                observation = None
                observation_plausible = False

            is_eligible = False
            if observation_plausible and observation is not None:
                is_eligible = self.wallclock_is_eligible(system["wallclock"].value, observation)
            else:
                state["invalid_observation_skips"] += 1

            if state["phase"] == "seek_start":
                if not observation_plausible or observation is None:
                    return
                if not is_eligible:
                    state["ineligible_window_skips"] += 1
                    return
                if state["skip_remaining"] > 0:
                    state["skip_remaining"] -= 1
                    return
                trace = self._policy_trace_for_observation(
                    policy,
                    observation,
                    wallclock=system["wallclock"].value,
                    previous_action=state["last_action"],
                )
                env.agent.action.value = trace["action"]
                state["phase"] = "rollout"
                state["pending_reward"] = True
                state["target_date"] = ts.date()
                state["control_steps_applied"] = 1
                state["start_observation"] = _plainify(observation)
                state["start_wallclock"] = str(system["wallclock"].value)
                state["last_action"] = trace["action"]
                state["action_trace"].append({
                    "wallclock": str(system["wallclock"].value),
                    "mode": "policy_step",
                    "setpoints": _plainify(trace["setpoints"]),
                    "raw_setpoints": _plainify(trace["raw_setpoints"]),
                    "signals": _plainify(trace["signals"]),
                })
                return

            if state["pending_reward"]:
                reward = float(env.agent.reward.value)
                state["total_reward"] += reward
                state["reward_trace"].append({
                    "wallclock": str(system["wallclock"].value),
                    "reward": reward,
                })
                state["pending_reward"] = False

            same_day = ts.date() == state["target_date"]
            if same_day and observation_plausible and observation is not None and is_eligible:
                trace = self._policy_trace_for_observation(
                    policy,
                    observation,
                    wallclock=system["wallclock"].value,
                    previous_action=state["last_action"],
                )
                env.agent.action.value = trace["action"]
                state["pending_reward"] = True
                state["control_steps_applied"] += 1
                state["last_action"] = trace["action"]
                state["action_trace"].append({
                    "wallclock": str(system["wallclock"].value),
                    "mode": "policy_step",
                    "setpoints": _plainify(trace["setpoints"]),
                    "raw_setpoints": _plainify(trace["raw_setpoints"]),
                    "signals": _plainify(trace["signals"]),
                })
                return

            state["end_wallclock"] = str(system["wallclock"].value)
            result.update({
                "start_wallclock": state["start_wallclock"],
                "end_wallclock": state["end_wallclock"],
                "target_date": None if state["target_date"] is None else str(state["target_date"]),
                "start_observation": state["start_observation"],
                "total_reward": float(state["total_reward"]),
                "control_steps_applied": int(state["control_steps_applied"]),
                "reward_trace": state["reward_trace"],
                "action_trace": state["action_trace"],
                "invalid_observation_skips": int(state["invalid_observation_skips"]),
                "ineligible_window_skips": int(state["ineligible_window_skips"]),
            })
            try:
                system.stop()
            except Exception:
                pass

        system.add("logging:progress").start().wait()
        if not result:
            raise RuntimeError("Failed to roll out closed-loop workday in Miami GRPO bandit environment.")
        return result

    def _make_env_and_system(self, report_subdir: str):
        env_cls = self.env_mod.UserEnv if self.include_forecast else self.env_mod.UserEnvNoForecast
        env = env_cls()
        system = self.env_mod.System(
            building=str(self._building_path.resolve()),
            weather=str(self._weather_path.resolve()),
            report=str((REPORTS_DIR / report_subdir).resolve()),
            repeat=False,
            verbose=0,
        )
        env.__attach__(system)
        self.env_mod.prime_env_bindings(env, system)
        return env, system

    def sample_state(self, skip_valid_steps: int) -> dict[str, Any]:
        report_subdir = f"gspo_state_{uuid.uuid4().hex[:10]}"
        env, system = self._make_env_and_system(report_subdir)
        result: dict[str, Any] = {}
        state = {
            "skip_remaining": int(skip_valid_steps),
            "last_seen_wallclock": None,
            "ineligible_window_skips": 0,
        }

        @system.events["timestep"].on
        def _on_timestep(_):
            self.env_mod.update_forecast_observation(system["wallclock"].value)
            wallclock_key = str(system["wallclock"].value)
            if state["last_seen_wallclock"] == wallclock_key:
                return
            state["last_seen_wallclock"] = wallclock_key
            try:
                observation = deepcopy(env._coerce_observation(env.agent.observation.value))
            except self.env_mod.TemporaryUnavailableError:
                return
            if not self.observation_is_plausible(observation):
                return
            if not self.wallclock_is_eligible(system["wallclock"].value, observation):
                state["ineligible_window_skips"] += 1
                return
            if state["skip_remaining"] > 0:
                state["skip_remaining"] -= 1
                return

            previous_action = self.default_action()
            request = self.build_training_request(
                observation,
                wallclock=system["wallclock"].value,
                previous_action=previous_action,
            )
            result.update({
                "skip_valid_steps": int(skip_valid_steps),
                "wallclock": str(system["wallclock"].value),
                "observation": _plainify(observation),
                "previous_action": previous_action,
                "request_mode": request["request_mode"],
                "prompt": request["messages"],
                "system_prompt": request["system_prompt"],
                "user_prompt": request["user_prompt"],
                "payload": request["payload"],
                "zone_ids": list(self.zone_ids),
                "ineligible_window_skips": int(state["ineligible_window_skips"]),
            })
            try:
                system.stop()
            except Exception:
                pass

        system.add("logging:progress").start().wait()
        if not result:
            raise RuntimeError(f"Failed to collect valid state for skip_valid_steps={skip_valid_steps}")
        return result

    def collect_states(
        self,
        *,
        count: int | None,
        skip_start: int = 0,
        skip_step: int = 1,
        row_filter: Any = None,
    ) -> dict[str, Any]:
        report_subdir = f"gspo_collect_{uuid.uuid4().hex[:10]}"
        env, system = self._make_env_and_system(report_subdir)
        rows: list[dict[str, Any]] = []
        state = {
            "last_seen_wallclock": None,
            "valid_steps_seen": 0,
            "invalid_observation_skips": 0,
            "ineligible_window_skips": 0,
        }
        target_count = None if count is None or int(count) <= 0 else int(count)
        skip_start = int(skip_start)
        skip_step = max(int(skip_step), 1)

        @system.events["timestep"].on
        def _on_timestep(_):
            self.env_mod.update_forecast_observation(system["wallclock"].value)
            wallclock_key = str(system["wallclock"].value)
            if state["last_seen_wallclock"] == wallclock_key:
                return
            state["last_seen_wallclock"] = wallclock_key
            try:
                observation = deepcopy(env._coerce_observation(env.agent.observation.value))
            except self.env_mod.TemporaryUnavailableError:
                return
            if not self.observation_is_plausible(observation):
                state["invalid_observation_skips"] += 1
                return
            if not self.wallclock_is_eligible(system["wallclock"].value, observation):
                state["ineligible_window_skips"] += 1
                return

            current_valid_index = int(state["valid_steps_seen"])
            state["valid_steps_seen"] += 1

            if current_valid_index < skip_start:
                return
            if (current_valid_index - skip_start) % skip_step != 0:
                return

            previous_action = self.default_action()
            request = self.build_training_request(
                observation,
                wallclock=system["wallclock"].value,
                previous_action=previous_action,
            )
            row = {
                "skip_valid_steps": current_valid_index,
                "wallclock": str(system["wallclock"].value),
                "observation": _plainify(observation),
                "previous_action": previous_action,
                "request_mode": request["request_mode"],
                "prompt": request["messages"],
                "system_prompt": request["system_prompt"],
                "user_prompt": request["user_prompt"],
                "payload": request["payload"],
                "zone_ids": list(self.zone_ids),
                "invalid_observation_skips": int(state["invalid_observation_skips"]),
                "ineligible_window_skips": int(state["ineligible_window_skips"]),
            }
            if row_filter is not None and not bool(row_filter(row)):
                return
            row["sample_index"] = len(rows)
            rows.append(row)
            if target_count is not None and len(rows) >= target_count:
                try:
                    system.stop()
                except Exception:
                    pass

        system.add("logging:progress").start().wait()
        return {
            "rows": rows,
            "valid_steps_seen": int(state["valid_steps_seen"]),
            "invalid_observation_skips": int(state["invalid_observation_skips"]),
            "ineligible_window_skips": int(state["ineligible_window_skips"]),
        }

    def _rollout_workday(
        self,
        *,
        skip_valid_steps: int,
        first_action: dict[str, dict[str, float]],
        continuation_action: dict[str, dict[str, float]],
        previous_action: dict[str, Any],
        max_control_steps: int | None = None,
    ) -> dict[str, Any]:
        report_subdir = f"gspo_day_{uuid.uuid4().hex[:10]}"
        env, system = self._make_env_and_system(report_subdir)
        state = {
            "skip_remaining": int(skip_valid_steps),
            "phase": "seek_start",
            "last_seen_wallclock": None,
            "target_date": None,
            "pending_reward": False,
            "control_steps_applied": 0,
            "invalid_observation_skips": 0,
            "ineligible_window_skips": 0,
            "total_reward": 0.0,
            "reward_trace": [],
            "action_trace": [],
            "start_observation": None,
            "start_wallclock": None,
            "end_wallclock": None,
            "active_step_index": 0,
            "last_action": previous_action,
        }
        result: dict[str, Any] = {}

        @system.events["timestep"].on
        def _on_timestep(_):
            self.env_mod.update_forecast_observation(system["wallclock"].value)
            wallclock_key = str(system["wallclock"].value)
            if state["last_seen_wallclock"] == wallclock_key:
                return
            state["last_seen_wallclock"] = wallclock_key

            ts = pd.Timestamp(system["wallclock"].value)
            if ts.tzinfo is not None or ts.tz is not None:
                ts = ts.tz_localize(None)

            try:
                observation = deepcopy(env._coerce_observation(env.agent.observation.value))
                observation_plausible = self.observation_is_plausible(observation)
            except self.env_mod.TemporaryUnavailableError:
                return
            except Exception:
                observation = None
                observation_plausible = False

            is_eligible = False
            if observation_plausible and observation is not None:
                is_eligible = self.wallclock_is_eligible(system["wallclock"].value, observation)
            else:
                state["invalid_observation_skips"] += 1

            if state["phase"] == "seek_start":
                if not observation_plausible or observation is None:
                    return
                if not is_eligible:
                    state["ineligible_window_skips"] += 1
                    return
                if state["skip_remaining"] > 0:
                    state["skip_remaining"] -= 1
                    return

                env.agent.action.value = first_action
                state["phase"] = "rollout"
                state["pending_reward"] = True
                state["target_date"] = ts.date()
                state["control_steps_applied"] = 1
                state["start_observation"] = _plainify(observation)
                state["start_wallclock"] = str(system["wallclock"].value)
                state["last_action"] = first_action
                state["action_trace"].append({
                    "wallclock": str(system["wallclock"].value),
                    "mode": "first_action",
                    "action": _plainify(first_action),
                })
                return

            if state["pending_reward"]:
                reward = float(env.agent.reward.value)
                state["total_reward"] += reward
                _phys = _extract_step_physics(observation, self.zone_ids) if observation is not None else {}
                state["reward_trace"].append({
                    "wallclock": str(system["wallclock"].value),
                    "reward": reward,
                    **_phys,
                })
                state["pending_reward"] = False

            same_day = ts.date() == state["target_date"]
            under_step_limit = (
                max_control_steps is None
                or int(state["control_steps_applied"]) < int(max_control_steps)
            )
            if same_day and under_step_limit and observation_plausible and observation is not None and is_eligible:
                env.agent.action.value = continuation_action
                state["pending_reward"] = True
                state["control_steps_applied"] += 1
                state["last_action"] = continuation_action
                state["action_trace"].append({
                    "wallclock": str(system["wallclock"].value),
                    "mode": "continuation",
                    "action": _plainify(continuation_action),
                })
                return

            state["end_wallclock"] = str(system["wallclock"].value)
            result.update({
                "start_wallclock": state["start_wallclock"],
                "end_wallclock": state["end_wallclock"],
                "target_date": None if state["target_date"] is None else str(state["target_date"]),
                "start_observation": state["start_observation"],
                "total_reward": float(state["total_reward"]),
                "control_steps_applied": int(state["control_steps_applied"]),
                "reward_trace": state["reward_trace"],
                "action_trace": state["action_trace"],
                "invalid_observation_skips": int(state["invalid_observation_skips"]),
                "ineligible_window_skips": int(state["ineligible_window_skips"]),
            })
            try:
                system.stop()
            except Exception:
                pass

        system.add("logging:progress").start().wait()
        if not result:
            raise RuntimeError("Failed to roll out workday in Miami GRPO bandit environment.")
        return result

    def _rollout_workday_requery_planner(
        self,
        *,
        skip_valid_steps: int,
        planner: LLMSetpointPlanner,
        previous_action: dict[str, Any],
        max_control_steps: int | None = None,
    ) -> dict[str, Any]:
        report_subdir = f"gspo_day_requery_{uuid.uuid4().hex[:10]}"
        env, system = self._make_env_and_system(report_subdir)
        state = {
            "skip_remaining": int(skip_valid_steps),
            "phase": "seek_start",
            "last_seen_wallclock": None,
            "target_date": None,
            "pending_reward": False,
            "control_steps_applied": 0,
            "invalid_observation_skips": 0,
            "ineligible_window_skips": 0,
            "total_reward": 0.0,
            "reward_trace": [],
            "action_trace": [],
            "start_observation": None,
            "start_wallclock": None,
            "end_wallclock": None,
            "last_action": previous_action,
        }
        result: dict[str, Any] = {}

        @system.events["timestep"].on
        def _on_timestep(_):
            self.env_mod.update_forecast_observation(system["wallclock"].value)
            wallclock_key = str(system["wallclock"].value)
            if state["last_seen_wallclock"] == wallclock_key:
                return
            state["last_seen_wallclock"] = wallclock_key

            ts = pd.Timestamp(system["wallclock"].value)
            if ts.tzinfo is not None or ts.tz is not None:
                ts = ts.tz_localize(None)

            try:
                observation = deepcopy(env._coerce_observation(env.agent.observation.value))
                observation_plausible = self.observation_is_plausible(observation)
            except self.env_mod.TemporaryUnavailableError:
                return
            except Exception:
                observation = None
                observation_plausible = False

            is_eligible = False
            if observation_plausible and observation is not None:
                is_eligible = self.wallclock_is_eligible(system["wallclock"].value, observation)
            else:
                state["invalid_observation_skips"] += 1

            if state["phase"] == "seek_start":
                if not observation_plausible or observation is None:
                    return
                if not is_eligible:
                    state["ineligible_window_skips"] += 1
                    return
                if state["skip_remaining"] > 0:
                    state["skip_remaining"] -= 1
                    return

                trace = planner.plan_next_action_with_trace(
                    observation,
                    wallclock=system["wallclock"].value,
                    previous_action=state["last_action"],
                )
                env.agent.action.value = trace["action"]
                state["phase"] = "rollout"
                state["pending_reward"] = True
                state["target_date"] = ts.date()
                state["control_steps_applied"] = 1
                state["start_observation"] = _plainify(observation)
                state["start_wallclock"] = str(system["wallclock"].value)
                state["last_action"] = trace["action"]
                state["action_trace"].append(
                    self._validate_and_build_planner_step_trace(
                        trace=trace,
                        observation=observation,
                        wallclock=system["wallclock"].value,
                    )
                )
                return

            if state["pending_reward"]:
                reward = float(env.agent.reward.value)
                state["total_reward"] += reward
                _phys = _extract_step_physics(observation, self.zone_ids) if observation is not None else {}
                state["reward_trace"].append({
                    "wallclock": str(system["wallclock"].value),
                    "reward": reward,
                    **_phys,
                })
                state["pending_reward"] = False

            same_day = ts.date() == state["target_date"]
            under_step_limit = (
                max_control_steps is None
                or int(state["control_steps_applied"]) < int(max_control_steps)
            )
            if same_day and under_step_limit and observation_plausible and observation is not None and is_eligible:
                trace = planner.plan_next_action_with_trace(
                    observation,
                    wallclock=system["wallclock"].value,
                    previous_action=state["last_action"],
                )
                env.agent.action.value = trace["action"]
                state["pending_reward"] = True
                state["control_steps_applied"] += 1
                state["last_action"] = trace["action"]
                state["action_trace"].append(
                    self._validate_and_build_planner_step_trace(
                        trace=trace,
                        observation=observation,
                        wallclock=system["wallclock"].value,
                    )
                )
                return

            state["end_wallclock"] = str(system["wallclock"].value)
            result.update({
                "start_wallclock": state["start_wallclock"],
                "end_wallclock": state["end_wallclock"],
                "target_date": None if state["target_date"] is None else str(state["target_date"]),
                "start_observation": state["start_observation"],
                "total_reward": float(state["total_reward"]),
                "control_steps_applied": int(state["control_steps_applied"]),
                "reward_trace": state["reward_trace"],
                "action_trace": state["action_trace"],
                "invalid_observation_skips": int(state["invalid_observation_skips"]),
                "ineligible_window_skips": int(state["ineligible_window_skips"]),
            })
            try:
                system.stop()
            except Exception:
                pass

        system.add("logging:progress").start().wait()
        if not result:
            raise RuntimeError("Failed to roll out rolling planner workday in Miami GRPO bandit environment.")
        return result

    def evaluate_completion(
        self,
        *,
        skip_valid_steps: int,
        completion_text: str,
        previous_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        initial_previous_action = previous_action or self.default_action()
        # Fail fast on clearly invalid planner output before spinning up EnergyPlus.
        self.planner.sanitize_setpoints(
            completion_text,
            previous_action=initial_previous_action,
        )

        report_subdir = f"gspo_eval_{uuid.uuid4().hex[:10]}"
        env, system = self._make_env_and_system(report_subdir)
        state = {
            "skip_remaining": int(skip_valid_steps),
            "phase": "seek_state",
            "last_seen_wallclock": None,
            "last_action": initial_previous_action,
            "invalid_observation_skips": 0,
            "ineligible_window_skips": 0,
        }
        result: dict[str, Any] = {}

        @system.events["timestep"].on
        def _on_timestep(_):
            self.env_mod.update_forecast_observation(system["wallclock"].value)
            wallclock_key = str(system["wallclock"].value)
            if state["last_seen_wallclock"] == wallclock_key:
                return
            state["last_seen_wallclock"] = wallclock_key

            try:
                observation = deepcopy(env._coerce_observation(env.agent.observation.value))
            except self.env_mod.TemporaryUnavailableError:
                return
            if not self.observation_is_plausible(observation):
                state["invalid_observation_skips"] += 1
                return
            if not self.wallclock_is_eligible(system["wallclock"].value, observation):
                state["ineligible_window_skips"] += 1
                return

            if state["phase"] == "seek_state":
                if state["skip_remaining"] > 0:
                    state["skip_remaining"] -= 1
                    return
                planner_request = self.planner.build_request(
                    observation,
                    wallclock=system["wallclock"].value,
                    previous_action=state["last_action"],
                )
                sanitized = self.planner.sanitize_setpoints(
                    completion_text,
                    previous_action=state["last_action"],
                )
                setpoints = self.planner.post_check_setpoints(
                    sanitized,
                    request=planner_request,
                    previous_action=state["last_action"],
                )
                action = self.planner.to_env_action(setpoints)
                env.agent.action.value = action
                result.update({
                    "current_wallclock": str(system["wallclock"].value),
                    "current_observation": _plainify(observation),
                    "sanitized_setpoints": sanitized,
                    "setpoints": setpoints,
                    "action": action,
                })
                state["phase"] = "await_reward"
                return

            reward = float(env.agent.reward.value)
            result.update({
                "next_wallclock": str(system["wallclock"].value),
                "next_observation": _plainify(observation),
                "reward": reward,
                "invalid_observation_skips": state["invalid_observation_skips"],
                "ineligible_window_skips": state["ineligible_window_skips"],
            })
            try:
                system.stop()
            except Exception:
                pass

        system.add("logging:progress").start().wait()
        if not result:
            raise RuntimeError(
                "Failed to evaluate completion in Miami GRPO bandit environment."
            )
        return result

    def evaluate_completion_workday(
        self,
        *,
        skip_valid_steps: int,
        completion_text: str,
        previous_action: dict[str, Any] | None = None,
        baseline_action: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        initial_previous_action = previous_action or self.default_action()
        baseline_action = baseline_action or self.default_action()

        self.planner.sanitize_setpoints(
            completion_text,
            previous_action=initial_previous_action,
        )

        state_row = self.sample_state(skip_valid_steps=skip_valid_steps)
        planner_request = self.planner.build_request(
            state_row["observation"],
            wallclock=state_row["wallclock"],
            previous_action=initial_previous_action,
        )
        sanitized = self.planner.sanitize_setpoints(
            completion_text,
            previous_action=initial_previous_action,
        )
        setpoints = self.planner.post_check_setpoints(
            sanitized,
            request=planner_request,
            previous_action=initial_previous_action,
        )
        candidate_action = self.planner.to_env_action(setpoints)

        candidate = self._rollout_workday(
            skip_valid_steps=skip_valid_steps,
            first_action=candidate_action,
            continuation_action=baseline_action,
            previous_action=initial_previous_action,
        )
        baseline, baseline_cache_hit = self._get_or_rollout_baseline_workday(
            skip_valid_steps=skip_valid_steps,
            baseline_action=baseline_action,
            previous_action=initial_previous_action,
        )

        candidate_day_return = float(candidate["total_reward"])
        baseline_day_return = float(baseline["total_reward"])
        relative_day_return = candidate_day_return - baseline_day_return

        return {
            "skip_valid_steps": int(skip_valid_steps),
            "wallclock": state_row["wallclock"],
            "target_date": candidate.get("target_date") or baseline.get("target_date"),
            "current_observation": state_row["observation"],
            "sanitized_setpoints": sanitized,
            "setpoints": setpoints,
            "action": candidate_action,
            "day_return": candidate_day_return,
            "baseline_day_return": baseline_day_return,
            "relative_day_return": relative_day_return,
            "baseline_action": baseline_action,
            "candidate_control_steps_applied": int(candidate["control_steps_applied"]),
            "baseline_control_steps_applied": int(baseline["control_steps_applied"]),
            "candidate_start_wallclock": candidate.get("start_wallclock"),
            "candidate_end_wallclock": candidate.get("end_wallclock"),
            "baseline_start_wallclock": baseline.get("start_wallclock"),
            "baseline_end_wallclock": baseline.get("end_wallclock"),
            "candidate_reward_trace": candidate.get("reward_trace", []),
            "baseline_reward_trace": baseline.get("reward_trace", []),
            "candidate_action_trace": candidate.get("action_trace", []),
            "baseline_action_trace": baseline.get("action_trace", []),
            "invalid_observation_skips": int(candidate.get("invalid_observation_skips", 0)),
            "ineligible_window_skips": int(candidate.get("ineligible_window_skips", 0)),
            "baseline_cache_hit": bool(baseline_cache_hit),
        }

    def evaluate_completion_workday_closed_loop(
        self,
        *,
        skip_valid_steps: int,
        completion_text: str,
        previous_action: dict[str, Any] | None = None,
        baseline_action: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        initial_previous_action = previous_action or self.default_action()
        baseline_action = baseline_action or self.default_action()
        policy = self.sanitize_workday_policy(completion_text)

        candidate = self._rollout_workday_closed_loop(
            skip_valid_steps=skip_valid_steps,
            policy=policy,
            previous_action=initial_previous_action,
        )
        baseline, baseline_cache_hit = self._get_or_rollout_baseline_workday(
            skip_valid_steps=skip_valid_steps,
            baseline_action=baseline_action,
            previous_action=initial_previous_action,
        )

        candidate_day_return = float(candidate["total_reward"])
        baseline_day_return = float(baseline["total_reward"])
        relative_day_return = candidate_day_return - baseline_day_return
        first_trace = candidate.get("action_trace", [{}])[0] if candidate.get("action_trace") else {}

        return {
            "skip_valid_steps": int(skip_valid_steps),
            "wallclock": candidate.get("start_wallclock"),
            "target_date": candidate.get("target_date") or baseline.get("target_date"),
            "current_observation": candidate.get("start_observation"),
            "policy": policy,
            "setpoints": first_trace.get("setpoints"),
            "raw_setpoints": first_trace.get("raw_setpoints"),
            "day_return": candidate_day_return,
            "baseline_day_return": baseline_day_return,
            "relative_day_return": relative_day_return,
            "baseline_action": baseline_action,
            "candidate_control_steps_applied": int(candidate["control_steps_applied"]),
            "baseline_control_steps_applied": int(baseline["control_steps_applied"]),
            "candidate_start_wallclock": candidate.get("start_wallclock"),
            "candidate_end_wallclock": candidate.get("end_wallclock"),
            "baseline_start_wallclock": baseline.get("start_wallclock"),
            "baseline_end_wallclock": baseline.get("end_wallclock"),
            "candidate_reward_trace": candidate.get("reward_trace", []),
            "baseline_reward_trace": baseline.get("reward_trace", []),
            "candidate_action_trace": candidate.get("action_trace", []),
            "baseline_action_trace": baseline.get("action_trace", []),
            "invalid_observation_skips": int(candidate.get("invalid_observation_skips", 0)),
            "ineligible_window_skips": int(candidate.get("ineligible_window_skips", 0)),
            "baseline_cache_hit": bool(baseline_cache_hit),
        }

    def evaluate_planner_workday_closed_loop(
        self,
        *,
        skip_valid_steps: int,
        planner: LLMSetpointPlanner,
        previous_action: dict[str, Any] | None = None,
        baseline_action: dict[str, dict[str, float]] | None = None,
        max_control_steps: int | None = None,
    ) -> dict[str, Any]:
        initial_previous_action = previous_action or self.default_action()
        baseline_action = baseline_action or self.default_action()

        candidate = self._rollout_workday_requery_planner(
            skip_valid_steps=skip_valid_steps,
            planner=planner,
            previous_action=initial_previous_action,
            max_control_steps=max_control_steps,
        )
        baseline, baseline_cache_hit = self._get_or_rollout_baseline_workday(
            skip_valid_steps=skip_valid_steps,
            baseline_action=baseline_action,
            previous_action=initial_previous_action,
            max_control_steps=max_control_steps,
        )

        candidate_day_return = float(candidate["total_reward"])
        baseline_day_return = float(baseline["total_reward"])
        relative_day_return = candidate_day_return - baseline_day_return
        first_trace = candidate.get("action_trace", [{}])[0] if candidate.get("action_trace") else {}

        return {
            "skip_valid_steps": int(skip_valid_steps),
            "wallclock": candidate.get("start_wallclock"),
            "target_date": candidate.get("target_date") or baseline.get("target_date"),
            "current_observation": candidate.get("start_observation"),
            "setpoints": first_trace.get("setpoints"),
            "sanitized_setpoints": first_trace.get("sanitized_setpoints"),
            "day_return": candidate_day_return,
            "baseline_day_return": baseline_day_return,
            "relative_day_return": relative_day_return,
            "baseline_action": baseline_action,
            "candidate_control_steps_applied": int(candidate["control_steps_applied"]),
            "baseline_control_steps_applied": int(baseline["control_steps_applied"]),
            "candidate_start_wallclock": candidate.get("start_wallclock"),
            "candidate_end_wallclock": candidate.get("end_wallclock"),
            "baseline_start_wallclock": baseline.get("start_wallclock"),
            "baseline_end_wallclock": baseline.get("end_wallclock"),
            "candidate_reward_trace": candidate.get("reward_trace", []),
            "baseline_reward_trace": baseline.get("reward_trace", []),
            "candidate_action_trace": candidate.get("action_trace", []),
            "baseline_action_trace": baseline.get("action_trace", []),
            "invalid_observation_skips": int(candidate.get("invalid_observation_skips", 0)),
            "ineligible_window_skips": int(candidate.get("ineligible_window_skips", 0)),
            "planner_mode": "rolling_step_action",
            "baseline_cache_hit": bool(baseline_cache_hit),
        }

    # ------------------------------------------------------------------
    # Block-based 3h planning
    # ------------------------------------------------------------------

    BLOCK_DEFINITIONS: list[tuple[time, time]] = [
        (time(6, 0), time(7, 0)),
        (time(7, 0), time(8, 0)),
        (time(8, 0), time(9, 0)),
        (time(9, 0), time(10, 0)),
        (time(10, 0), time(11, 0)),
        (time(11, 0), time(12, 0)),
        (time(12, 0), time(13, 0)),
        (time(13, 0), time(14, 0)),
        (time(14, 0), time(15, 0)),
        (time(15, 0), time(16, 0)),
        (time(16, 0), time(17, 0)),
        (time(17, 0), time(18, 0)),
        (time(18, 0), time(19, 0)),
    ]
    STEP_MINUTES = 10          # env step duration
    KNOT_ENV_STEPS = 3         # 30min / 10min = 3 env steps per knot

    @classmethod
    def _block_env_steps(cls, block_index: int) -> int:
        """Number of env steps for a given block (handles variable-length last block)."""
        start, end = cls.BLOCK_DEFINITIONS[block_index]
        minutes = (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
        return minutes // cls.STEP_MINUTES

    @classmethod
    def _block_knots(cls, block_index: int) -> int:
        """Number of knots for a given block."""
        return cls._block_env_steps(block_index) // cls.KNOT_ENV_STEPS

    def _rollout_block_with_replay(
        self,
        *,
        skip_valid_steps: int,
        replay_actions: list[dict[str, dict[str, float]]],
        block_actions: list[dict[str, dict[str, float]]],
        baseline_action: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Run one EP instance: skip → replay previous blocks → execute *block_actions* for current block.

        *replay_actions* is a flat list of per-env-step actions from all previous winner blocks.
        *block_actions* is an 18-element list (one action dict per env step) for the current block.
        Steps outside the replay+block range use *baseline_action*.

        Returns dict with keys: block_reward, block_reward_trace, block_action_trace,
        total_reward, total_reward_trace, control_steps_applied, target_date, ...
        """
        report_subdir = f"gspo_block_{uuid.uuid4().hex[:10]}"
        env, system = self._make_env_and_system(report_subdir)

        replay_len = len(replay_actions)
        block_len = len(block_actions)

        state = {
            "skip_remaining": int(skip_valid_steps),
            "phase": "seek_start",
            "last_seen_wallclock": None,
            "target_date": None,
            "pending_reward": False,
            "control_steps_applied": 0,
            "total_reward": 0.0,
            "block_reward": 0.0,
            "reward_trace": [],
            "block_reward_trace": [],
            "action_trace": [],
            "block_action_trace": [],
            "start_wallclock": None,
            "block_start_wallclock": None,
            "end_wallclock": None,
            "action_cursor": 0,
            "block_start_observation": None,
        }
        result: dict[str, Any] = {}

        @system.events["timestep"].on
        def _on_timestep(_):
            self.env_mod.update_forecast_observation(system["wallclock"].value)
            wallclock_key = str(system["wallclock"].value)
            if state["last_seen_wallclock"] == wallclock_key:
                return
            state["last_seen_wallclock"] = wallclock_key

            ts = pd.Timestamp(system["wallclock"].value)
            if ts.tzinfo is not None or ts.tz is not None:
                ts = ts.tz_localize(None)

            try:
                observation = deepcopy(env._coerce_observation(env.agent.observation.value))
                observation_plausible = self.observation_is_plausible(observation)
            except self.env_mod.TemporaryUnavailableError:
                return
            except Exception:
                observation = None
                observation_plausible = False

            is_eligible = False
            if observation_plausible and observation is not None:
                is_eligible = self.wallclock_is_eligible(system["wallclock"].value, observation)

            # --- seek_start: skip until first eligible step ---
            if state["phase"] == "seek_start":
                if not observation_plausible or observation is None or not is_eligible:
                    return
                if state["skip_remaining"] > 0:
                    state["skip_remaining"] -= 1
                    return
                state["phase"] = "running"
                state["target_date"] = ts.date()
                state["start_wallclock"] = str(system["wallclock"].value)

            # --- collect reward from previous step ---
            if state["pending_reward"]:
                reward = float(env.agent.reward.value)
                physics = _extract_step_physics(observation, self.zone_ids) if observation is not None else {}
                state["total_reward"] += reward
                state["reward_trace"].append({
                    "wallclock": str(system["wallclock"].value),
                    "reward": reward,
                    **physics,
                })
                cursor = state["action_cursor"]
                if cursor > replay_len:
                    state["block_reward"] += reward
                    state["block_reward_trace"].append({
                        "wallclock": str(system["wallclock"].value),
                        "reward": reward,
                        **physics,
                    })
                state["pending_reward"] = False

            # --- decide action ---
            cursor = state["action_cursor"]

            # Capture observation at the replay→block boundary
            if cursor >= replay_len and state["block_start_observation"] is None:
                if observation is not None:
                    state["block_start_observation"] = deepcopy(observation)
                if state["block_start_wallclock"] is None:
                    state["block_start_wallclock"] = str(system["wallclock"].value)

            total_target = replay_len + block_len
            same_day = ts.date() == state["target_date"]

            if not same_day or cursor >= total_target or not observation_plausible or observation is None or not is_eligible:
                state["end_wallclock"] = str(system["wallclock"].value)
                result.update({
                    "target_date": None if state["target_date"] is None else str(state["target_date"]),
                    "start_wallclock": state["start_wallclock"],
                    "block_start_wallclock": state["block_start_wallclock"],
                    "end_wallclock": state["end_wallclock"],
                    "total_reward": float(state["total_reward"]),
                    "block_reward": float(state["block_reward"]),
                    "control_steps_applied": int(state["control_steps_applied"]),
                    "reward_trace": state["reward_trace"],
                    "block_reward_trace": state["block_reward_trace"],
                    "action_trace": state["action_trace"],
                    "block_action_trace": state["block_action_trace"],
                    "block_start_observation": state["block_start_observation"],
                })
                try:
                    system.stop()
                except Exception:
                    pass
                return

            if cursor < replay_len:
                action = replay_actions[cursor]
                mode = "replay"
            else:
                block_idx = cursor - replay_len
                action = block_actions[block_idx]
                mode = "block"

            env.agent.action.value = action
            state["pending_reward"] = True
            state["control_steps_applied"] += 1
            state["action_cursor"] += 1

            trace_entry = {
                "wallclock": str(system["wallclock"].value),
                "mode": mode,
                "action": _plainify(action),
            }
            state["action_trace"].append(trace_entry)
            if mode == "block":
                state["block_action_trace"].append(trace_entry)

        system.add("logging:progress").start().wait()
        if not result:
            raise RuntimeError("Failed to roll out block in Miami GRPO bandit environment.")
        return result

    def _expand_knots_to_env_steps(
        self,
        knots: list[dict[str, float]],
        *,
        block_index: int = 0,
        allow_partial: bool = False,
    ) -> list[dict[str, dict[str, float]]]:
        """Expand knot setpoints into env-step action dicts.

        Each knot is a dict mapping zone_id -> setpoint_c.
        Each knot is repeated KNOT_ENV_STEPS times.

        When allow_partial=True, accepts fewer knots than expected
        (e.g. last block of the day may be cut short by EP clock jitter).
        """
        expected_steps = self._block_env_steps(block_index)
        expected_knots = self._block_knots(block_index)
        actions: list[dict[str, dict[str, float]]] = []
        for knot in knots:
            action = {zone_id: {"thermostat": float(knot[zone_id])} for zone_id in self.zone_ids}
            for _ in range(self.KNOT_ENV_STEPS):
                actions.append(deepcopy(action))
        if not allow_partial and len(actions) != expected_steps:
            raise ValueError(
                f"Expected {expected_steps} env steps from {expected_knots} knots, got {len(actions)}"
            )
        return actions

    def _rollout_block_rolling(
        self,
        *,
        skip_valid_steps: int,
        replay_actions: list[dict[str, dict[str, float]]],
        baseline_action: dict[str, dict[str, float]],
        planner: Any,
        block_index: int,
        block_start: Any,
        block_end: Any,
        mode: str,
    ) -> dict[str, Any]:
        """Run one EP instance with rolling knot-by-knot planning within a block.

        Instead of receiving pre-computed block_actions, this method queries the
        planner at each knot boundary (every KNOT_ENV_STEPS env steps) using the
        current EP observation.

        Returns dict with keys: block_reward, block_reward_trace, block_action_trace,
        knot_plans (list of per-knot plan dicts for gradient computation), ...
        """
        import time as _time_mod
        _t_rollout_start = _time_mod.time()
        report_subdir = f"gspo_block_rolling_{uuid.uuid4().hex[:10]}"
        env, system = self._make_env_and_system(report_subdir)
        _t_env_ready = _time_mod.time()

        replay_len = len(replay_actions)

        state = {
            "skip_remaining": int(skip_valid_steps),
            "phase": "seek_start",
            "last_seen_wallclock": None,
            "target_date": None,
            "pending_reward": False,
            "control_steps_applied": 0,
            "total_reward": 0.0,
            "block_reward": 0.0,
            "reward_trace": [],
            "block_reward_trace": [],
            "action_trace": [],
            "block_action_trace": [],
            "start_wallclock": None,
            "block_start_wallclock": None,
            "block_start_observation": None,
            "end_wallclock": None,
            "action_cursor": 0,
            "block_env_step": 0,
            "current_knot_action": None,
            "knot_plans": [],
        }
        result: dict[str, Any] = {}

        @system.events["timestep"].on
        def _on_timestep(_):
            self.env_mod.update_forecast_observation(system["wallclock"].value)
            wallclock_key = str(system["wallclock"].value)
            if state["last_seen_wallclock"] == wallclock_key:
                return
            state["last_seen_wallclock"] = wallclock_key

            ts = pd.Timestamp(system["wallclock"].value)
            if ts.tzinfo is not None or ts.tz is not None:
                ts = ts.tz_localize(None)

            try:
                observation = deepcopy(env._coerce_observation(env.agent.observation.value))
                observation_plausible = self.observation_is_plausible(observation)
            except self.env_mod.TemporaryUnavailableError:
                return
            except Exception:
                observation = None
                observation_plausible = False

            is_eligible = False
            if observation_plausible and observation is not None:
                is_eligible = self.wallclock_is_eligible(system["wallclock"].value, observation)

            # --- seek_start: skip until first eligible step ---
            if state["phase"] == "seek_start":
                if not observation_plausible or observation is None or not is_eligible:
                    return
                if state["skip_remaining"] > 0:
                    state["skip_remaining"] -= 1
                    return
                state["phase"] = "running"
                state["target_date"] = ts.date()
                state["start_wallclock"] = str(system["wallclock"].value)

            # --- collect reward from previous step ---
            if state["pending_reward"]:
                reward = float(env.agent.reward.value)
                physics = _extract_step_physics(observation, self.zone_ids) if observation is not None else {}
                state["total_reward"] += reward
                state["reward_trace"].append({
                    "wallclock": str(system["wallclock"].value),
                    "reward": reward,
                    **physics,
                })
                cursor = state["action_cursor"]
                if cursor > replay_len:
                    state["block_reward"] += reward
                    state["block_reward_trace"].append({
                        "wallclock": str(system["wallclock"].value),
                        "reward": reward,
                        **physics,
                    })
                state["pending_reward"] = False

            # --- decide action ---
            cursor = state["action_cursor"]
            same_day = ts.date() == state["target_date"]
            block_done = state["block_env_step"] >= self._block_env_steps(block_index)

            if not same_day or not observation_plausible or observation is None or not is_eligible:
                _finish_rolling_block(state, result, system)
                return

            # Replay phase: use previous winner actions
            if cursor < replay_len:
                action = replay_actions[cursor]
                env.agent.action.value = action
                state["pending_reward"] = True
                state["control_steps_applied"] += 1
                state["action_cursor"] += 1
                state["action_trace"].append({
                    "wallclock": str(system["wallclock"].value),
                    "mode": "replay",
                    "action": _plainify(action),
                })
                return

            # Record block start
            if state["block_start_wallclock"] is None:
                state["block_start_wallclock"] = str(system["wallclock"].value)
                state["block_start_observation"] = deepcopy(observation)

            # Block done
            if block_done:
                _finish_rolling_block(state, result, system)
                return

            # Rolling knot planning: query planner every KNOT_ENV_STEPS
            knot_step_in_block = state["block_env_step"] % self.KNOT_ENV_STEPS
            if knot_step_in_block == 0:
                knot_index = state["block_env_step"] // self.KNOT_ENV_STEPS
                _t_knot_start = _time_mod.time()
                knot_plan = planner.plan_knot(
                    block_index=block_index,
                    knot_index=knot_index,
                    block_start=block_start,
                    block_end=block_end,
                    mode=mode,
                    observation=observation,
                    wallclock=system["wallclock"].value,
                )
                _t_knot_end = _time_mod.time()
                print(f"    [KNOT] block={block_index} mode={mode} knot={knot_index} "
                      f"time={_t_knot_end - _t_knot_start:.2f}s", flush=True)
                knot = knot_plan["knot"]
                state["current_knot_action"] = {
                    zone_id: {"thermostat": float(knot[zone_id])} for zone_id in self.zone_ids
                }
                state["knot_plans"].append(knot_plan)

            action = state["current_knot_action"]
            env.agent.action.value = action
            state["pending_reward"] = True
            state["control_steps_applied"] += 1
            state["action_cursor"] += 1
            state["block_env_step"] += 1
            state["block_action_trace"].append({
                "wallclock": str(system["wallclock"].value),
                "mode": "block_rolling",
                "action": _plainify(action),
            })
            state["action_trace"].append({
                "wallclock": str(system["wallclock"].value),
                "mode": "block_rolling",
                "action": _plainify(action),
            })

        def _finish_rolling_block(st, res, sys_obj):
            st["end_wallclock"] = str(sys_obj["wallclock"].value)
            res.update({
                "target_date": None if st["target_date"] is None else str(st["target_date"]),
                "start_wallclock": st["start_wallclock"],
                "block_start_wallclock": st["block_start_wallclock"],
                "end_wallclock": st["end_wallclock"],
                "total_reward": float(st["total_reward"]),
                "block_reward": float(st["block_reward"]),
                "control_steps_applied": int(st["control_steps_applied"]),
                "reward_trace": st["reward_trace"],
                "block_reward_trace": st["block_reward_trace"],
                "action_trace": st["action_trace"],
                "block_action_trace": st["block_action_trace"],
                "knot_plans": st["knot_plans"],
                "block_start_observation": st.get("block_start_observation"),
            })
            try:
                sys_obj.stop()
            except Exception:
                pass

        system.add("logging:progress").start().wait()
        _t_ep_done = _time_mod.time()
        _n_knots = len(state.get("knot_plans", []))
        print(f"  [TIMING] block={block_index} mode={mode} replay={replay_len} "
              f"env_make={_t_env_ready - _t_rollout_start:.1f}s "
              f"ep_run={_t_ep_done - _t_env_ready:.1f}s "
              f"total={_t_ep_done - _t_rollout_start:.1f}s "
              f"knots={_n_knots} steps={state.get('control_steps_applied', 0)}", flush=True)
        if not result:
            raise RuntimeError("Failed to roll out rolling block in Miami GRPO bandit environment.")
        return result

    def _probe_block_observation(
        self,
        *,
        skip_valid_steps: int,
        replay_actions: list[dict[str, dict[str, float]]],
        baseline_action: dict[str, dict[str, float]],
    ) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
        """Run EP with replay only (no block actions) to capture observation at block start.

        Returns (observation, wallclock) at the point where the current block would begin.
        For block 0, replay_actions should be empty.
        """
        result = self._rollout_block_with_replay(
            skip_valid_steps=skip_valid_steps,
            replay_actions=replay_actions,
            block_actions=[],
            baseline_action=baseline_action,
        )
        return result.get("block_start_observation"), result.get("block_start_wallclock")

    def _rollout_baseline_full_day_blocks(
        self,
        *,
        skip_valid_steps: int,
        baseline_action: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Run baseline (fixed action) for a full day, return per-block reward breakdown.

        Reuses _rollout_workday with the baseline action for both first and continuation.
        Then slices reward_trace into block-level sums based on BLOCK_DEFINITIONS.
        """
        cache_key = (int(skip_valid_steps), self._serialize_action(baseline_action), None)
        cached = self._baseline_workday_cache.get(cache_key)
        if cached is not None:
            baseline = deepcopy(cached)
            baseline_cache_hit = True
        else:
            baseline = self._rollout_workday(
                skip_valid_steps=skip_valid_steps,
                first_action=baseline_action,
                continuation_action=baseline_action,
                previous_action=baseline_action,
                max_control_steps=None,
            )
            self._baseline_workday_cache[cache_key] = deepcopy(baseline)
            baseline_cache_hit = False

        reward_trace = baseline.get("reward_trace", [])
        block_rewards = self._slice_reward_trace_into_blocks(reward_trace)
        baseline["block_rewards"] = block_rewards
        baseline["baseline_cache_hit"] = baseline_cache_hit
        return baseline

    def _slice_reward_trace_into_blocks(
        self,
        reward_trace: list[dict[str, Any]],
    ) -> list[float]:
        """Partition reward_trace into 4 block sums based on wallclock times."""
        block_sums = [0.0] * len(self.BLOCK_DEFINITIONS)
        for entry in reward_trace:
            try:
                ts = pd.Timestamp(entry["wallclock"])
                if ts.tzinfo is not None or ts.tz is not None:
                    ts = ts.tz_localize(None)
                t = ts.time()
                for block_idx, (bstart, bend) in enumerate(self.BLOCK_DEFINITIONS):
                    if bstart <= t < bend:
                        block_sums[block_idx] += float(entry["reward"])
                        break
            except Exception:
                continue
        return block_sums

    def evaluate_workday_blocks(
        self,
        *,
        skip_valid_steps: int,
        planner: Any,
        baseline_action: dict[str, dict[str, float]] | None = None,
        candidate_modes: list[str] | None = None,
        mode_selector: Any | None = None,
    ) -> dict[str, Any]:
        """Orchestrate block-based evaluation for one workday using rolling knot planning.

        For each block:
        1. If mode_selector is provided, let it choose a single mode (eval/deployment)
        2. Otherwise, run all candidate modes and select winner by reward (training)

        Returns a dict with per-block results and overall summary.
        """
        baseline_action = baseline_action or self.default_action()
        if candidate_modes is None:
            candidate_modes = ["comfort", "balanced", "energy_saving"]

        baseline_result = self._rollout_baseline_full_day_blocks(
            skip_valid_steps=skip_valid_steps,
            baseline_action=baseline_action,
        )
        baseline_block_rewards = baseline_result["block_rewards"]

        winner_actions_history: list[dict[str, dict[str, float]]] = []
        block_results: list[dict[str, Any]] = []

        for block_index, (block_start_time, block_end_time) in enumerate(self.BLOCK_DEFINITIONS):
            replay_actions = list(winner_actions_history)
            block_candidates: list[dict[str, Any]] = []

            # If mode_selector provided, only run the selected mode (eval/deployment)
            if mode_selector is not None:
                probe_obs, probe_wc = self._probe_block_observation(
                    skip_valid_steps=skip_valid_steps,
                    replay_actions=replay_actions,
                    baseline_action=baseline_action,
                )
                selected = mode_selector(
                    block_index=block_index,
                    block_start=str(block_start_time),
                    block_end=str(block_end_time),
                    observation=probe_obs,
                    wallclock=probe_wc,
                    candidate_modes=candidate_modes,
                )
                eval_modes = [selected]
            else:
                eval_modes = candidate_modes

            for mode in eval_modes:
                candidate_result = self._rollout_block_rolling(
                    skip_valid_steps=skip_valid_steps,
                    replay_actions=replay_actions,
                    baseline_action=baseline_action,
                    planner=planner,
                    block_index=block_index,
                    block_start=block_start_time,
                    block_end=block_end_time,
                    mode=mode,
                )

                baseline_block_reward = baseline_block_rewards[block_index]
                candidate_block_reward = candidate_result["block_reward"]
                relative_block_reward = candidate_block_reward - baseline_block_reward
                knot_plans = candidate_result.get("knot_plans", [])

                block_candidates.append({
                    "mode": mode,
                    "block_reward": candidate_block_reward,
                    "baseline_block_reward": baseline_block_reward,
                    "relative_block_reward": relative_block_reward,
                    "knot_plans": knot_plans,
                    "block_action_trace": candidate_result.get("block_action_trace", []),
                    "block_reward_trace": candidate_result.get("block_reward_trace", []),
                    "control_steps_applied": candidate_result.get("control_steps_applied", 0),
                    "target_date": candidate_result.get("target_date"),
                    "block_start_wallclock": candidate_result.get("block_start_wallclock"),
                })

            rewards = [c["relative_block_reward"] for c in block_candidates]
            winner_idx = int(max(range(len(rewards)), key=lambda i: rewards[i]))
            winner_candidate = block_candidates[winner_idx]
            winner_knots = [kp["knot"] for kp in winner_candidate["knot_plans"]]
            winner_block_actions_flat = self._expand_knots_to_env_steps(
                winner_knots, block_index=block_index, allow_partial=True,
            )
            winner_actions_history.extend(winner_block_actions_flat)

            block_results.append({
                "block_index": block_index,
                "block_start": str(block_start_time),
                "block_end": str(block_end_time),
                "candidates": block_candidates,
                "winner_index": winner_idx,
                "winner_mode": winner_candidate["mode"],
                "winner_relative_block_reward": winner_candidate["relative_block_reward"],
                "baseline_block_reward": baseline_block_rewards[block_index],
            })

        total_winner_reward = sum(
            br["winner_relative_block_reward"] for br in block_results
        )

        return {
            "skip_valid_steps": int(skip_valid_steps),
            "target_date": baseline_result.get("target_date"),
            "planner_mode": "block_based",
            "block_definitions": [
                {"start": str(bs), "end": str(be)}
                for bs, be in self.BLOCK_DEFINITIONS
            ],
            "block_results": block_results,
            "total_winner_relative_reward": total_winner_reward,
            "baseline_day_return": float(baseline_result["total_reward"]),
            "baseline_block_rewards": baseline_block_rewards,
            "baseline_cache_hit": baseline_result.get("baseline_cache_hit", False),
            "candidate_modes": candidate_modes,
        }
