from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np
import pythermalcomfort as pytc


DEFAULT_ZONE_IDS = (
    "1FNW",
    "1FNE",
    "0FNW",
    "0FNE",
    "1FSW",
    "1FSE",
    "0FSW",
    "0FSE",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _as_vector(value: Any, length: int = 6) -> list[float]:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return [0.0] * int(length)
    if arr.size == 0:
        return [0.0] * int(length)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), constant_values=0.0)
    return [float(x) for x in arr[:length]]


def _extract_json_payload(text: str) -> str:
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        return stripped
    match = re.search(r"(\{.*\}|\[.*\])", stripped, re.S)
    if not match:
        raise ValueError("No JSON payload found in planner response.")
    return match.group(0)


def _quantize(value: float, quantum: float | None) -> float:
    if quantum is None or quantum <= 0.0:
        return float(value)
    return round(value / quantum) * quantum


def joules_to_kwh(value: Any) -> float:
    return _as_float(value) / 3.6e6


def _format_optional(value: Any, digits: int = 2, default: str = "n/a") -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return f"{out:.{digits}f}"


def _zone_recommended_band(
    *,
    occupancy: float,
    drybulb_c: float,
    pmv: float | None,
    delta_from_mean_c: float,
) -> tuple[float, float, str]:
    if occupancy <= 0.0:
        return 26.0, 28.5, "unoccupied_setback"
    if pmv is not None and pmv <= -0.6:
        return 25.0, 27.0, "too_cold"
    if pmv is not None and pmv <= -0.3:
        return 24.5, 26.0, "slightly_cool"
    if pmv is not None and pmv >= 0.7:
        return 21.5, 23.5, "too_warm"
    if pmv is not None and pmv >= 0.3:
        return 22.5, 24.0, "slightly_warm"
    if drybulb_c <= 22.8:
        return 24.5, 26.5, "cool_zone"
    if drybulb_c >= 25.0:
        return 22.0, 24.0, "warm_zone"
    if delta_from_mean_c <= -0.35:
        return 24.5, 25.8, "cooler_than_peers"
    if delta_from_mean_c >= 0.35:
        return 23.0, 24.4, "warmer_than_peers"
    return 23.8, 25.2, "near_neutral"


def estimate_zone_pmv(
    *,
    temperature_drybulb: float,
    temperature_radiant: float,
    humidity: float,
    airspeed: float = 0.1,
    met: float = 1.0,
    clo: float = 0.5,
) -> float | None:
    try:
        pmv = pytc.models.pmv_ppd(
            temperature_drybulb,
            tr=temperature_radiant,
            vr=pytc.utilities.v_relative(v=airspeed, met=np.asarray(met)),
            rh=humidity,
            met=np.asarray(met),
            clo=pytc.utilities.clo_dynamic(clo=np.asarray(clo), met=np.asarray(met)),
            limit_inputs=False,
        )["pmv"]
    except Exception:
        return None
    pmv = _as_float(pmv, default=float("nan"))
    if not math.isfinite(pmv):
        return None
    return pmv


@dataclass
class PlannerConstraints:
    min_setpoint_c: float = 20.0
    max_setpoint_c: float = 30.0
    max_delta_per_step_c: float = 2.0
    fallback_setpoint_c: float = 24.0
    quantization_c: float | None = 0.1
    symmetry_temp_tol_c: float = 0.15
    symmetry_pmv_tol: float = 0.12
    symmetry_prev_setpoint_tol_c: float = 0.2
    monotonic_temp_margin_c: float = 0.2
    monotonic_pmv_margin: float = 0.08
    merge_only_if_diff_below_c: float = 0.1


@dataclass
class ZonePlannerState:
    zone_id: str
    temperature_drybulb_c: float
    temperature_radiant_c: float
    humidity_pct: float
    occupancy: float
    previous_setpoint_c: float
    estimated_pmv: float | None


@dataclass
class ForecastSummary:
    available: bool
    temperature_6h_c: list[float]
    humidity_6h_pct: list[float]
    cloudcover_6h_pct: list[float]
    precip_prob_6h_pct: list[float]
    precip_6h_mm: list[float]


@dataclass
class PlannerInput:
    timestamp_utc: str | None
    step_minutes: int
    facility_electricity_kwh: float
    pv_kwh: float
    net_grid_kwh: float
    zones: list[ZonePlannerState]
    forecast: ForecastSummary

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerRequest:
    system_prompt: str
    user_prompt: str
    payload: dict[str, Any]
    constraints: PlannerConstraints


class PlannerBackend(Protocol):
    def generate(self, request: PlannerRequest) -> Any:
        ...


class HeuristicPlannerBackend:
    def generate(self, request: PlannerRequest) -> dict[str, float]:
        payload = request.payload
        constraints = request.constraints
        forecast = payload["forecast"]
        hot_next_hours = max(forecast["temperature_6h_c"]) if forecast["available"] else None
        wet_next_hours = max(forecast["precip_prob_6h_pct"]) if forecast["available"] else None
        setpoints: dict[str, float] = {}
        for zone in payload["zones"]:
            previous = _as_float(zone["previous_setpoint_c"], constraints.fallback_setpoint_c)
            occupancy = _as_float(zone["occupancy"])
            drybulb = _as_float(zone["temperature_drybulb_c"])
            pmv = zone.get("estimated_pmv")
            setpoint = previous

            if occupancy > 0.0:
                if pmv is not None and pmv > 0.6:
                    setpoint -= min(2.0, 0.7 + 0.5 * (pmv - 0.6))
                elif pmv is not None and pmv < -0.6:
                    setpoint += min(1.8, 0.6 + 0.4 * (-0.6 - pmv))
                elif drybulb > 25.0:
                    setpoint -= 0.8
                elif drybulb < 23.0:
                    setpoint += 0.6
            else:
                setpoint = max(previous, 26.0)

            if occupancy > 0.0 and hot_next_hours is not None and hot_next_hours >= 34.0:
                setpoint -= 0.5
            if wet_next_hours is not None and wet_next_hours >= 70.0 and occupancy <= 0.0:
                setpoint += 0.5

            setpoints[zone["zone_id"]] = float(setpoint)
        return setpoints


class OpenAIResponsesBackend:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)

    def _maybe_disable_qwen_thinking(self, text: str) -> str:
        model_name = (self.model or "").lower()
        if "qwen3" in model_name and "/no_think" not in text:
            return "/no_think\n" + text
        return text

    def generate(self, request: PlannerRequest) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is not installed in the current environment.") from exc

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)
        if self.base_url:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": self._maybe_disable_qwen_thinking(request.user_prompt)},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_output_tokens,
            )
            try:
                return str(response.choices[0].message.content)
            except Exception as exc:
                raise RuntimeError("Planner response did not include chat completion text.") from exc

        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": self._maybe_disable_qwen_thinking(request.user_prompt)},
            ],
            max_output_tokens=self.max_output_tokens,
        )
        text = getattr(response, "output_text", None)
        if text:
            return str(text)
        raise RuntimeError("Planner response did not include output_text.")


class TransformersSamplingBackend:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_name: str | None = None,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = (
            model_name
            or getattr(getattr(model, "config", None), "_name_or_path", None)
            or getattr(tokenizer, "name_or_path", None)
            or ""
        )
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)

    def _maybe_disable_qwen_thinking(self, text: str) -> str:
        model_name = (self.model_name or "").lower()
        if "qwen3" in model_name and "/no_think" not in text:
            return "/no_think\n" + text
        return text

    def _input_device(self) -> Any:
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        try:
            return next(self.model.parameters()).device
        except Exception as exc:
            raise RuntimeError("Unable to determine model device for local transformers backend.") from exc

    def generate(self, request: PlannerRequest) -> str:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required for TransformersSamplingBackend.") from exc

        user_prompt = self._maybe_disable_qwen_thinking(request.user_prompt)
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = (
                f"System:\n{request.system_prompt}\n\n"
                f"User:\n{user_prompt}\n\n"
                "Assistant:\n"
            )

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = self._input_device()
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        do_sample = bool(self.temperature > 0.0 or self.top_p < 1.0 or self.top_k > 0)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.max_output_tokens),
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(float(self.temperature), 1e-5)
            if self.top_p < 1.0:
                generation_kwargs["top_p"] = float(self.top_p)
            if self.top_k > 0:
                generation_kwargs["top_k"] = int(self.top_k)
        if self.repetition_penalty and abs(self.repetition_penalty - 1.0) > 1e-6:
            generation_kwargs["repetition_penalty"] = float(self.repetition_penalty)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = generated[0][prompt_tokens:]
        text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return str(text).strip()


class LLMSetpointPlanner:
    def __init__(
        self,
        backend: PlannerBackend,
        *,
        constraints: PlannerConstraints | None = None,
        zone_ids: tuple[str, ...] = DEFAULT_ZONE_IDS,
        step_minutes: int = 10,
        candidate_count: int = 1,
        max_generation_attempts: int = 2,
    ):
        self.backend = backend
        self.constraints = constraints or PlannerConstraints()
        self.zone_ids = tuple(zone_ids)
        self.step_minutes = int(step_minutes)
        self.candidate_count = max(int(candidate_count), 1)
        self.max_generation_attempts = max(int(max_generation_attempts), 1)
        self._last_pv_timestamp_utc: datetime | None = None
        self._last_pv_kwh: float | None = None

    def _build_near_term_energy_hints(self, payload: dict[str, Any]) -> dict[str, Any]:
        forecast = payload.get("forecast", {})
        available = bool(forecast.get("available", False))
        precip_prob_6h = [float(x) for x in list(forecast.get("precip_prob_6h_pct", []))[:6]]
        precip_mm_6h = [float(x) for x in list(forecast.get("precip_6h_mm", []))[:6]]
        cloudcover_6h = [float(x) for x in list(forecast.get("cloudcover_6h_pct", []))[:6]]
        precip_prob_2h = precip_prob_6h[:2]
        precip_mm_2h = precip_mm_6h[:2]
        cloudcover_2h = cloudcover_6h[:2]

        pv_kwh_now = _as_float(payload.get("pv_kwh"))
        timestamp_utc = payload.get("timestamp_utc")
        recent_delta_kwh = None
        recent_trend = "unknown"

        ts = None
        if isinstance(timestamp_utc, str):
            try:
                ts = datetime.fromisoformat(timestamp_utc)
            except ValueError:
                ts = None

        if ts is not None:
            should_reset = False
            if self._last_pv_timestamp_utc is not None:
                dt_minutes = (ts - self._last_pv_timestamp_utc).total_seconds() / 60.0
                if dt_minutes <= 0.0 or dt_minutes > max(45.0, 3.0 * float(self.step_minutes)):
                    should_reset = True
            if should_reset:
                self._last_pv_timestamp_utc = None
                self._last_pv_kwh = None
            if self._last_pv_timestamp_utc is not None and self._last_pv_kwh is not None:
                recent_delta_kwh = float(pv_kwh_now - self._last_pv_kwh)
                if abs(recent_delta_kwh) < 0.25:
                    recent_trend = "flat"
                elif recent_delta_kwh > 0.0:
                    recent_trend = "rising"
                else:
                    recent_trend = "falling"
            self._last_pv_timestamp_utc = ts
            self._last_pv_kwh = float(pv_kwh_now)

        precip_prob_next_2h_max = max(precip_prob_2h) if available and precip_prob_2h else 0.0
        precip_mm_next_2h_sum = sum(precip_mm_2h) if available and precip_mm_2h else 0.0
        cloudcover_next_2h_max = max(cloudcover_2h) if available and cloudcover_2h else 0.0

        if pv_kwh_now < 2.0:
            near_term_pv_risk = "low"
        elif (
            precip_mm_next_2h_sum >= 5.0
            or precip_prob_next_2h_max >= 45.0
            or (precip_prob_next_2h_max >= 20.0 and cloudcover_next_2h_max >= 80.0)
            or (cloudcover_next_2h_max >= 95.0 and pv_kwh_now >= 4.0)
        ):
            near_term_pv_risk = "high"
        elif (
            precip_mm_next_2h_sum > 0.0
            or precip_prob_next_2h_max >= 20.0
            or cloudcover_next_2h_max >= 80.0
        ):
            near_term_pv_risk = "medium"
        else:
            near_term_pv_risk = "low"

        if near_term_pv_risk != "high" and recent_trend == "falling" and pv_kwh_now >= 4.0:
            near_term_pv_risk = "medium"

        return {
            "pv_kwh_now": float(pv_kwh_now),
            "pv_recent_delta_kwh": None if recent_delta_kwh is None else float(recent_delta_kwh),
            "pv_recent_trend": recent_trend,
            "precip_prob_next_2h_max_pct": float(precip_prob_next_2h_max),
            "precip_mm_next_2h_sum": float(precip_mm_next_2h_sum),
            "cloudcover_next_2h_max_pct": float(cloudcover_next_2h_max),
            "near_term_pv_risk": near_term_pv_risk,
        }

    def _extract_candidate_payloads(self, raw_output: Any) -> list[Any]:
        if isinstance(raw_output, str):
            parsed = json.loads(_extract_json_payload(raw_output))
        elif isinstance(raw_output, (dict, list)):
            parsed = raw_output
        else:
            raise TypeError("Planner output must be a dict, list, or JSON string.")

        if isinstance(parsed, dict):
            if "candidates" in parsed and isinstance(parsed["candidates"], list):
                return list(parsed["candidates"])
            compact_named_candidates = []
            for key, value in parsed.items():
                if not isinstance(value, list):
                    continue
                if len(value) != len(self.zone_ids):
                    continue
                try:
                    setpoints = {
                        zone_id: _as_float(candidate_value)
                        for zone_id, candidate_value in zip(self.zone_ids, value)
                    }
                except Exception:
                    continue
                compact_named_candidates.append({
                    "name": key,
                    "setpoints": setpoints,
                })
            if compact_named_candidates:
                return compact_named_candidates
            candidate_keys = [
                key for key, value in parsed.items()
                if isinstance(value, dict) and (key.startswith("candidate_") or key.startswith("option_"))
            ]
            if candidate_keys:
                return [parsed[key] for key in sorted(candidate_keys)]
            return [parsed]
        if isinstance(parsed, list):
            return list(parsed)
        raise TypeError("Planner JSON payload must be an object or list.")

    def _generate_candidate_payloads(self, request: PlannerRequest) -> tuple[str, list[Any]]:
        last_exc: Exception | None = None
        last_raw_output: str | None = None
        for _ in range(self.max_generation_attempts):
            raw_output = self.backend.generate(request)
            last_raw_output = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=True)
            try:
                payloads = self._extract_candidate_payloads(raw_output)
            except Exception as exc:
                last_exc = exc
                continue
            if payloads:
                return str(raw_output), payloads
        if last_exc is not None:
            raise RuntimeError(f"Planner candidate parsing failed after {self.max_generation_attempts} attempts: {last_exc!r}; raw_output={last_raw_output!r}") from last_exc
        raise RuntimeError("Planner did not return any candidate payloads.")

    def _extract_previous_setpoint(
        self,
        previous_action: dict[str, Any] | None,
        zone_id: str,
    ) -> float:
        if not previous_action:
            return self.constraints.fallback_setpoint_c
        candidate = previous_action.get(zone_id, self.constraints.fallback_setpoint_c)
        if isinstance(candidate, dict):
            candidate = candidate.get("thermostat", self.constraints.fallback_setpoint_c)
        return _as_float(candidate, self.constraints.fallback_setpoint_c)

    def build_input(
        self,
        observation: dict[str, dict[str, Any]],
        *,
        wallclock: Any = None,
        previous_action: dict[str, Any] | None = None,
    ) -> PlannerInput:
        zone_ids = tuple(zone_id for zone_id in self.zone_ids if zone_id in observation)
        if not zone_ids:
            raise ValueError("Observation does not contain any planner zone ids.")

        first_zone = observation[zone_ids[0]]
        facility_electricity_kwh = joules_to_kwh(first_zone.get("energy_consumption", 0.0))
        pv_kwh = joules_to_kwh(first_zone.get("PV", 0.0))
        net_grid_kwh = max(facility_electricity_kwh - pv_kwh, 0.0)

        if wallclock is None:
            timestamp_utc = None
        elif isinstance(wallclock, datetime):
            if wallclock.tzinfo is None:
                wallclock = wallclock.replace(tzinfo=timezone.utc)
            timestamp_utc = wallclock.astimezone(timezone.utc).isoformat()
        else:
            try:
                timestamp_utc = datetime.fromtimestamp(_as_float(wallclock), tz=timezone.utc).isoformat()
            except Exception:
                timestamp_utc = str(wallclock)

        zones: list[ZonePlannerState] = []
        for zone_id in zone_ids:
            zone_obs = observation[zone_id]
            drybulb = _as_float(zone_obs.get("temperature_drybulb"))
            radiant = _as_float(zone_obs.get("temperature:radiant"))
            humidity = _as_float(zone_obs.get("humidity"))
            occupancy = _as_float(zone_obs.get("occupancy"))
            previous_setpoint = self._extract_previous_setpoint(previous_action, zone_id)
            zones.append(
                ZonePlannerState(
                    zone_id=zone_id,
                    temperature_drybulb_c=drybulb,
                    temperature_radiant_c=radiant,
                    humidity_pct=humidity,
                    occupancy=occupancy,
                    previous_setpoint_c=previous_setpoint,
                    estimated_pmv=estimate_zone_pmv(
                        temperature_drybulb=drybulb,
                        temperature_radiant=radiant,
                        humidity=humidity,
                    ),
                )
            )

        forecast = ForecastSummary(
            available=bool(round(_as_float(first_zone.get("forecast_available", 0.0)))),
            temperature_6h_c=_as_vector(first_zone.get("forecast_temperature_6h", [])),
            humidity_6h_pct=_as_vector(first_zone.get("forecast_humidity_6h", [])),
            cloudcover_6h_pct=_as_vector(first_zone.get("forecast_cloudcover_6h", [])),
            precip_prob_6h_pct=_as_vector(first_zone.get("forecast_precip_prob_6h", [])),
            precip_6h_mm=_as_vector(first_zone.get("forecast_precip_6h", [])),
        )

        return PlannerInput(
            timestamp_utc=timestamp_utc,
            step_minutes=self.step_minutes,
            facility_electricity_kwh=facility_electricity_kwh,
            pv_kwh=pv_kwh,
            net_grid_kwh=net_grid_kwh,
            zones=zones,
            forecast=forecast,
        )

    def build_request(
        self,
        observation: dict[str, dict[str, Any]],
        *,
        wallclock: Any = None,
        previous_action: dict[str, Any] | None = None,
    ) -> PlannerRequest:
        planner_input = self.build_input(
            observation,
            wallclock=wallclock,
            previous_action=previous_action,
        )
        payload = planner_input.to_payload()
        near_term_energy_hints = self._build_near_term_energy_hints(payload)
        occupied_zones = [zone for zone in payload["zones"] if _as_float(zone["occupancy"]) > 0.0]
        occupied_zones_by_temp = sorted(
            occupied_zones,
            key=lambda zone: _as_float(zone["temperature_drybulb_c"]),
        )
        occupied_temps = [_as_float(zone["temperature_drybulb_c"]) for zone in occupied_zones]
        occupied_pmvs = [
            _as_float(zone["estimated_pmv"], default=float("nan"))
            for zone in occupied_zones
            if zone.get("estimated_pmv") is not None
        ]
        occupied_temp_mean = float(np.mean(occupied_temps)) if occupied_temps else 0.0
        occupied_temp_spread = (max(occupied_temps) - min(occupied_temps)) if occupied_temps else 0.0
        occupied_pmv_spread = (max(occupied_pmvs) - min(occupied_pmvs)) if occupied_pmvs else 0.0
        occupied_warm_count = sum(
            1
            for zone in occupied_zones
            if (
                (zone.get("estimated_pmv") is not None and _as_float(zone["estimated_pmv"]) >= 0.3)
                or _as_float(zone["temperature_drybulb_c"]) >= 24.8
            )
        )
        occupied_cool_count = sum(
            1
            for zone in occupied_zones
            if (
                (zone.get("estimated_pmv") is not None and _as_float(zone["estimated_pmv"]) <= -0.3)
                or _as_float(zone["temperature_drybulb_c"]) <= 23.0
            )
        )
        temp_rank_map = {
            zone["zone_id"]: rank
            for rank, zone in enumerate(
                sorted(payload["zones"], key=lambda zone: _as_float(zone["temperature_drybulb_c"]), reverse=True),
                start=1,
            )
        }
        zone_hints: list[dict[str, Any]] = []
        zone_hint_lines: list[str] = []
        for zone in payload["zones"]:
            zone_id = zone["zone_id"]
            occupancy = _as_float(zone["occupancy"])
            drybulb_c = _as_float(zone["temperature_drybulb_c"])
            pmv = zone.get("estimated_pmv")
            delta_from_mean_c = drybulb_c - occupied_temp_mean if occupancy > 0.0 and occupied_temps else 0.0
            band_low_c, band_high_c, label = _zone_recommended_band(
                occupancy=occupancy,
                drybulb_c=drybulb_c,
                pmv=pmv,
                delta_from_mean_c=delta_from_mean_c,
            )
            hint = {
                "zone_id": zone_id,
                "temp_rank_hot_first": int(temp_rank_map[zone_id]),
                "delta_from_occupied_mean_c": float(delta_from_mean_c),
                "recommended_band_c": [float(band_low_c), float(band_high_c)],
                "reason": label,
            }
            zone_hints.append(hint)
            zone_hint_lines.append(
                f"- {zone_id}: occ={_format_optional(occupancy, 0)}, "
                f"temp={_format_optional(drybulb_c)} C, "
                f"pmv={_format_optional(pmv)}, "
                f"prev={_format_optional(zone['previous_setpoint_c'])} C, "
                f"hot_rank={temp_rank_map[zone_id]}/{len(payload['zones'])}, "
                f"peer_delta={_format_optional(delta_from_mean_c)} C, "
                f"suggested={band_low_c:.1f}-{band_high_c:.1f} C, "
                f"reason={label}"
            )
        payload["planner_hints"] = {
            "occupied_temp_mean_c": float(occupied_temp_mean),
            "occupied_temp_spread_c": float(occupied_temp_spread),
            "occupied_pmv_spread": float(occupied_pmv_spread),
            "occupied_warm_count": int(occupied_warm_count),
            "occupied_cool_count": int(occupied_cool_count),
            "coldest_occupied_zone": occupied_zones_by_temp[0]["zone_id"] if occupied_zones_by_temp else None,
            "warmest_occupied_zone": occupied_zones_by_temp[-1]["zone_id"] if occupied_zones_by_temp else None,
            "near_term_energy_hints": near_term_energy_hints,
            "zone_hints": zone_hints,
        }
        coldest_occupied = occupied_zones_by_temp[0]["zone_id"] if occupied_zones_by_temp else None
        warmest_occupied = occupied_zones_by_temp[-1]["zone_id"] if occupied_zones_by_temp else None
        system_prompt = (
            "You are an HVAC planning assistant. "
            "Choose one cooling setpoint in Celsius for each zone for the next control step. "
            "Your goal is to maintain occupied-zone comfort while reducing unnecessary net grid purchase. "
            "Return JSON only. "
            "You may use differentiated zone setpoints whenever zone conditions differ in temperature, PMV, or occupancy. "
            "Do not force equal setpoints unless zones are truly indistinguishable. "
            "If occupied zones are warm or have positive PMV, you may cool them decisively. "
            "If occupied zones are cool or have negative PMV, raise setpoints to avoid overcooling. "
            "Unoccupied zones should usually have higher setpoints unless there is a strong reason not to. "
            "A hotter occupied zone must not receive a higher setpoint than a cooler occupied zone. "
            "Rain or high precipitation probability in the next 1-2 hours can reduce PV availability and increase later grid dependence."
            " High cloud cover in the next 1-2 hours can also weaken PV even if rain totals remain small."
        )
        zone_keys = [zone["zone_id"] for zone in payload["zones"]]
        output_instruction = (
            "Return a JSON object with exactly these top-level keys and numeric Celsius values: "
            f"{zone_keys}.\n"
        )
        if self.candidate_count > 1:
            output_instruction = (
                "Return exactly one compact JSON object and nothing else.\n"
                "Use this fixed schema with arrays in the fixed zone order below:\n"
                "{\n"
                '  "comfort_first": [v1, v2, ...],\n'
                '  "balanced": [v1, v2, ...],\n'
                '  "energy_saving": [v1, v2, ...]\n'
                "}\n"
                f"Generate exactly {self.candidate_count} candidates.\n"
                f"Fixed zone order: {zone_keys}.\n"
                "Each array must contain exactly one numeric Celsius value per zone in that order.\n"
                "The candidates must be meaningfully different; do not return duplicates.\n"
                "Candidate roles and separation rules:\n"
                "- candidate 1 comfort_first: for warm or positive-PMV occupied zones, usually 0.5-1.0 C lower than balanced.\n"
                "- candidate 2 balanced: middle choice between comfort and net grid.\n"
                "- candidate 3 energy_saving: for occupied zones, usually 0.5-1.0 C higher than balanced when PMV allows; unoccupied zones should be the highest.\n"
                "If the three candidates would otherwise be identical, force at least one occupied-zone difference of 0.5 C while staying within comfort limits and hard bounds.\n"
                "If you generate more than 3 candidates, continue with distinct trade-offs using additional keys.\n"
            )
        user_prompt = (
            f"{output_instruction}"
            "Do not include markdown or explanations.\n"
            f"Hard bounds: {self.constraints.min_setpoint_c} to {self.constraints.max_setpoint_c} C.\n"
            f"Soft rate limit relative to previous setpoint: about +/-{self.constraints.max_delta_per_step_c} C.\n"
            f"Round each output value to the nearest {self.constraints.quantization_c:.1f} C.\n"
            "Use the full 20-30 C range when justified by occupancy, PMV, temperature, and forecast.\n"
            "Do not simply repeat the previous setpoint for every zone.\n"
            "If a zone is unoccupied, prefer a higher setpoint than an occupied zone unless the current state strongly suggests otherwise.\n"
            "Decision guide for occupied zones:\n"
            "- If dry-bulb is low or PMV is negative, prefer a higher setpoint to avoid overcooling.\n"
            "- If dry-bulb is near neutral and PMV is near neutral, moderate setpoints are appropriate.\n"
            "- If dry-bulb is high or PMV is positive, use lower setpoints as needed to recover comfort.\n"
            "- When multiple occupied zones differ, allow meaningful zone-to-zone differentiation if it improves comfort or avoids unnecessary cooling.\n"
            "- Do not keep zones artificially equal when one zone is clearly warmer or less comfortable.\n"
            "- Use lower setpoints aggressively when the zone is persistently warm, not just when discomfort is extreme.\n"
            "- Use higher setpoints aggressively for unoccupied zones when comfort is not needed.\n"
            "- If two occupied zones differ in temperature, the colder one should get an equal or higher setpoint than the warmer one.\n"
            f"Coldest occupied zone right now: {coldest_occupied}.\n"
            f"Warmest occupied zone right now: {warmest_occupied}.\n"
            f"Occupied-zone temperature spread: {occupied_temp_spread:.2f} C.\n"
            f"Occupied-zone PMV spread: {occupied_pmv_spread:.2f}.\n"
            f"Occupied warm-zone count: {occupied_warm_count}.\n"
            f"Occupied cool-zone count: {occupied_cool_count}.\n"
            f"Current PV generation: {near_term_energy_hints['pv_kwh_now']:.2f} kWh.\n"
            f"Recent PV trend: {near_term_energy_hints['pv_recent_trend']} "
            f"({_format_optional(near_term_energy_hints['pv_recent_delta_kwh'], default='n/a')} kWh vs previous step).\n"
            f"Max precipitation probability in next 2 hours: {near_term_energy_hints['precip_prob_next_2h_max_pct']:.1f}%.\n"
            f"Expected precipitation sum in next 2 hours: {near_term_energy_hints['precip_mm_next_2h_sum']:.1f} mm.\n"
            f"Max cloud cover in next 2 hours: {near_term_energy_hints['cloudcover_next_2h_max_pct']:.1f}%.\n"
            f"Near-term PV risk: {near_term_energy_hints['near_term_pv_risk']}.\n"
            "Near-term rain risk can reduce PV availability and increase later grid dependence.\n"
            "High cloud cover in the next 1-2 hours is a stronger direct sign that PV may weaken, even when precipitation totals stay small.\n"
            "Treat precipitation probability as uncertainty about weather deterioration, and treat cloud cover as a more direct indicator of reduced solar availability.\n"
            "If near-term PV risk is elevated and occupancy remains meaningful, modest pre-cooling before PV availability worsens can be beneficial.\n"
            "Do not overreact to distant rain; prioritize rain risk within the next 1-2 hours.\n"
            "If occupied zones are nearly identical, equality is acceptable, but do not force it when differentiation is useful.\n"
            "Only avoid large zone-to-zone differences when temperature, PMV, and occupancy are all truly similar.\n"
            "Short examples:\n"
            "- Example A: if all occupied zones are near 24.0 C, PMV near 0.0, and previous setpoint is 24.0 C, returning nearly equal values is acceptable.\n"
            "- Example B: if most occupied zones are near 24.8 C with PMV around 0.3 and previous setpoint is 24.5 C, returning a common lower value such as 23.8-24.0 C for most occupied zones is acceptable.\n"
            "- Example C: if one occupied zone is clearly hotter or has higher PMV than another, the hotter zone should receive an equal or lower setpoint, and the difference may be material.\n"
            "Per-zone decision hints:\n"
            f"{chr(10).join(zone_hint_lines)}\n"
            "Current building and forecast summary JSON:\n"
            f"{json.dumps(payload, ensure_ascii=True)}"
        )
        return PlannerRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            payload=payload,
            constraints=self.constraints,
        )

    def sanitize_setpoints(
        self,
        raw_output: Any,
        *,
        previous_action: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if isinstance(raw_output, str):
            parsed = json.loads(_extract_json_payload(raw_output))
        elif isinstance(raw_output, dict):
            parsed = raw_output
        else:
            raise TypeError("Planner output must be a dict or JSON string.")

        if "setpoints" in parsed and isinstance(parsed["setpoints"], dict):
            parsed = parsed["setpoints"]

        sanitized: dict[str, float] = {}
        for zone_id in self.zone_ids:
            previous = self._extract_previous_setpoint(previous_action, zone_id)
            value = _as_float(parsed.get(zone_id, previous), previous)
            value = max(self.constraints.min_setpoint_c, min(self.constraints.max_setpoint_c, value))
            value = max(previous - self.constraints.max_delta_per_step_c, min(previous + self.constraints.max_delta_per_step_c, value))
            value = _quantize(value, self.constraints.quantization_c)
            sanitized[zone_id] = float(value)
        return sanitized

    def _requantize_bounded_value(
        self,
        *,
        zone_id: str,
        value: float,
        previous_action: dict[str, Any] | None = None,
    ) -> float:
        previous = self._extract_previous_setpoint(previous_action, zone_id)
        value = max(self.constraints.min_setpoint_c, min(self.constraints.max_setpoint_c, float(value)))
        value = max(previous - self.constraints.max_delta_per_step_c, min(previous + self.constraints.max_delta_per_step_c, value))
        value = _quantize(value, self.constraints.quantization_c)
        return float(value)

    def _zones_are_near_symmetric(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_occ = _as_float(left.get("occupancy"))
        right_occ = _as_float(right.get("occupancy"))
        if round(left_occ) != round(right_occ):
            return False
        if abs(_as_float(left.get("temperature_drybulb_c")) - _as_float(right.get("temperature_drybulb_c"))) > self.constraints.symmetry_temp_tol_c:
            return False
        left_prev = _as_float(left.get("previous_setpoint_c"), self.constraints.fallback_setpoint_c)
        right_prev = _as_float(right.get("previous_setpoint_c"), self.constraints.fallback_setpoint_c)
        if abs(left_prev - right_prev) > self.constraints.symmetry_prev_setpoint_tol_c:
            return False
        left_pmv = left.get("estimated_pmv")
        right_pmv = right.get("estimated_pmv")
        if left_pmv is None or right_pmv is None:
            return True
        return abs(_as_float(left_pmv) - _as_float(right_pmv)) <= self.constraints.symmetry_pmv_tol

    def _left_is_materially_hotter(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        if _as_float(left.get("occupancy")) <= 0.0 or _as_float(right.get("occupancy")) <= 0.0:
            return False
        temp_diff = _as_float(left.get("temperature_drybulb_c")) - _as_float(right.get("temperature_drybulb_c"))
        left_pmv = left.get("estimated_pmv")
        right_pmv = right.get("estimated_pmv")
        if left_pmv is None or right_pmv is None:
            return temp_diff >= self.constraints.monotonic_temp_margin_c
        pmv_diff = _as_float(left_pmv) - _as_float(right_pmv)
        if temp_diff >= self.constraints.monotonic_temp_margin_c and pmv_diff >= -0.03:
            return True
        if pmv_diff >= self.constraints.monotonic_pmv_margin and temp_diff >= -0.1:
            return True
        return False

    def post_check_setpoints(
        self,
        setpoints: dict[str, float],
        *,
        request: PlannerRequest,
        previous_action: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        corrected = {zone_id: float(setpoints[zone_id]) for zone_id in self.zone_ids}
        zone_meta = {
            zone["zone_id"]: zone
            for zone in request.payload["zones"]
            if zone["zone_id"] in corrected
        }
        zone_ids = [zone_id for zone_id in self.zone_ids if zone_id in zone_meta]

        for idx, left_id in enumerate(zone_ids):
            for right_id in zone_ids[idx + 1:]:
                left = zone_meta[left_id]
                right = zone_meta[right_id]
                if not self._zones_are_near_symmetric(left, right):
                    continue
                if abs(corrected[left_id] - corrected[right_id]) > self.constraints.merge_only_if_diff_below_c:
                    continue
                merged = _quantize((corrected[left_id] + corrected[right_id]) / 2.0, self.constraints.quantization_c)
                corrected[left_id] = self._requantize_bounded_value(
                    zone_id=left_id,
                    value=merged,
                    previous_action=previous_action,
                )
                corrected[right_id] = self._requantize_bounded_value(
                    zone_id=right_id,
                    value=merged,
                    previous_action=previous_action,
                )

        for _ in range(len(zone_ids)):
            updated = False
            for idx, left_id in enumerate(zone_ids):
                for right_id in zone_ids[idx + 1:]:
                    left = zone_meta[left_id]
                    right = zone_meta[right_id]
                    if self._left_is_materially_hotter(left, right):
                        if corrected[left_id] > corrected[right_id]:
                            corrected[left_id] = self._requantize_bounded_value(
                                zone_id=left_id,
                                value=corrected[right_id],
                                previous_action=previous_action,
                            )
                            updated = True
                    elif self._left_is_materially_hotter(right, left):
                        if corrected[right_id] > corrected[left_id]:
                            corrected[right_id] = self._requantize_bounded_value(
                                zone_id=right_id,
                                value=corrected[left_id],
                                previous_action=previous_action,
                            )
                            updated = True
            if not updated:
                break

        return corrected

    def score_setpoints(
        self,
        setpoints: dict[str, float],
        *,
        request: PlannerRequest,
        previous_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        zone_meta = {
            zone["zone_id"]: zone
            for zone in request.payload["zones"]
            if zone["zone_id"] in setpoints
        }
        hint_map = {
            hint["zone_id"]: hint
            for hint in request.payload.get("planner_hints", {}).get("zone_hints", [])
            if hint["zone_id"] in setpoints
        }

        band_penalty = 0.0
        movement_penalty = 0.0
        unoccupied_low_penalty = 0.0
        symmetry_penalty = 0.0
        monotonic_penalty = 0.0

        occupied_zone_ids = []

        for zone_id, value in setpoints.items():
            zone = zone_meta[zone_id]
            hint = hint_map.get(zone_id, {})
            occupancy = _as_float(zone.get("occupancy"))
            low, high = hint.get("recommended_band_c", [24.0, 25.0])
            if value < low:
                band_penalty += (low - value) * (3.0 if occupancy > 0.0 else 1.2)
            elif value > high:
                band_penalty += (value - high) * (2.0 if occupancy > 0.0 else 0.8)

            previous = self._extract_previous_setpoint(previous_action, zone_id)
            movement_penalty += 0.04 * abs(value - previous)
            if occupancy <= 0.0:
                unoccupied_low_penalty += max(26.0 - value, 0.0) * 0.8
            else:
                occupied_zone_ids.append(zone_id)

        for idx, left_id in enumerate(occupied_zone_ids):
            for right_id in occupied_zone_ids[idx + 1:]:
                left = zone_meta[left_id]
                right = zone_meta[right_id]
                diff = abs(setpoints[left_id] - setpoints[right_id])
                if self._zones_are_near_symmetric(left, right):
                    symmetry_penalty += 0.35 * diff
                if self._left_is_materially_hotter(left, right) and setpoints[left_id] > setpoints[right_id]:
                    monotonic_penalty += 4.0 * (setpoints[left_id] - setpoints[right_id])
                if self._left_is_materially_hotter(right, left) and setpoints[right_id] > setpoints[left_id]:
                    monotonic_penalty += 4.0 * (setpoints[right_id] - setpoints[left_id])

        total = (
            band_penalty
            + movement_penalty
            + unoccupied_low_penalty
            + symmetry_penalty
            + monotonic_penalty
        )
        return {
            "total": float(total),
            "band_penalty": float(band_penalty),
            "movement_penalty": float(movement_penalty),
            "unoccupied_low_penalty": float(unoccupied_low_penalty),
            "symmetry_penalty": float(symmetry_penalty),
            "monotonic_penalty": float(monotonic_penalty),
            "global_mode_penalty": 0.0,
        }

    def to_env_action(self, setpoints: dict[str, float]) -> dict[str, dict[str, float]]:
        return {
            zone_id: {"thermostat": float(setpoints[zone_id])}
            for zone_id in self.zone_ids
        }

    def plan_next_action_with_trace(
        self,
        observation: dict[str, dict[str, Any]],
        *,
        wallclock: Any = None,
        previous_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = self.build_request(
            observation,
            wallclock=wallclock,
            previous_action=previous_action,
        )
        raw_output, raw_candidates = self._generate_candidate_payloads(request)
        candidates = []
        for candidate_index, raw_candidate in enumerate(raw_candidates):
            sanitized_setpoints = self.sanitize_setpoints(
                raw_candidate,
                previous_action=previous_action,
            )
            setpoints = self.post_check_setpoints(
                sanitized_setpoints,
                request=request,
                previous_action=previous_action,
            )
            score = self.score_setpoints(
                setpoints,
                request=request,
                previous_action=previous_action,
            )
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_name": raw_candidate.get("name") if isinstance(raw_candidate, dict) else None,
                    "raw_output": raw_candidate,
                    "sanitized_setpoints": sanitized_setpoints,
                    "setpoints": setpoints,
                    "score": score,
                }
            )
        if not candidates:
            raise RuntimeError("Planner did not return any candidate setpoints.")
        best = min(candidates, key=lambda candidate: candidate["score"]["total"])
        action = self.to_env_action(best["setpoints"])
        return {
            "request": request,
            "raw_output": raw_output,
            "sanitized_setpoints": best["sanitized_setpoints"],
            "setpoints": best["setpoints"],
            "action": action,
            "candidate_count": len(candidates),
            "candidate_summaries": [
                {
                    "candidate_index": candidate["candidate_index"],
                    "candidate_name": candidate["candidate_name"],
                    "setpoints": candidate["setpoints"],
                    "score": candidate["score"],
                }
                for candidate in candidates
            ],
        }

    def plan_next_action(
        self,
        observation: dict[str, dict[str, Any]],
        *,
        wallclock: Any = None,
        previous_action: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, float]]:
        trace = self.plan_next_action_with_trace(
            observation,
            wallclock=wallclock,
            previous_action=previous_action,
        )
        return trace["action"]


# ======================================================================
# Block-based 3h planner
# ======================================================================

CANDIDATE_MODE_DESCRIPTIONS: dict[str, str] = {
    "comfort": (
        "Maximise occupant comfort: target PMV ≈ 0 (thermally neutral). "
        "Use LOWER setpoints (more cooling) to push PMV toward 0."
    ),
    "balanced": (
        "Balance comfort and energy: target PMV between 0 and +0.3. "
        "Use MODERATE setpoints — slightly higher than comfort mode to save some energy."
    ),
    "energy_saving": (
        "Minimise net grid energy: target PMV near +0.5 (slightly warm but acceptable). "
        "Use HIGHER setpoints (less cooling) to reduce energy. "
        "Set unoccupied zones near the upper bound."
    ),
}

# Brief PMV explanation embedded in system prompts.
PMV_EXPLANATION = (
    "PMV (Predicted Mean Vote) measures thermal comfort: "
    "0 = neutral (ideal), +0.5 = slightly warm, -0.5 = slightly cool. "
    "Relationship to cooling setpoint: LOWER setpoint → more cooling → LOWER PMV. "
    "HIGHER setpoint → less cooling → HIGHER PMV. "
    "For example, if a zone's current PMV is +0.4 and you want PMV ≈ 0, "
    "you need to LOWER the setpoint to increase cooling."
)

# Zone metadata for prompt enrichment.
# Houston 2-story office: 1F = upper floor, 0F = ground floor;
# N/S = north/south facing; W/E = west/east wing.
ZONE_DESCRIPTIONS: dict[str, str] = {
    "1FNW": "Upper north-west: minimal direct sun, roof heat gain",
    "1FNE": "Upper north-east: morning indirect light, roof heat gain",
    "0FNW": "Ground north-west: earth-cooled, least solar gain",
    "0FNE": "Ground north-east: mild morning light, earth-cooled",
    "1FSW": "Upper south-west: strong afternoon sun + roof, warmest zone",
    "1FSE": "Upper south-east: morning sun + roof, warm in AM",
    "0FSW": "Ground south-west: afternoon direct sun, moderate gain",
    "0FSE": "Ground south-east: morning direct sun, moderate gain",
}

KNOTS_PER_BLOCK = 6
BLOCK_MINUTES = 180
KNOT_MINUTES = 30


class BlockPlanner:
    """Plans a 3-hour block of HVAC setpoints as 6 half-hour knots."""

    def __init__(
        self,
        backend: PlannerBackend,
        *,
        constraints: PlannerConstraints | None = None,
        zone_ids: tuple[str, ...] = DEFAULT_ZONE_IDS,
        max_generation_attempts: int = 2,
    ):
        self.backend = backend
        self.constraints = constraints or PlannerConstraints()
        self.zone_ids = tuple(zone_ids)
        self.max_generation_attempts = max(int(max_generation_attempts), 1)
        self._last_observation: dict[str, dict[str, Any]] | None = None
        self._last_wallclock: Any = None

    def set_current_state(
        self,
        observation: dict[str, dict[str, Any]],
        wallclock: Any = None,
    ) -> None:
        """Cache the current observation so plan_block can use it."""
        self._last_observation = observation
        self._last_wallclock = wallclock

    def _build_block_system_prompt(self, mode: str) -> str:
        mode_desc = CANDIDATE_MODE_DESCRIPTIONS.get(mode, CANDIDATE_MODE_DESCRIPTIONS["balanced"])
        zone_desc_lines = []
        for zid in self.zone_ids:
            desc = ZONE_DESCRIPTIONS.get(zid, zid)
            zone_desc_lines.append(f"  {zid}: {desc}")
        zone_layout = "\n".join(zone_desc_lines)
        return (
            "You are an HVAC planning assistant for a 2-story 8-zone office building in Houston.\n"
            f"Output a 3-hour cooling setpoint plan for {len(self.zone_ids)} zones, "
            f"consisting of {KNOTS_PER_BLOCK} {KNOT_MINUTES}-minute control knots.\n"
            "Each knot specifies one cooling setpoint per zone.\n\n"
            f"{PMV_EXPLANATION}\n\n"
            "Zone layout (each zone has different solar exposure and thermal characteristics):\n"
            f"{zone_layout}\n\n"
            "Set each zone's setpoint based on its temperature, PMV, occupancy, and solar exposure. "
            "Adjust knots over time as conditions change.\n\n"
            f"Planning mode: {mode_desc}\n"
            "Return JSON only. No markdown, no explanations.\n"
            f"Hard bounds: {self.constraints.min_setpoint_c} to {self.constraints.max_setpoint_c} C.\n"
            "Round to nearest 0.1 C. Use the full range when justified."
        )

    def _build_block_user_prompt(
        self,
        *,
        block_index: int,
        block_start: Any,
        block_end: Any,
        mode: str,
        observation: dict[str, dict[str, Any]],
        wallclock: Any,
    ) -> str:
        zone_keys = list(self.zone_ids)
        zone_lines = []
        for zone_id in self.zone_ids:
            zone_obs = observation.get(zone_id, {})
            drybulb = _as_float(zone_obs.get("temperature_drybulb"))
            humidity = _as_float(zone_obs.get("humidity"))
            occupancy = _as_float(zone_obs.get("occupancy"))
            pmv = estimate_zone_pmv(
                temperature_drybulb=drybulb,
                temperature_radiant=_as_float(zone_obs.get("temperature:radiant")),
                humidity=humidity,
            )
            zone_desc = ZONE_DESCRIPTIONS.get(zone_id, "")
            zone_lines.append(
                f"- {zone_id} ({zone_desc}): temp={drybulb:.1f}C, humidity={humidity:.0f}%, "
                f"occupancy={occupancy:.0f}, PMV={pmv:.2f}"
            )

        first_zone = observation.get(self.zone_ids[0], {})
        pv_kwh = joules_to_kwh(first_zone.get("PV", 0.0))
        facility_kwh = joules_to_kwh(first_zone.get("energy_consumption", 0.0))
        net_grid_kwh = max(facility_kwh - pv_kwh, 0.0)

        forecast_lines = []
        forecast_available = bool(round(_as_float(first_zone.get("forecast_available", 0.0))))
        if forecast_available:
            temp_6h = [float(x) for x in list(first_zone.get("forecast_temperature_6h", []))[:6]]
            humidity_6h = [float(x) for x in list(first_zone.get("forecast_humidity_6h", []))[:6]]
            precip_prob_6h = [float(x) for x in list(first_zone.get("forecast_precip_prob_6h", []))[:6]]
            precip_6h = [float(x) for x in list(first_zone.get("forecast_precip_6h", []))[:6]]
            forecast_lines.append(f"Forecast temperature (next 6h): {temp_6h}")
            forecast_lines.append(f"Forecast humidity (next 6h): {humidity_6h}")
            forecast_lines.append(f"Forecast precip probability (next 6h): {precip_prob_6h}")
            forecast_lines.append(f"Forecast precipitation mm (next 6h): {precip_6h}")
            cloudcover_6h = list(first_zone.get("forecast_cloudcover_6h", []))
            if cloudcover_6h:
                forecast_lines.append(f"Forecast cloud cover (next 6h): {[float(x) for x in cloudcover_6h[:6]]}")

        prompt = (
            f"Block {block_index + 1}/4: {block_start} to {block_end}\n"
            f"Current time: {wallclock}\n"
            f"Current PV: {pv_kwh:.2f} kWh, Net grid: {net_grid_kwh:.2f} kWh\n"
            f"Zone order: {zone_keys}\n"
            "Current zone states:\n"
            f"{chr(10).join(zone_lines)}\n"
        )
        if forecast_lines:
            prompt += "Forecast:\n" + chr(10).join(forecast_lines) + "\n"

        prompt += (
            f"\nReturn a JSON object with a \"plan\" array of exactly {KNOTS_PER_BLOCK} objects.\n"
            "Each object has a \"slot\" (1-6) and a \"setpoints\" array of "
            f"{len(self.zone_ids)} numeric Celsius values in zone order.\n"
            "Round values to nearest 0.1 C.\n"
            "Each zone MUST get its own setpoint. Each knot SHOULD differ as conditions change.\n"
            "Example (note: different values per zone and per knot):\n"
            '{"plan": [\n'
            '  {"slot": 1, "setpoints": [24.1, 24.3, 25.2, 25.0, 23.4, 23.8, 24.6, 24.2]},\n'
            '  {"slot": 2, "setpoints": [24.0, 24.2, 25.1, 24.8, 23.3, 23.7, 24.5, 24.1]},\n'
            "  ...\n"
            "]}\n"
        )
        return prompt

    def _build_knot_system_prompt(self, mode: str) -> str:
        mode_desc = CANDIDATE_MODE_DESCRIPTIONS.get(mode, CANDIDATE_MODE_DESCRIPTIONS["balanced"])
        zone_desc_lines = []
        for zid in self.zone_ids:
            desc = ZONE_DESCRIPTIONS.get(zid, zid)
            zone_desc_lines.append(f"  {zid}: {desc}")
        zone_layout = "\n".join(zone_desc_lines)
        return (
            "You are an HVAC planning assistant for a 2-story 8-zone office building in Houston.\n"
            f"Output the next {KNOT_MINUTES}-minute cooling setpoint for each of the {len(self.zone_ids)} zones.\n\n"
            f"{PMV_EXPLANATION}\n\n"
            "Zone layout:\n"
            f"{zone_layout}\n\n"
            "Set each zone's setpoint based on its current temperature, PMV, occupancy, "
            "and solar exposure.\n\n"
            f"Planning mode: {mode_desc}\n"
            "Return JSON only. No markdown, no explanations.\n"
            f"Hard bounds: {self.constraints.min_setpoint_c} to {self.constraints.max_setpoint_c} C.\n"
            "Round to nearest 0.1 C."
        )

    def _build_knot_user_prompt(
        self,
        *,
        block_index: int,
        knot_index: int,
        block_start: Any,
        block_end: Any,
        mode: str,
        observation: dict[str, dict[str, Any]],
        wallclock: Any,
    ) -> str:
        zone_keys = list(self.zone_ids)
        zone_lines = []
        for zone_id in self.zone_ids:
            zone_obs = observation.get(zone_id, {})
            drybulb = _as_float(zone_obs.get("temperature_drybulb"))
            humidity = _as_float(zone_obs.get("humidity"))
            occupancy = _as_float(zone_obs.get("occupancy"))
            pmv = estimate_zone_pmv(
                temperature_drybulb=drybulb,
                temperature_radiant=_as_float(zone_obs.get("temperature:radiant")),
                humidity=humidity,
            )
            zone_desc = ZONE_DESCRIPTIONS.get(zone_id, "")
            zone_lines.append(
                f"- {zone_id} ({zone_desc}): temp={drybulb:.1f}C, humidity={humidity:.0f}%, "
                f"occupancy={occupancy:.0f}, PMV={pmv:.2f}"
            )

        first_zone = observation.get(self.zone_ids[0], {})
        pv_kwh = joules_to_kwh(first_zone.get("PV", 0.0))
        facility_kwh = joules_to_kwh(first_zone.get("energy_consumption", 0.0))
        net_grid_kwh = max(facility_kwh - pv_kwh, 0.0)

        forecast_lines = []
        forecast_available = bool(round(_as_float(first_zone.get("forecast_available", 0.0))))
        if forecast_available:
            temp_6h = [float(x) for x in list(first_zone.get("forecast_temperature_6h", []))[:6]]
            humidity_6h = [float(x) for x in list(first_zone.get("forecast_humidity_6h", []))[:6]]
            precip_prob_6h = [float(x) for x in list(first_zone.get("forecast_precip_prob_6h", []))[:6]]
            precip_6h = [float(x) for x in list(first_zone.get("forecast_precip_6h", []))[:6]]
            forecast_lines.append(f"Forecast temperature (next 6h): {temp_6h}")
            forecast_lines.append(f"Forecast humidity (next 6h): {humidity_6h}")
            forecast_lines.append(f"Forecast precip probability (next 6h): {precip_prob_6h}")
            forecast_lines.append(f"Forecast precipitation mm (next 6h): {precip_6h}")
            cloudcover_6h = list(first_zone.get("forecast_cloudcover_6h", []))
            if cloudcover_6h:
                forecast_lines.append(f"Forecast cloud cover (next 6h): {[float(x) for x in cloudcover_6h[:6]]}")

        prompt = (
            f"Block {block_index + 1}/4: {block_start} to {block_end}, "
            f"Knot {knot_index + 1}/{KNOTS_PER_BLOCK}\n"
            f"Current time: {wallclock}\n"
            f"Current PV: {pv_kwh:.2f} kWh, Net grid: {net_grid_kwh:.2f} kWh\n"
            f"Zone order: {zone_keys}\n"
            "Current zone states:\n"
            f"{chr(10).join(zone_lines)}\n"
        )
        if forecast_lines:
            prompt += "Forecast:\n" + chr(10).join(forecast_lines) + "\n"

        prompt += (
            f"\nReturn a JSON object with a \"setpoints\" array of "
            f"{len(self.zone_ids)} numeric Celsius values in zone order.\n"
            "Each zone MUST get its own setpoint based on its state.\n"
            "Example: {\"setpoints\": [24.1, 24.3, 25.2, 25.0, 23.4, 23.8, 24.6, 24.2]}\n"
        )
        return prompt

    def _parse_knot_output(self, raw_output: str) -> dict[str, float] | None:
        """Parse LLM output into a single knot dict (zone_id -> setpoint)."""
        try:
            json_text = _extract_json_payload(raw_output)
            data = json.loads(json_text)
        except Exception:
            return None

        setpoints_raw = None
        if isinstance(data, dict):
            setpoints_raw = data.get("setpoints")
        if not isinstance(setpoints_raw, list) or len(setpoints_raw) != len(self.zone_ids):
            return None

        knot: dict[str, float] = {}
        for zone_id, val in zip(self.zone_ids, setpoints_raw):
            try:
                v = float(val)
            except (TypeError, ValueError):
                v = self.constraints.fallback_setpoint_c
            v = max(self.constraints.min_setpoint_c, min(self.constraints.max_setpoint_c, v))
            if self.constraints.quantization_c and self.constraints.quantization_c > 0:
                v = _quantize(v, self.constraints.quantization_c)
            knot[zone_id] = v
        return knot

    def plan_knot(
        self,
        *,
        block_index: int,
        knot_index: int,
        block_start: Any,
        block_end: Any,
        mode: str = "balanced",
        observation: dict[str, dict[str, Any]] | None = None,
        wallclock: Any = None,
    ) -> dict[str, Any]:
        """Generate a single 30-min knot for the given mode and current observation.

        Returns dict with keys: knot, raw_output, system_prompt, user_prompt, mode.
        knot is a dict mapping zone_id -> setpoint_c.
        """
        obs = observation if observation is not None else self._last_observation
        wc = wallclock if wallclock is not None else self._last_wallclock
        if obs is None:
            raise ValueError("No observation available for knot planning.")

        system_prompt = self._build_knot_system_prompt(mode)
        user_prompt = self._build_knot_user_prompt(
            block_index=block_index,
            knot_index=knot_index,
            block_start=block_start,
            block_end=block_end,
            mode=mode,
            observation=obs,
            wallclock=wc,
        )
        request = PlannerRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            payload={},
            constraints=self.constraints,
        )

        knot = None
        raw_output = None
        for attempt in range(self.max_generation_attempts):
            try:
                raw_output = self.backend.generate(request)
                knot = self._parse_knot_output(str(raw_output))
                if knot is not None:
                    break
            except Exception:
                continue

        if knot is None:
            fallback = self.constraints.fallback_setpoint_c
            knot = {zone_id: fallback for zone_id in self.zone_ids}

        return {
            "knot": knot,
            "raw_output": raw_output,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "mode": mode,
            "block_index": block_index,
            "knot_index": knot_index,
        }

    def _parse_block_output(self, raw_output: str) -> list[dict[str, float]] | None:
        """Parse LLM output into a list of 6 knot dicts (zone_id -> setpoint)."""
        try:
            json_text = _extract_json_payload(raw_output)
            data = json.loads(json_text)
        except Exception:
            return None

        plan = data.get("plan") if isinstance(data, dict) else None
        if not isinstance(plan, list) or len(plan) != KNOTS_PER_BLOCK:
            return None

        knots: list[dict[str, float]] = []
        for entry in plan:
            if not isinstance(entry, dict):
                return None
            setpoints_raw = entry.get("setpoints")
            if not isinstance(setpoints_raw, list) or len(setpoints_raw) != len(self.zone_ids):
                return None
            knot: dict[str, float] = {}
            for zone_id, val in zip(self.zone_ids, setpoints_raw):
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    v = self.constraints.fallback_setpoint_c
                v = max(self.constraints.min_setpoint_c, min(self.constraints.max_setpoint_c, v))
                if self.constraints.quantization_c and self.constraints.quantization_c > 0:
                    v = _quantize(v, self.constraints.quantization_c)
                knot[zone_id] = v
            knots.append(knot)
        return knots

    def plan_block(
        self,
        *,
        block_index: int,
        block_start: Any,
        block_end: Any,
        mode: str = "balanced",
        replay_step_count: int = 0,
        observation: dict[str, dict[str, Any]] | None = None,
        wallclock: Any = None,
    ) -> dict[str, Any]:
        """Generate a 3h block plan for the given mode.

        Returns dict with keys: knots, raw_output, system_prompt, user_prompt, mode.
        knots is a list of 6 dicts mapping zone_id -> setpoint_c.
        """
        obs = observation if observation is not None else self._last_observation
        wc = wallclock if wallclock is not None else self._last_wallclock
        if obs is None:
            raise ValueError("No observation available for block planning. Call set_current_state first.")

        system_prompt = self._build_block_system_prompt(mode)
        user_prompt = self._build_block_user_prompt(
            block_index=block_index,
            block_start=block_start,
            block_end=block_end,
            mode=mode,
            observation=obs,
            wallclock=wc,
        )
        request = PlannerRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            payload={},
            constraints=self.constraints,
        )

        knots = None
        raw_output = None
        for attempt in range(self.max_generation_attempts):
            try:
                raw_output = self.backend.generate(request)
                knots = self._parse_block_output(str(raw_output))
                if knots is not None:
                    break
            except Exception:
                continue

        if knots is None:
            fallback = self.constraints.fallback_setpoint_c
            knots = [
                {zone_id: fallback for zone_id in self.zone_ids}
                for _ in range(KNOTS_PER_BLOCK)
            ]

        return {
            "knots": knots,
            "raw_output": raw_output,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "mode": mode,
            "block_index": block_index,
            "block_start": str(block_start),
            "block_end": str(block_end),
        }
