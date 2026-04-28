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


def _estimate_pmv_tool(temp: float, humidity: float, radiant: float) -> float:
    """PMV calculator exposed to the LLM via the <tool_call> protocol.

    Fixed parameters (match reward function defaults):
      met=1.0, clo=0.5 (summer), airspeed=0.1 m/s.
    User-varying: temp (°C), humidity (%), radiant (°C; zone MRT, typically
    higher than drybulb in summer when walls/roof haven't cooled down).

    Returns a finite float. Raises on invalid inputs.
    """
    if not (math.isfinite(temp) and math.isfinite(humidity) and math.isfinite(radiant)):
        raise ValueError("temp/humidity/radiant must be finite")
    if not (10.0 <= temp <= 40.0):
        raise ValueError(f"temp {temp} out of reasonable range 10-40°C")
    if not (0.0 <= humidity <= 100.0):
        raise ValueError(f"humidity {humidity} out of range 0-100%")
    if not (10.0 <= radiant <= 40.0):
        raise ValueError(f"radiant {radiant} out of reasonable range 10-40°C")
    pmv = estimate_zone_pmv(
        temperature_drybulb=temp,
        temperature_radiant=radiant,
        humidity=humidity,
        met=1.0,
        clo=0.5,
        airspeed=0.1,
    )
    if pmv is None:
        raise ValueError("PMV calculation returned non-finite")
    return float(pmv)


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
    # Optional occupancy-aware fallback overrides (back-compat: when both Nones,
    # the static `fallback_setpoint_c` above is used everywhere — identical to
    # the pre-2026-04-28 behavior).
    #
    # Motivation: when the planner falls back (parser failure on perfectionist
    # tool-call loops), a static 24°C wastes HVAC during occ=0 and a static
    # 30°C overheats during occ=1. Smart fallback picks the right side based
    # on current observed occupancy, giving cleaner GRPO signal in both
    # regimes without the noisy "30°C uniform sometimes wins, sometimes
    # explodes PMV" pattern.
    fallback_setpoint_low_occ_c: float | None = None   # used when avg_occ <= low_threshold
    fallback_setpoint_high_occ_c: float | None = None  # used when avg_occ >= high_threshold
    fallback_occ_low_threshold: float = 0.15
    fallback_occ_high_threshold: float = 0.5


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
    hvac_electricity_kwh: float
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

    # Concise guard: thinking that fits in ~600 tokens so JSON always emits.
    # Reward: -w_E*HVAC - sum(50*max(|PMV|-0.5,0)*occ), w_E=20.
    # Features available: per-zone (PMV, temp, occupancy); forecast_temperature_6h,
    # forecast_cloudcover_6h; outdoor_temp; current wallclock. Full list in prompt.
    _THINKING_GUARD = (
        "BE CONCISE. Your entire <think> must be UNDER 500 tokens so there is room "
        "for the JSON after </think>. Do NOT repeat observation numbers you do not "
        "use. Do NOT describe the task back to yourself. Jump straight to:\n"
        "  1. (1-2 lines) Key signals in THIS observation that drive the decision\n"
        "     (which zones have PMV>0.5? forecast heat-wave coming? low PV?)\n"
        "  2. (2-3 lines) State a short control rule combining those signals\n"
        "     (free form: if-else or delta formula, invent coefficients).\n"
        "  3. (1-2 lines) Apply the rule → per-zone setpoints.\n"
        "Then close </think> and output JSON {\"setpoints\": [...]}.\n"
        "Forecast caveat: forecast_*_6h arrays REFRESH ONLY EVERY 3 HOURS. Between "
        "refreshes they are stale (same numbers reused). Treat the forecast as a "
        "TREND direction, not as precise hourly truth; don't over-fit to exact values.\n"
        "Do NOT cite textbook formulas. Do NOT derive universal thermodynamics. "
        "Use concrete numbers from this observation only."
    )

    def _maybe_disable_qwen_thinking(self, text: str) -> str:
        import os
        enable_thinking = bool(int(os.environ.get("ASIM_ENABLE_THINKING", "0")))
        # ASIM_THINKING_GUARD=0 disables the guard even when thinking is on —
        # useful to test whether the guard itself is suppressing </think>.
        inject_guard = bool(int(os.environ.get("ASIM_THINKING_GUARD", "1")))
        model_name = (self.model_name or "").lower()
        if "qwen3" not in model_name:
            return text
        if enable_thinking:
            if "/no_think" in text:
                text = text.replace("/no_think\n", "").replace("/no_think", "")
            if inject_guard and self._THINKING_GUARD not in text:
                text = self._THINKING_GUARD + "\n\n" + text
            return text
        if "/no_think" not in text:
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

        import os
        enable_thinking = bool(int(os.environ.get("ASIM_ENABLE_THINKING", "0")))
        enable_pmv_tool = bool(int(os.environ.get("ASIM_ENABLE_PMV_TOOL", "0")))
        max_tool_calls = int(os.environ.get("ASIM_MAX_TOOL_CALLS", "30"))

        user_prompt = self._maybe_disable_qwen_thinking(request.user_prompt)
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            _chat_kwargs: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
            if "qwen3" in (self.model_name or "").lower():
                _chat_kwargs["enable_thinking"] = enable_thinking
            # Register PMV tool via Qwen3's native tools schema so the model
            # formats tool calls in its trained format.
            if enable_pmv_tool:
                _chat_kwargs["tools"] = [{
                    "type": "function",
                    "function": {
                        "name": "estimate_pmv",
                        "description": (
                            "Compute the PMV (Predicted Mean Vote) thermal comfort score "
                            "given zone dry-bulb temperature, relative humidity, and mean "
                            "radiant temperature. Fixed parameters: met=1.0, clo=0.5 (summer), "
                            "air_speed=0.1 m/s. Returns float PMV (typically -2 to +2; "
                            "|PMV|<0.5 is comfortable)."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "temp": {"type": "number", "description": "Zone dry-bulb temperature in °C (≈ setpoint if AC is active)"},
                                "humidity": {"type": "number", "description": "Relative humidity in percent (0-100)"},
                                "radiant": {"type": "number", "description": "Mean radiant temp in °C. USE the 'radiant=' value shown in the observation for that zone — do NOT assume radiant=temp. MRT is often 2-6°C warmer than drybulb in summer (walls/roof retain heat)."},
                            },
                            "required": ["temp", "humidity", "radiant"],
                        },
                    },
                }]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                **_chat_kwargs,
            )
        else:
            prompt_text = (
                f"System:\n{request.system_prompt}\n\n"
                f"User:\n{user_prompt}\n\n"
                "Assistant:\n"
            )

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        do_sample = bool(self.temperature > 0.0 or self.top_p < 1.0 or self.top_k > 0)
        base_generation_kwargs: dict[str, Any] = {
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            base_generation_kwargs["temperature"] = max(float(self.temperature), 1e-5)
            if self.top_p < 1.0:
                base_generation_kwargs["top_p"] = float(self.top_p)
            if self.top_k > 0:
                base_generation_kwargs["top_k"] = int(self.top_k)
        if self.repetition_penalty and abs(self.repetition_penalty - 1.0) > 1e-6:
            base_generation_kwargs["repetition_penalty"] = float(self.repetition_penalty)

        device = self._input_device()
        assistant_text = ""
        total_completion_tokens = 0
        tool_calls_used = 0
        consecutive_dup_calls = 0
        seen_call_args: set[tuple[float, float, float]] = set()
        # Narrative triggers that mean "the model verbally referenced the PMV
        # tool". If generation stops at one of these, we force a real tool call
        # by appending <tool_call> and letting the model complete the JSON.
        # Without this, Qwen3-8B writes "using the PMV calculator tool, I can
        # check..." but never emits the XML, defeating the tool's purpose.
        narrative_triggers = ["PMV calculator", "PMV tool", "pmv calculator", "pmv tool"]
        # Tool-calling loop: stop at </tool_call> OR narrative trigger, then
        # either parse the call or force one. Without PMV tool enabled, this
        # loop runs exactly once.
        while True:
            remaining_budget = int(self.max_output_tokens) - total_completion_tokens
            if remaining_budget <= 0:
                break
            combined = prompt_text + assistant_text
            inputs = self.tokenizer(combined, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            gen_kwargs = dict(base_generation_kwargs)
            gen_kwargs["max_new_tokens"] = remaining_budget
            # Use stop_strings so we can pause at </tool_call> or (only while
            # the model hasn't made any actual call yet) narrative triggers.
            # After the first real call, the model naturally references "the
            # PMV calculator" / "PMV tool" when summarizing its reasoning —
            # keeping the trigger on would create a positive feedback loop
            # (narrate → forced call → tool_response → narrate → loop).
            if enable_pmv_tool and tool_calls_used < max_tool_calls:
                if tool_calls_used == 0:
                    stop_strs = ["</tool_call>"] + narrative_triggers
                else:
                    stop_strs = ["</tool_call>"]
            else:
                stop_strs = []
            if stop_strs:
                gen_kwargs["stop_strings"] = stop_strs
                gen_kwargs["tokenizer"] = self.tokenizer
            with torch.no_grad():
                generated = self.model.generate(**inputs, **gen_kwargs)
            prompt_tokens = int(inputs["input_ids"].shape[1])
            new_tokens = generated[0][prompt_tokens:]
            new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            total_completion_tokens += int(new_tokens.numel())
            assistant_text += new_text

            # Case 1: model emitted </tool_call> — parse, inject response, continue.
            # Tool calls can appear during <think> OR after </think> (Qwen3
            # emits them as a separate assistant turn); we handle both.
            if (
                enable_pmv_tool
                and tool_calls_used < max_tool_calls
                and new_text.rstrip().endswith("</tool_call>")
            ):
                tool_calls_used += 1
                # Parse the just-emitted call's args to detect duplicates.
                current_args = self._extract_last_tool_call_args(assistant_text)
                is_dup = current_args is not None and current_args in seen_call_args
                if current_args is not None:
                    seen_call_args.add(current_args)
                if is_dup:
                    consecutive_dup_calls += 1
                else:
                    consecutive_dup_calls = 0
                tool_response = self._handle_pmv_tool_call(assistant_text, is_dup=is_dup)
                assistant_text += tool_response
                # Force finalize on cap-hit OR 2+ consecutive identical-arg
                # calls (stuck loop). Both keep the JSON output alive when the
                # model would otherwise spin.
                should_force_finalize = (
                    tool_calls_used >= max_tool_calls
                    or consecutive_dup_calls >= 2
                )
                if should_force_finalize:
                    reason = (
                        f"You have used all {max_tool_calls} PMV tool calls"
                        if tool_calls_used >= max_tool_calls
                        else f"You have made {consecutive_dup_calls + 1} consecutive "
                             "tool calls with IDENTICAL arguments"
                    )
                    assistant_text += (
                        "<|im_end|>\n"
                        "<|im_start|>user\n"
                        f"{reason}. Do NOT emit more <tool_call> — they will not "
                        "be processed. Close your reasoning with </think> and "
                        'output the final {"setpoints": [...]} JSON now.'
                        "<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                    # Also stop the loop's use of the narrative trigger by
                    # setting tool_calls_used to cap (this makes stop_strs []).
                    tool_calls_used = max_tool_calls
                continue

            # Case 2: model hit a narrative trigger (e.g., "PMV calculator")
            # without an actual <tool_call>. Force it to make a real call by
            # appending <tool_call> so the next generation must complete the JSON.
            if (
                enable_pmv_tool
                and tool_calls_used < max_tool_calls
                and any(new_text.rstrip().endswith(trig) for trig in narrative_triggers)
            ):
                # Append a short bridge + open tool_call so the model can only
                # continue by completing the JSON args and </tool_call>.
                assistant_text += ". <tool_call>"
                continue
            break

        text = assistant_text
        import os as _os
        if bool(int(_os.environ.get("ASIM_DEBUG_THINKING", "0"))):
            has_close = "</think>" in text
            marker = "HAS_CLOSE" if has_close else "NO_CLOSE"
            tool_marker = f" tool_calls={tool_calls_used}" if enable_pmv_tool else ""
            print(
                f"\n===== GENERATED ({len(text)} chars, {total_completion_tokens} tokens, {marker}{tool_marker}) =====\n"
                f"{text}\n"
                f"===== END =====\n",
                flush=True,
            )
            jsonl_path = _os.environ.get("ASIM_THINKING_JSONL", "")
            if jsonl_path:
                try:
                    import json as _json_trace, time as _time_trace
                    with open(jsonl_path, "a", encoding="utf-8") as _f:
                        _f.write(_json_trace.dumps({
                            "ts": _time_trace.time(),
                            "chars": len(text),
                            "tokens": total_completion_tokens,
                            "has_close": has_close,
                            "tool_calls": tool_calls_used,
                            "text": text,
                        }) + "\n")
                except Exception:
                    pass
        return str(text).strip()

    def _extract_last_tool_call_args(
        self, assistant_text: str
    ) -> tuple[float, float, float] | None:
        """Parse the most recent <tool_call> block's (temp, humidity, radiant)
        args. Returns None if unparseable. Used to detect duplicate calls."""
        import re as _re, json as _json_tool
        matches = list(_re.finditer(r"<tool_call>(.*?)</tool_call>",
                                     assistant_text, flags=_re.DOTALL))
        if not matches:
            return None
        try:
            call = _json_tool.loads(matches[-1].group(1).strip())
            args = call.get("arguments", {}) or {}
            return (
                round(float(args.get("temp", 0.0)), 2),
                round(float(args.get("humidity", args.get("rh", 0.0))), 1),
                round(float(args.get("radiant", args.get("tr", 0.0))), 2),
            )
        except Exception:
            return None

    def _handle_pmv_tool_call(
        self, assistant_text: str, *, is_dup: bool = False
    ) -> str:
        """Parse the most recent <tool_call>...</tool_call> block at the end
        of the assistant text, compute the requested PMV, and return a
        properly-formatted turn-boundary string to append.

        When is_dup=True, augments the tool_response body with a 'duplicate'
        warning field so the model sees a clear 'stop repeating' signal
        instead of just the same PMV value again.

        Qwen3's chat template expects tool responses wrapped as a user turn:
          <|im_end|>\\n<|im_start|>user\\n<tool_response>{...}</tool_response><|im_end|>\\n<|im_start|>assistant\\n
        Just embedding <tool_response> inline (no turn boundary) confuses the
        model — it keeps emitting tool calls as if still in its own turn.
        """
        import re as _re, json as _json_tool

        def _wrap(body: str) -> str:
            # Wrap the response with proper Qwen3 turn boundaries so the model
            # treats it as new input from user and starts a fresh assistant
            # reasoning pass.
            return (
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"<tool_response>\n{body}\n</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        matches = list(_re.finditer(r"<tool_call>(.*?)</tool_call>", assistant_text, flags=_re.DOTALL))
        if not matches:
            return _wrap('{"error": "no tool_call parsed"}')
        last = matches[-1]
        call_body = last.group(1).strip()
        try:
            call = _json_tool.loads(call_body)
            name = call.get("name", "")
            args = call.get("arguments", {}) or {}
        except Exception as _exc:
            return _wrap(f'{{"error": "invalid json: {str(_exc)[:80]}"}}')
        if name != "estimate_pmv":
            return _wrap(f'{{"error": "unknown tool {name}"}}')
        try:
            temp = float(args.get("temp", args.get("temperature", 24.0)))
            humidity = float(args.get("humidity", args.get("rh", 60.0)))
            radiant = args.get("radiant", args.get("radiant_temp", args.get("tr")))
            radiant = float(radiant) if radiant is not None else temp
            pmv_val = _estimate_pmv_tool(temp=temp, humidity=humidity, radiant=radiant)
            warnings = []
            if is_dup:
                warnings.append(
                    "DUPLICATE call — you already have this result. "
                    "Either vary (temp, humidity, radiant) or close </think> and "
                    "output the final setpoints JSON."
                )
            # PMV near-limit warning: at |PMV|>=0.4 the next env step's PMV
            # (transient ramp) may breach ±0.5 and trigger comfort penalty.
            # Leave at least ~0.1 buffer when picking a final setpoint.
            if abs(pmv_val) >= 0.4:
                if pmv_val >= 0.4:
                    warnings.append(
                        f"PMV={pmv_val:.3f} is within 0.1 of the +0.5 upper limit. "
                        "Next step's PMV may overshoot due to transient. Pick a "
                        "LOWER setpoint with PMV ≤ +0.4 for safety."
                    )
                else:
                    warnings.append(
                        f"PMV={pmv_val:.3f} is within 0.1 of the -0.5 lower limit. "
                        "Next step's PMV may undershoot. Pick a HIGHER setpoint "
                        "with PMV ≥ -0.4 for safety."
                    )
            if warnings:
                # Join with ' | ' and embed as one warning field. Model needs to
                # see all signals; keeping them in one body line keeps the
                # response compact.
                joined = " | ".join(warnings).replace('"', "'")
                return _wrap(f'{{"pmv": {pmv_val:.3f}, "warning": "{joined}"}}')
            return _wrap(f'{{"pmv": {pmv_val:.3f}}}')
        except Exception as _exc:
            return _wrap(f'{{"error": "calc failed: {str(_exc)[:80]}"}}')


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
        building_electricity_kwh = joules_to_kwh(first_zone.get("energy_building", 0.0))
        hvac_electricity_kwh = facility_electricity_kwh - building_electricity_kwh
        pv_kwh = joules_to_kwh(first_zone.get("PV", 0.0))
        net_grid_kwh = hvac_electricity_kwh - pv_kwh

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
            hvac_electricity_kwh=hvac_electricity_kwh,
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
        observation: dict[str, dict[str, Any]] | None = None,
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
    "cooling": (
        "Active cooling: target PMV in [-0.5, 0]. "
        "Set occupied zones 1-2°C BELOW current temperature to ensure cooling. "
        "South-facing zones need lower setpoints than north-facing. "
        "Unoccupied zones can be 2-3°C higher than occupied zones."
    ),
    "balanced": (
        "Balance comfort and energy: target PMV in [-0.1, +0.2]. "
        "Set occupied zones near current temperature (±0.5°C). "
        "When cloud cover is low (<30%), solar generation is high — lean toward cooling. "
        "When cloud cover is high (>70%) or increasing, solar is weak — lean toward energy saving. "
        "Unoccupied zones should get noticeably higher setpoints than occupied zones."
    ),
    "energy_saving": (
        "Minimise energy: target PMV in [+0.2, +0.5]. "
        "Set occupied zones 1-2°C ABOVE balanced setpoint to reduce cooling. "
        "Unoccupied zones should be at upper bound (28-30°C). "
        "Only lower setpoint if PMV approaches +0.5."
    ),
}

# Flat 3-mode structure (no hierarchical tiers needed)
STRATEGY_TIERS: dict[str, list[str]] = {
    "cooling": ["cooling"],
    "balanced": ["balanced"],
    "energy_saving": ["energy_saving"],
}
ALL_CANDIDATE_MODES: list[str] = ["cooling", "balanced", "energy_saving"]

# Brief PMV explanation embedded in system prompts.
PMV_EXPLANATION = (
    "PMV (Predicted Mean Vote) is a thermal comfort score centered on 0:\n"
    "  0 = neutral (ideal),  +0.5 = slightly warm,  -0.5 = slightly cool,\n"
    "  |PMV| > 0.5 → occupants feel uncomfortable.\n"
    "\n"
    "Decision cadence: you choose ONE setpoint per zone for the next 10 minutes.\n"
    "Reward is computed every 10 minutes (each control step). PMV must stay\n"
    "within [-0.5, +0.5] at every 10-min mark — you'll see the new observation\n"
    "before your next decision and can adjust.\n"
    "\n"
    "How setpoint affects PMV:\n"
    "  LOWER cooling setpoint  → more cooling  → LOWER (cooler) PMV\n"
    "  HIGHER cooling setpoint → less cooling  → HIGHER (warmer) PMV\n"
    "\n"
    "How PMV affects reward (important):\n"
    "  step_reward = -w_energy * HVAC_kWh - sum_zone(50 * max(|PMV|-0.5, 0) * occupancy)\n"
    "  w_energy is large (20). PMV only costs reward WHEN occupancy > 0.\n"
    "  If occupancy = 0 (unoccupied), PMV can be any value with NO penalty -> free to save HVAC energy.\n"
    "  If occupancy > 0, you MUST keep |PMV| <= 0.5 to avoid a heavy comfort penalty (50 per unit of excess).\n"
    "\n"
    "  SAFETY BUFFER: When the PMV calculator returns |PMV| >= 0.4, you're\n"
    "  already too close to the ±0.5 limit. The HVAC takes 1-2 minutes to\n"
    "  reach setpoint, so the next 10-min step's actual PMV may overshoot.\n"
    "  RULE: Pick a setpoint whose tool-reported |PMV| <= 0.4. If a candidate\n"
    "  gives 0.4 < |PMV| < 0.5, retest with a more conservative setpoint\n"
    "  (lower setpoint when PMV>0, higher setpoint when PMV<0).\n"
    "\n"
    "Practical rules of thumb:\n"
    "  - Unoccupied zones (occ=0): raise setpoints to save energy; PMV doesn't matter.\n"
    "  - Occupied zones with PMV in [-0.5, +0.5]: already comfortable; only tweak for energy.\n"
    "  - Occupied zones with PMV > +0.5: LOWER setpoint to pull PMV down into [-0.5, +0.5].\n"
    "  - Occupied zones with PMV < -0.5: RAISE setpoint to pull PMV up into [-0.5, +0.5]."
)

# PMV tool description (enabled when ASIM_ENABLE_PMV_TOOL=1).
PMV_TOOL_INSTRUCTIONS = (
    "\n"
    "═══════════════════ PMV CALCULATOR TOOL ═══════════════════\n"
    "You have access to a PMV calculator. To USE IT, you must emit the EXACT XML below\n"
    "inside your <think> block (do NOT just talk about using it — actually emit the tags):\n"
    "\n"
    '  <tool_call>{"name": "estimate_pmv", "arguments": {"temp": 24.5, "humidity": 60, "radiant": 24.5}}</tool_call>\n'
    "\n"
    "The system will pause, compute PMV, and inject back:\n"
    "\n"
    '  <tool_response>{"pmv": 0.321}</tool_response>\n'
    "\n"
    "You then continue thinking with the result.\n"
    "\n"
    "━━━ FULL WORKED EXAMPLE ━━━\n"
    "Observation shows Zone 1FNW: temp=22.0C, radiant=28.0C, humidity=61%, occupancy=0.5, PMV=-0.43.\n"
    "Note: radiant (28.0C) is ~6C higher than drybulb — walls/roof still warm from outside.\n"
    "<think>\n"
    "PMV=-0.43 is slightly cool but within [-0.5, +0.5]. I want to RAISE the setpoint to save\n"
    "energy while keeping PMV in range. Let me test setpoint 23.5°C. Air will approach 23.5,\n"
    "but radiant stays ~28.0 (surfaces change slowly), so I pass radiant=28.0 from the obs.\n"
    '<tool_call>{"name": "estimate_pmv", "arguments": {"temp": 23.5, "humidity": 61, "radiant": 28.0}}</tool_call>\n'
    '<tool_response>{"pmv": -0.090}</tool_response>\n'
    "23.5°C → PMV=-0.09, nearly neutral. Let me also try 24°C to save even more energy.\n"
    '<tool_call>{"name": "estimate_pmv", "arguments": {"temp": 24.0, "humidity": 61, "radiant": 28.0}}</tool_call>\n'
    '<tool_response>{"pmv": 0.030}</tool_response>\n'
    "24°C → PMV=+0.03, essentially neutral and still ≤ +0.5. 24°C saves more energy. Pick 24.0.\n"
    "</think>\n"
    "\n"
    '{"setpoints": [24.0, ...]}\n'
    "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "Arguments:\n"
    "  temp     = expected indoor dry-bulb after the step (≈ setpoint if cooling is active)\n"
    "  humidity = relative humidity % (use current obs value)\n"
    "  radiant  = mean radiant temp °C — USE the 'radiant=' value shown in the zone's obs line.\n"
    "             In summer, radiant is often 2-6°C warmer than drybulb because walls/roof\n"
    "             retain heat. If you pass radiant=temp you'll get a WRONG (too cool) estimate.\n"
    "Fixed: met=1.0, clo=0.5 (summer), air_speed=0.1 m/s. These match the reward's PMV.\n"
    "\n"
    "IMPORTANT RULES:\n"
    "  1. Writing 'let me call the tool' or 'assuming the PMV is X' WITHOUT emitting the\n"
    "     literal <tool_call>...</tool_call> XML produces NO result. You will get NOTHING\n"
    "     back. You must emit the XML characters exactly as shown above.\n"
    "  2. Do NOT guess the PMV — let the tool give you the real value.\n"
    "  3. Each <tool_call> will PAUSE your thinking; the response comes back as <tool_response>.\n"
    "  4. Call the tool as many times as you need (test different candidate setpoints),\n"
    "     but do NOT call with the IDENTICAL (temp, humidity, radiant) args twice — you\n"
    "     already have that answer from the first call, and duplicates are penalized.\n"
    "  5. After gathering enough PMV data, finalize with </think> then the JSON answer.\n"
    "  6. ALWAYS pass the zone's observed radiant value (not drybulb) to the tool.\n"
    "\n"
    "REQUIRED WORKFLOW FOR EVERY KNOT:\n"
    "  Step A (mandatory for occupied zones): CALL estimate_pmv for the worst-PMV zone\n"
    "         with your CANDIDATE setpoint. Wait for <tool_response>. If PMV is still\n"
    "         out of [-0.5, +0.5], adjust and call again.\n"
    "  Step B: When all tested candidates give PMV ∈ [-0.5, +0.5], pick the one that\n"
    "         SAVES the most energy (highest cooling setpoint that still keeps PMV\n"
    "         in range for occupied zones).\n"
    "  Step C: Emit </think> then the final JSON answer.\n"
    "\n"
    "Narrating 'let me call the tool' or 'suppose the PMV is 0.2' WITHOUT emitting the\n"
    "actual <tool_call> XML is a FAILURE — no real computation happens, and your\n"
    "guess is likely wrong. ALWAYS emit the XML to get a real answer.\n"
    "═══════════════════════════════════════════════════════════"
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

KNOTS_PER_BLOCK = 6   # default for 1h block / 10min knot (matches KNOT_ENV_STEPS=1)
BLOCK_MINUTES = 60
KNOT_MINUTES = 10


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
        import threading
        self._inference_lock = threading.Lock()
        self.backend = backend
        self._current_date: str | None = None
        self.constraints = constraints or PlannerConstraints()
        self.zone_ids = tuple(zone_ids)
        self.max_generation_attempts = max(int(max_generation_attempts), 1)
        self._last_observation: dict[str, dict[str, Any]] | None = None
        self._last_wallclock: Any = None
        self._prev_block_results: list[dict[str, Any]] = []  # previous block results for cross-block context
        self._prev_knot_results: list[dict[str, Any]] = []   # per-knot action+reward feedback within current day

    def set_current_state(
        self,
        observation: dict[str, dict[str, Any]],
        wallclock: Any = None,
    ) -> None:
        """Cache the current observation so plan_block can use it."""
        self._last_observation = observation
        self._last_wallclock = wallclock

    def record_block_result(
        self,
        block_index: int,
        block_start: str,
        block_end: str,
        winner_mode: str,
        winner_reward: float,
        hvac_kwh: float = 0.0,
        pmv_violation: float = 0.0,
    ) -> None:
        """Record a completed block's result for cross-block context injection."""
        self._prev_block_results.append({
            "block_index": block_index,
            "block_time": f"{block_start}-{block_end}",
            "mode": winner_mode,
            "reward": round(winner_reward, 3),
            "hvac_kwh": round(hvac_kwh, 1),
            "pmv": round(pmv_violation, 3),
        })

    def clear_block_results(self) -> None:
        """Clear previous block results at start of new day."""
        self._prev_block_results = []

    def record_knot_result(
        self,
        *,
        block_index: int,
        knot_index: int,
        wallclock: str,
        setpoints: dict[str, float],
        hvac_kwh: float,
        pmv_violation_per_zone: dict[str, float] | None = None,
        occupancy_per_zone: dict[str, float] | None = None,
    ) -> None:
        """Record a completed knot's action + decomposed reward components.

        Exposed to the next knot's prompt so the model sees what it just did
        and how the physics responded (energy cost, per-zone PMV violations).
        """
        viol = pmv_violation_per_zone or {}
        occ = occupancy_per_zone or {}
        self._prev_knot_results.append({
            "block_index": int(block_index),
            "knot_index": int(knot_index),
            "wallclock": str(wallclock),
            "setpoints": {z: round(float(v), 1) for z, v in setpoints.items()},
            "hvac_kwh": round(float(hvac_kwh), 2),
            "pmv_violation_per_zone": {z: round(float(v), 3) for z, v in viol.items() if abs(float(v)) > 1e-4},
            "occupancy_per_zone": {z: round(float(v), 2) for z, v in occ.items()},
        })

    def clear_knot_results(self) -> None:
        """Clear per-knot history at start of a new day rollout."""
        self._prev_knot_results = []

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
            radiant = _as_float(zone_obs.get("temperature:radiant"))
            humidity = _as_float(zone_obs.get("humidity"))
            occupancy = _as_float(zone_obs.get("occupancy"))
            pmv = estimate_zone_pmv(
                temperature_drybulb=drybulb,
                temperature_radiant=radiant,
                humidity=humidity,
            )
            zone_desc = ZONE_DESCRIPTIONS.get(zone_id, "")
            zone_lines.append(
                f"- {zone_id} ({zone_desc}): temp={drybulb:.1f}C, radiant={radiant:.1f}C, "
                f"humidity={humidity:.0f}%, occupancy={occupancy:.0f}, PMV={pmv:.2f}"
            )

        first_zone = observation.get(self.zone_ids[0], {})
        pv_kwh = joules_to_kwh(first_zone.get("PV", 0.0))
        facility_kwh = joules_to_kwh(first_zone.get("energy_consumption", 0.0))
        net_grid_kwh = facility_kwh - pv_kwh

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
        reflection_ctx = self.get_reflection_context(self._current_date) if hasattr(self, "_reflection_memory") and self._reflection_memory else ""
        block_ref_ctx = self.get_block_reflection_context() if hasattr(self, "_block_reflections") and self._block_reflections else ""
        prompt = (
            "You are an HVAC planning assistant for a 2-story 8-zone office building.\n"
            f"Output the next {KNOT_MINUTES}-minute cooling setpoint for each of the {len(self.zone_ids)} zones.\n\n"
            f"{PMV_EXPLANATION}\n\n"
            "Zone layout:\n"
            f"{zone_layout}\n\n"
            "Set each zone's setpoint based on its current temperature, PMV, occupancy, "
            "and solar exposure.\n\n"
        )
        if reflection_ctx:
            prompt += reflection_ctx + "\n"
        if block_ref_ctx:
            prompt += block_ref_ctx + "\n"
        prompt += (
            f"Planning mode: {mode_desc}\n"
            "Return JSON only. No markdown, no explanations.\n"
            f"Hard bounds: {self.constraints.min_setpoint_c} to {self.constraints.max_setpoint_c} C.\n"
            "Round to nearest 0.1 C."
        )
        return prompt

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
            radiant = _as_float(zone_obs.get("temperature:radiant"))
            humidity = _as_float(zone_obs.get("humidity"))
            occupancy = _as_float(zone_obs.get("occupancy"))
            pmv = estimate_zone_pmv(
                temperature_drybulb=drybulb,
                temperature_radiant=radiant,
                humidity=humidity,
            )
            zone_desc = ZONE_DESCRIPTIONS.get(zone_id, "")
            zone_lines.append(
                f"- {zone_id} ({zone_desc}): temp={drybulb:.1f}C, radiant={radiant:.1f}C, "
                f"humidity={humidity:.0f}%, occupancy={occupancy:.0f}, PMV={pmv:.2f}"
            )

        first_zone = observation.get(self.zone_ids[0], {})
        pv_kwh = joules_to_kwh(first_zone.get("PV", 0.0))
        facility_kwh = joules_to_kwh(first_zone.get("energy_consumption", 0.0))
        net_grid_kwh = facility_kwh - pv_kwh

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

        # Build zone-level PMV hints with direction guidance
        mode_desc = CANDIDATE_MODE_DESCRIPTIONS.get(mode, "")
        pmv_targets = {"cooling": (-0.5, 0.0), "balanced": (-0.1, 0.2), "energy_saving": (0.2, 0.5)}
        pmv_lo, pmv_hi = pmv_targets.get(mode, (-0.5, 0.5))
        zone_hints = []
        for zone_id in self.zone_ids:
            zone_obs = observation.get(zone_id, {})
            drybulb = _as_float(zone_obs.get("temperature_drybulb"))
            pmv = estimate_zone_pmv(
                temperature_drybulb=drybulb,
                temperature_radiant=_as_float(zone_obs.get("temperature:radiant")),
                humidity=_as_float(zone_obs.get("humidity")),
            )
            if pmv > pmv_hi + 0.05:
                zone_hints.append(f"  {zone_id}: PMV={pmv:+.2f} TOO HIGH → LOWER setpoint")
            elif pmv < pmv_lo - 0.05:
                zone_hints.append(f"  {zone_id}: PMV={pmv:+.2f} TOO LOW → HIGHER setpoint")

        # Outdoor weather from observation
        outdoor_temp = _as_float(first_zone.get("outdoor_temp", 0))
        cloud_cover = _as_float(first_zone.get("cloud_cover", 0))

        prompt = (
            f"Block {block_index + 1}: {block_start} to {block_end}, "
            f"Knot {knot_index + 1}\n"
            f"Current time: {wallclock}\n"
            f"Outdoor: {outdoor_temp:.1f}°C, Cloud cover: {cloud_cover:.0f}/10\n"
            f"Current PV: {pv_kwh:.2f} kWh, Net grid: {net_grid_kwh:.2f} kWh\n"
            f"Zone order: {zone_keys}\n"
            "Current zone states:\n"
            f"{chr(10).join(zone_lines)}\n"
        )
        if zone_hints:
            prompt += f"Setpoint adjustment hints (target PMV {pmv_lo:+.1f} to {pmv_hi:+.1f}):\n"
            prompt += chr(10).join(zone_hints) + "\n"
        if forecast_lines:
            prompt += "Forecast:\n" + chr(10).join(forecast_lines) + "\n"

        # Forecast bias: compare real-time cloud cover with forecast
        if forecast_available and cloudcover_6h and cloud_cover > 0:
            forecast_cloud_now = float(cloudcover_6h[0])  # forecast for current hour (%)
            actual_cloud_pct = cloud_cover * 10  # EP 0-10 → %
            bias = actual_cloud_pct - forecast_cloud_now
            if abs(bias) > 15:
                direction = "cloudier" if bias > 0 else "clearer"
                pv_impact = "lower" if bias > 0 else "higher"
                prompt += (f"Forecast bias: actual cloud {actual_cloud_pct:.0f}% vs forecast {forecast_cloud_now:.0f}% "
                           f"({direction} than predicted → PV may be {pv_impact} than forecast suggests)\n")

        # Previous block results for cross-block coordination
        if self._prev_block_results:
            prompt += "Previous blocks today:\n"
            for pb in self._prev_block_results:
                prompt += (f"  Block {pb['block_index']+1} ({pb['block_time']}): "
                           f"mode={pb['mode']}, reward={pb['reward']:+.3f}, "
                           f"HVAC={pb['hvac_kwh']:.0f}kWh, PMV_viol={pb['pmv']:.3f}\n")

        prompt += (
            f"\nReturn a JSON object with a \"setpoints\" array of "
            f"{len(self.zone_ids)} numeric Celsius values in zone order.\n"
            "Each zone MUST get its own setpoint based on its current PMV and the hints above.\n"
            "Zones with TOO HIGH PMV need LOWER setpoints. Zones with TOO LOW PMV need HIGHER setpoints.\n"
            f"PMV hard limits: all occupied zones must stay within [-0.5, +0.5].\n"
            f"Example format: {{\"setpoints\": [<float>, <float>, ..., <float>]}}\n"
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
                with self._inference_lock:
                    raw_output = self.backend.generate(request)
                knot = self._parse_knot_output(str(raw_output))
                if knot is not None:
                    break
            except Exception:
                continue

        if knot is None:
            fallback = self.constraints.fallback_setpoint_c
            knot = {zone_id: fallback for zone_id in self.zone_ids}

        # Sanitize: clamp to hard bounds only (no PMV clamp — let LLM self-correct via hints)
        for zone_id in self.zone_ids:
            value = _as_float(knot.get(zone_id, self.constraints.fallback_setpoint_c))
            value = max(self.constraints.min_setpoint_c, min(self.constraints.max_setpoint_c, value))
            knot[zone_id] = round(float(value), 1)

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

    # ------------------------------------------------------------------
    # Reflexion: cross-day reflection and memory
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def _init_reflection_state(self):
        """Initialize reflection memory (call once before multi-day eval)."""
        if not hasattr(self, "_reflection_memory"):
            self._reflection_memory: list[str] = []
        if not hasattr(self, "_reflection_by_date"):
            self._reflection_by_date: dict[str, list[str]] = {}
        if not hasattr(self, "_block_reflections"):
            self._block_reflections: list[str] = []
        if not hasattr(self, "_compressed_rules"):
            self._compressed_rules = None

    def generate_block_reflection(
        self,
        *,
        date: str,
        block_index: int,
        block_start: str,
        block_end: str,
        all_mode_rewards: dict[str, float],
        winner_mode: str,
        candidate_breakdowns: list[dict[str, Any]] | None = None,
        winner_index: int | None = None,
        observation_trajectory: dict | None = None,
        zone_pmv_summary: str = "",
    ) -> str:
        """Generate a per-block reflection immediately after a block completes.

        Args:
            date: Current date string
            block_index: Block index (0-based)
            block_start, block_end: Block time range
            all_mode_rewards: Dict mapping mode name -> relative reward
            winner_mode: Mode that won this block
            candidate_breakdowns: Optional per-sample reward decomposition used by
                unified/free-mode training. Each entry may include mode, sample_type,
                relative_reward, reward_sum, energy_reward, pmv_reward, hvac_kwh,
                net_grid_kwh, and pmv_violation.
            observation_trajectory: Optional dict with observation changes during the block:
                start_temp, end_temp, start_pv, end_pv, start_cloudcover, end_cloudcover,
                outdoor_temp, etc.
        """
        self._init_reflection_state()

        # Format candidate reward breakdown. In free-mode multiple samples may
        # choose the same mode, so sample-level labels are more accurate than a
        # mode->reward dict.
        mode_lines = []
        if candidate_breakdowns:
            sorted_candidates = sorted(
                candidate_breakdowns,
                key=lambda x: float(x.get("relative_reward", 0.0)),
                reverse=True,
            )
            for cand in sorted_candidates:
                idx = int(cand.get("sample_index", 0))
                mode = str(cand.get("mode", "?"))
                stype = str(cand.get("sample_type", "?"))
                rel = float(cand.get("relative_reward", 0.0))
                reward_sum = float(cand.get("reward_sum", 0.0))
                energy_reward = float(cand.get("energy_reward", 0.0))
                pmv_reward = float(cand.get("pmv_reward", 0.0))
                hvac_kwh = float(cand.get("hvac_kwh", 0.0))
                net_grid_kwh = float(cand.get("net_grid_kwh", 0.0))
                pmv_violation = float(cand.get("pmv_violation", 0.0))
                tag = " <- WINNER" if bool(cand.get("is_winner", False)) else ""
                mode_lines.append(
                    f"    sample{idx} {mode}/{stype}: total_rel={rel:+.3f}, "
                    f"env_sum={reward_sum:+.3f}, energy_term={energy_reward:+.3f}, "
                    f"pmv_term={pmv_reward:+.3f}, HVAC={hvac_kwh:.1f}kWh, "
                    f"net_grid={net_grid_kwh:.1f}kWh, PMV_viol={pmv_violation:.3f}{tag}"
                )
            best_candidate = sorted_candidates[0]
            worst_candidate = sorted_candidates[-1]
            best_reward = float(best_candidate.get("relative_reward", 0.0))
            worst_reward = float(worst_candidate.get("relative_reward", 0.0))
            worst_mode = str(worst_candidate.get("mode", "?"))
            winner_label = str(best_candidate.get("mode", winner_mode))
        else:
            for m, r in sorted(all_mode_rewards.items(), key=lambda x: -x[1]):
                tag = " <- WINNER" if m == winner_mode else ""
                mode_lines.append(f"    {m}: {r:+.3f}{tag}")
            worst_mode = min(all_mode_rewards, key=all_mode_rewards.get)
            worst_reward = all_mode_rewards[worst_mode]
            best_reward = all_mode_rewards[winner_mode]
            winner_label = winner_mode
        mode_summary = "\n".join(mode_lines)

        # Format observation trajectory if available
        obs_summary = ""
        if observation_trajectory:
            ot = observation_trajectory
            changes = []
            if "start_pv" in ot and "end_pv" in ot:
                changes.append(f"PV: {ot['start_pv']:.1f} → {ot['end_pv']:.1f} kWh")
            if "start_cloudcover" in ot and "end_cloudcover" in ot:
                changes.append(f"Cloud: {ot['start_cloudcover']:.0f}% → {ot['end_cloudcover']:.0f}%")
            if "start_outdoor_temp" in ot and "end_outdoor_temp" in ot:
                changes.append(f"Outdoor: {ot['start_outdoor_temp']:.1f} → {ot['end_outdoor_temp']:.1f}°C")
            if "avg_zone_temp" in ot:
                changes.append(f"Avg zone temp: {ot['avg_zone_temp']:.1f}°C")
            if changes:
                obs_summary = "Conditions during block: " + ", ".join(changes)

        gap = best_reward - worst_reward

        system_prompt = (
            "You are an HVAC control analyst. After one 2-hour control block, provide a brief "
            "reflection (2-3 sentences) covering:\n"
            "1. Why the winner outperformed using the reward breakdown\n"
            "2. Whether it won by saving energy, reducing PMV violation, or both\n"
            "3. Why the worst candidate failed (energy vs PMV tradeoff)\n"
            "4. Which zones had PMV violations and what setpoint adjustments are needed\n"
            "5. A rule: 'When [condition], use [mode], and adjust [zone] setpoint'\n"
            "Be very specific about temperatures, PV, cloud cover, and zone names. No generic statements."
        )
        user_prompt = (
            f"Date: {date}, Block {block_index+1} ({block_start}-{block_end})\n"
            f"Candidate reward breakdown:\n{mode_summary}\n"
            f"Winner: {winner_label} ({best_reward:+.3f}), Worst: {worst_mode} ({worst_reward:+.3f}), Gap: {gap:.3f}\n"
        )
        if obs_summary:
            user_prompt += f"{obs_summary}\n"
        if zone_pmv_summary:
            user_prompt += f"Zone PMV status:\n{zone_pmv_summary}\n"
        user_prompt += "Reflection:"

        request = PlannerRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            payload={},
            constraints=self.constraints,
        )
        try:
            with self._inference_lock:
                reflection = self.backend.generate(request)
            import re
            reflection = re.sub(r'<think>.*?</think>\s*', '', reflection, flags=re.DOTALL).strip()
        except Exception:
            reflection = f"Block {block_index+1} ({block_start}-{block_end}): {winner_mode} won ({best_reward:+.3f})"

        block_ref = f"[{date} B{block_index+1} {block_start}-{block_end}] {reflection}"
        self._block_reflections.append(block_ref)
        return reflection

    def get_block_reflection_context(self, block_index: int | None = None) -> str:
        """Return recent per-block reflections for similar time slots."""
        self._init_reflection_state()
        if not self._block_reflections:
            return ""
        # Get the most recent block reflections (up to 10)
        recent = self._block_reflections[-10:]
        # If block_index given, prioritize same-block reflections
        if block_index is not None:
            tag = f" B{block_index+1} "
            same_block = [r for r in self._block_reflections if tag in r]
            other = [r for r in recent if tag not in r]
            combined = same_block[-3:] + other[-4:]
        else:
            combined = recent[-7:]
        if not combined:
            return ""
        return (
            "Per-block experience:\n"
            + "\n".join(f"- {r[:200]}" for r in combined)
            + "\n"
        )

    def generate_day_reflection(
        self,
        *,
        date: str,
        block_results: list[dict],
        total_reward: float,
        baseline_reward: float,
        weather_summary: str = "",
    ) -> str:
        """Generate a natural-language reflection after a day's control.

        Args:
            date: The date string (e.g. "2025-08-25")
            block_results: List of per-block dicts with keys:
                block_index, block_start, block_end, winner_mode, winner_reward,
                baseline_reward, relative_reward
            total_reward: Day's total reward (candidate)
            baseline_reward: Day's total baseline reward
            weather_summary: Optional weather context string for the day

        Returns:
            Reflection text string.
        """
        self._init_reflection_state()

        block_summary_lines = []
        for br in block_results:
            line = (
                f"  Block {br['block_index']+1} ({br.get('block_start','?')}-{br.get('block_end','?')}): "
                f"winner={br.get('winner_mode','?')} (reward={br.get('relative_reward', 0):+.3f})"
            )
            all_rewards = br.get("all_mode_rewards", {})
            if all_rewards:
                mode_strs = [f"{m}={r:+.3f}" for m, r in all_rewards.items()]
                line += f"  | all modes: {', '.join(mode_strs)}"
            block_summary_lines.append(line)
        block_summary = "\n".join(block_summary_lines)

        relative_total = total_reward - baseline_reward
        system_prompt = (
            "You are an HVAC control analyst for a Houston office building. "
            "Given today's block-by-block control results with all candidate mode rewards, "
            "write a brief reflection (3-5 sentences) summarizing:\n"
            "1. Which blocks performed well or poorly, and which modes won or lost\n"
            "2. Why certain modes outperformed others (relate to time of day, cooling demand, PV availability)\n"
            "3. One specific actionable insight for controlling this building on a similar day\n"
            "Be concise and specific. Reference times and mode names."
        )
        user_prompt = (
            f"Date: {date}\n"
            f"Total relative reward vs baseline: {relative_total:+.3f}\n"
            f"Baseline total reward: {baseline_reward:.3f}\n\n"
        )
        if weather_summary:
            user_prompt += f"Weather context:\n{weather_summary}\n\n"
        user_prompt += f"Per-block results:\n{block_summary}\n"
        if self._reflection_memory:
            user_prompt += (
                f"\nPrevious reflections:\n"
                + "\n".join(f"  - {r}" for r in self._reflection_memory[-3:])
                + "\n"
            )
        user_prompt += "\nWrite your reflection:"

        request = PlannerRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            payload={},
            constraints=self.constraints,
        )
        try:
            reflection = self.backend.generate(request)
            # Strip Qwen3 thinking tags if present
            import re
            reflection = re.sub(r'<think>.*?</think>\s*', '', reflection, flags=re.DOTALL).strip()
        except Exception:
            reflection = f"Day {date}: relative_reward={relative_total:+.3f}"

        self._reflection_memory.append(f"[{date}] {reflection}")
        self._reflection_by_date.setdefault(date, []).append(reflection)
        return reflection

    def get_reflection_context(self, current_date: str | None = None) -> str:
        """Return reflections relevant to the current date.

        If current_date is given and we have past reflections for that same date
        (from previous episodes), return those. Otherwise fall back to the most
        recent 5 reflections.
        """
        self._init_reflection_state()
        lines = []
        # Same-date reflections from previous episodes (most valuable)
        if current_date and current_date in self._reflection_by_date:
            for r in self._reflection_by_date[current_date][-3:]:
                lines.append(f"- [same-date prev episode] {r}")
        # Recent reflections for general context
        for r in self._reflection_memory[-5:]:
            lines.append(f"- {r}")
        if not lines:
            return ""
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique.append(line)
        return (
            "Previous reflections (use these to improve today's decisions):\n"
            + "\n".join(unique[:8])
            + "\n"
        )

    def clear_reflections(self):
        """Reset reflection memory."""
        self._reflection_memory = []
        self._reflection_by_date = {}
        self._block_reflections = []
        self._compressed_rules = None

    def compress_reflections(self) -> str:
        """Compress all accumulated reflections into a concise set of rules.

        Call after training completes. The compressed rules replace verbose
        reflections for eval/deployment, giving select_mode cleaner context.

        Returns the compressed rules string.
        """
        self._init_reflection_state()
        if not self._reflection_memory and not self._block_reflections:
            return ""

        # Gather all reflections
        all_refs = []
        for r in self._reflection_memory[-30:]:  # last 30 day reflections
            all_refs.append(f"[day] {r[:300]}")
        for r in self._block_reflections[-30:]:  # last 30 block reflections
            all_refs.append(f"[block] {r[:200]}")
        ref_text = "\n".join(all_refs)

        system_prompt = (
            "You are an HVAC control strategist. Given a collection of daily and per-block "
            "reflections from training, compress them into 5-8 concise rules.\n\n"
            "Each rule should follow this format:\n"
            "- WHEN [specific condition] → USE [mode] BECAUSE [reason]\n\n"
            "Rules should cover:\n"
            "1. Which mode works best for each time-of-day period (morning/midday/afternoon/evening)\n"
            "2. How weather conditions (PV, cloud cover, rain) affect mode choice\n"
            "3. When NOT to use a certain mode\n\n"
            "Be specific with times, thresholds, and mode names. "
            "Avoid generic statements like 'comfort is usually best'."
        )
        user_prompt = (
            f"Total reflections: {len(self._reflection_memory)} days, {len(self._block_reflections)} blocks\n\n"
            f"Reflections:\n{ref_text}\n\n"
            "Compress into 5-8 actionable rules:"
        )

        request = PlannerRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            payload={},
            constraints=self.constraints,
        )
        try:
            with self._inference_lock:
                raw = self.backend.generate(request)
            import re
            rules = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
        except Exception:
            rules = "Default: use comfort for afternoon blocks, balanced for morning, energy_saving for early morning."

        self._compressed_rules = rules
        return rules

    def get_compressed_rules(self) -> str:
        """Return compressed rules if available, otherwise fall back to recent reflections."""
        if hasattr(self, "_compressed_rules") and self._compressed_rules:
            return f"Control rules (from training experience):\n{self._compressed_rules}\n"
        return self.get_reflection_context(self._current_date)

    def save_reflections(self, path):
        """Save all reflection data to JSON for eval loading."""
        import json
        self._init_reflection_state()
        data = {
            "day_reflections": list(self._reflection_memory),
            "block_reflections": list(self._block_reflections),
            "by_date": {k: list(v) for k, v in self._reflection_by_date.items()},
            "compressed_rules": self._compressed_rules,
        }
        from pathlib import Path as _Path
        _Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def load_reflections(self, path):
        """Load reflection data from JSON."""
        import json
        self._init_reflection_state()
        from pathlib import Path as _Path
        data = json.loads(_Path(path).read_text())
        self._reflection_memory = data.get("day_reflections", [])
        self._block_reflections = data.get("block_reflections", [])
        self._reflection_by_date = {k: list(v) for k, v in data.get("by_date", {}).items()}
        self._compressed_rules = data.get("compressed_rules")

    # ------------------------------------------------------------------
    # Mode Selection: Reflexion-guided meta-decision for eval/deployment
    # ------------------------------------------------------------------

    def _get_outdoor_temp_from_obs(
        self, observation: dict[str, dict[str, Any]] | None,
    ) -> float | None:
        """Extract outdoor temperature from observation dict."""
        if not observation:
            return None
        for zone_obs in observation.values():
            for key in ("outdoor_drybulb_temperature", "temperature_outdoor"):
                ot = zone_obs.get(key)
                if ot is not None:
                    return float(ot)
            fc_temp = zone_obs.get("forecast_temperature_6h")
            if fc_temp is not None and len(fc_temp) > 0:
                return float(fc_temp[0])
        return None

    @staticmethod
    def _temp_bucket(temp: float | None) -> str:
        if temp is None:
            return "28-31"
        if temp < 28:
            return "<28"
        if temp < 31:
            return "28-31"
        if temp < 34:
            return "31-34"
        return ">=34"

    def _build_statistical_evidence(
        self,
        block_index: int,
        outdoor_temp: float | None,
        candidate_modes: list[str],
    ) -> str:
        """Build statistical evidence string from training data for the LLM."""
        if not hasattr(self, "_stat_lookup") or not self._stat_lookup:
            return ""

        stats = getattr(self, "_stat_full_stats", {})
        tb = self._temp_bucket(outdoor_temp)

        key = (block_index, tb)
        if key not in stats:
            # Try fallback buckets
            for fb in ["28-31", "31-34", "<28", ">=34"]:
                if (block_index, fb) in stats:
                    key = (block_index, fb)
                    tb = fb
                    break

        if key not in stats:
            recommended = self._stat_lookup.get((block_index, tb), "balanced")
            return (
                f"[Training data recommendation] → {recommended}\n"
                f"  (no detailed stats for block {block_index}, outdoor {tb}°C)\n"
            )

        s = stats[key]
        total_n = s["n"]
        lines = [
            f"[Training data: block {block_index} at outdoor_temp {tb}°C, {total_n} samples]"
        ]
        for mode in candidate_modes:
            wins = s["wins"].get(mode, 0)
            pct = wins / total_n * 100 if total_n > 0 else 0
            rewards = s["rewards"].get(mode, [])
            avg_r = sum(rewards) / len(rewards) if rewards else 0.0
            lines.append(f"  {mode}: win_rate={pct:.0f}%, avg_reward={avg_r:+.3f}")

        recommended = self._stat_lookup.get(key, "balanced")
        lines.append(f"  → Recommended: {recommended}")
        return "\n".join(lines) + "\n"

    def select_mode(
        self,
        *,
        block_index: int,
        block_start: str,
        block_end: str,
        observation: dict[str, dict[str, Any]] | None = None,
        wallclock: Any = None,
        candidate_modes: list[str] | None = None,
    ) -> str:
        """Use statistical evidence + reflexion + observation to select mode.

        The LLM sees quantitative training statistics (win rates, avg rewards)
        as primary evidence, plus current conditions and reflexion context.
        This is only used during eval/deployment (not training).

        Returns the selected mode name (e.g. "comfort", "balanced", "energy_saving").
        """
        if candidate_modes is None:
            candidate_modes = list(CANDIDATE_MODE_DESCRIPTIONS.keys())

        self._init_reflection_state()

        outdoor_temp = self._get_outdoor_temp_from_obs(observation)

        # --- Build statistical evidence (primary reference) ---
        stat_evidence = self._build_statistical_evidence(
            block_index, outdoor_temp, candidate_modes,
        )

        # --- Reflexion context (secondary reference) ---
        reflection_ctx = self.get_compressed_rules()
        block_ref_ctx = (
            self.get_block_reflection_context(block_index)
            if hasattr(self, "_block_reflections") and self._block_reflections
            else ""
        )

        # --- Build observation summary ---
        obs_lines = []
        if observation:
            for zone_id in self.zone_ids:
                zone_obs = observation.get(zone_id, {})
                temp = zone_obs.get("temperature_drybulb", "?")
                try:
                    pmv = estimate_zone_pmv(
                        temperature_drybulb=float(temp) if isinstance(temp, (int, float)) else 24.0,
                        temperature_radiant=float(temp) if isinstance(temp, (int, float)) else 24.0,
                        humidity=float(zone_obs.get("humidity", 50)),
                    )
                except Exception:
                    pmv = 0.0
                occ = zone_obs.get("occupancy", "?")
                obs_lines.append(f"  {zone_id}: temp={temp}C, PMV={pmv:+.2f}, occ={occ}")

        # --- Forecast summary ---
        forecast_lines = []
        if observation:
            first_zone = list(observation.values())[0] if observation else {}
            cloud_6h = list(first_zone.get("forecast_cloudcover_6h", []))
            precip_6h = list(first_zone.get("forecast_precip_prob_6h", []))
            pv = first_zone.get("PV", 0)
            if cloud_6h:
                forecast_lines.append(f"  Cloud cover (next 6h): {[round(float(c), 0) for c in cloud_6h[:6]]}")
            if precip_6h:
                forecast_lines.append(f"  Precip prob (next 6h): {[round(float(p), 0) for p in precip_6h[:6]]}")
            forecast_lines.append(f"  Current PV: {float(pv):.1f} kWh")
        if outdoor_temp is not None:
            forecast_lines.insert(0, f"  Outdoor temperature: {outdoor_temp:.1f}°C")

        mode_descriptions = "\n".join(
            f"  - {m}: {CANDIDATE_MODE_DESCRIPTIONS[m][:100]}"
            for m in candidate_modes
        )

        system_prompt = (
            "You are an HVAC control strategist. Choose the best control mode for this block.\n\n"
            f"Available modes:\n{mode_descriptions}\n\n"
            "You are given training statistics showing each mode's historical win rate and "
            "average reward for similar conditions. Use this as your PRIMARY reference. "
            "You may override the recommendation ONLY if current conditions (weather, PV, "
            "zone PMV) strongly suggest a different choice.\n\n"
            "Reply with ONLY the mode name (e.g. 'comfort' or 'balanced' or 'energy_saving'). "
            "No explanation needed."
        )

        user_prompt = f"Block {block_index + 1}: {block_start} to {block_end}\n"
        if wallclock:
            user_prompt += f"Current time: {wallclock}\n"
        if outdoor_temp is not None:
            user_prompt += f"Outdoor temp: {outdoor_temp:.1f}°C\n"
        user_prompt += "\n"

        # Statistical evidence first (most important)
        if stat_evidence:
            user_prompt += stat_evidence + "\n"

        if obs_lines:
            user_prompt += "Current zone states:\n" + "\n".join(obs_lines) + "\n"
        if forecast_lines:
            user_prompt += "Forecast:\n" + "\n".join(forecast_lines) + "\n"
        if reflection_ctx:
            user_prompt += "\n" + reflection_ctx
        if block_ref_ctx:
            user_prompt += "\n" + block_ref_ctx
        user_prompt += f"\nWhich mode? Choose from: {candidate_modes}"

        request = PlannerRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            payload={},
            constraints=self.constraints,
        )

        # Determine statistical fallback
        stat_fallback = None
        if hasattr(self, "_stat_lookup") and self._stat_lookup:
            tb = self._temp_bucket(outdoor_temp)
            stat_fallback = self._stat_lookup.get((block_index, tb))

        try:
            with self._inference_lock:
                raw = self.backend.generate(request)
            import re
            raw = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip().lower()
            for mode in candidate_modes:
                if mode in raw:
                    return mode
        except Exception:
            pass

        # Fallback: use statistical recommendation, then balanced
        return stat_fallback or "balanced"

    # ------------------------------------------------------------------
    # Statistical Mode Selector: data-driven, no LLM for mode selection
    # ------------------------------------------------------------------

    @staticmethod
    def build_statistical_rules(
        phase_trace_path: str,
        dataset_path: str,
        epw_path: str,
        block_mid_hours: list[int] | None = None,
        block_labels: list[str] | None = None,
    ) -> dict:
        """Build a quantitative mode selection table from training phase_trace.

        Returns a dict with:
          - 'lookup': {(block_index, temp_bucket) -> best_mode}
          - 'rules_text': human-readable rules string
          - 'stats': full stats for inspection
        """
        from collections import Counter, defaultdict
        from pathlib import Path as _Path

        # --- Load phase trace ---
        traces = []
        with open(phase_trace_path) as f:
            for line in f:
                traces.append(json.loads(line))

        # --- Date mapping from dataset ---
        dates = []
        with open(dataset_path) as f:
            for line in f:
                d = json.loads(line)
                dates.append(d["wallclock"].split(" ")[0])

        step_date = {}
        for t in traces:
            if t["phase"] == "step_start":
                step_date[t["step_index"]] = dates[t["dataset_index"] % len(dates)]

        # --- Outdoor temp from EPW ---
        outdoor_temp_by_dh: dict[tuple[str, int], float] = {}
        with open(epw_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    try:
                        yr, mo, dy, hr = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]) - 1
                        outdoor_temp_by_dh[(f"{yr}-{mo:02d}-{dy:02d}", hr)] = float(parts[6])
                    except (ValueError, IndexError):
                        pass

        # --- Determine block structure ---
        block_indices = set()
        for t in traces:
            if t["phase"] == "block_candidate_done":
                block_indices.add(t["block_index"])
        n_blocks = max(block_indices) + 1 if block_indices else 6

        if block_mid_hours is None:
            # Auto-detect from block_start phases
            block_start_hours: dict[int, list[int]] = defaultdict(list)
            for t in traces:
                if t["phase"] == "block_start" and "block_start" in t:
                    h = int(t["block_start"].split(":")[0])
                    block_start_hours[t["block_index"]].append(h)
            block_mid_hours = []
            for bi in range(n_blocks):
                if bi in block_start_hours:
                    avg_h = sum(block_start_hours[bi]) / len(block_start_hours[bi])
                    block_mid_hours.append(int(avg_h) + 1)
                else:
                    block_mid_hours.append(7 + bi * 2)

        if block_labels is None:
            block_labels = []
            for t in traces:
                if t["phase"] == "block_start":
                    bi = t["block_index"]
                    while len(block_labels) <= bi:
                        block_labels.append(f"block_{len(block_labels)}")
                    block_labels[bi] = t["block_start"]

        # --- Temp bucketing ---
        def _temp_bucket(temp: float) -> str:
            if temp < 28:
                return "<28"
            if temp < 31:
                return "28-31"
            if temp < 34:
                return "31-34"
            return ">=34"

        # --- Collect block rewards ---
        block_rewards: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
        for t in traces:
            if t["phase"] == "block_candidate_done":
                block_rewards[(t["step_index"], t["block_index"])][t["mode"]] = t["relative_block_reward"]

        # --- Aggregate stats ---
        stats: dict[tuple[int, str], dict] = defaultdict(
            lambda: {"wins": Counter(), "rewards": defaultdict(list), "n": 0}
        )
        for (step_idx, block_idx), modes in block_rewards.items():
            if len(modes) < 3:
                continue
            date = step_date.get(step_idx, "")
            if not date:
                continue
            mid_h = block_mid_hours[block_idx] if block_idx < len(block_mid_hours) else 12
            ot = outdoor_temp_by_dh.get((date, mid_h))
            tb = _temp_bucket(ot) if ot is not None else "28-31"

            winner = max(modes, key=modes.get)
            key = (block_idx, tb)
            stats[key]["wins"][winner] += 1
            stats[key]["n"] += 1
            for m, r in modes.items():
                stats[key]["rewards"][m].append(r)

        # --- Build lookup and rules text ---
        lookup: dict[tuple[int, str], str] = {}
        rules_lines = []
        for bi in range(n_blocks):
            for tb in ["<28", "28-31", "31-34", ">=34"]:
                key = (bi, tb)
                if key not in stats or stats[key]["n"] < 2:
                    continue
                s = stats[key]
                best_mode = s["wins"].most_common(1)[0][0]
                best_pct = s["wins"][best_mode] / s["n"] * 100
                best_avg = sum(s["rewards"][best_mode]) / len(s["rewards"][best_mode])
                lookup[key] = best_mode

                bl = block_labels[bi] if bi < len(block_labels) else f"block_{bi}"
                rule = (
                    f"IF block={bl} AND outdoor_temp {tb}°C "
                    f"→ {best_mode} (win={best_pct:.0f}%, n={s['n']}, avg_r={best_avg:+.3f})"
                )
                rules_lines.append(rule)

        rules_text = "\n".join(rules_lines)
        return {"lookup": lookup, "rules_text": rules_text, "stats": dict(stats)}

    def load_statistical_rules(self, rules_data: dict) -> None:
        """Load pre-built statistical rules for select_mode and statistical_select_mode."""
        self._stat_lookup: dict[tuple[int, str], str] = rules_data["lookup"]
        self._stat_rules_text: str = rules_data["rules_text"]
        self._stat_full_stats: dict = rules_data.get("stats", {})

    def statistical_select_mode(
        self,
        *,
        block_index: int,
        block_start: str,
        block_end: str,
        observation: dict[str, dict[str, Any]] | None = None,
        wallclock: Any = None,
        candidate_modes: list[str] | None = None,
    ) -> str:
        """Select mode using statistical lookup table (no LLM call).

        Looks up (block_index, temp_bucket) in the pre-built table.
        Falls back to 'balanced' if no matching rule.
        """
        if not hasattr(self, "_stat_lookup") or not self._stat_lookup:
            return "balanced"

        # Get outdoor temp from observation (use forecast_temperature_6h[0] as proxy)
        outdoor_temp = None
        if observation:
            for zone_obs in observation.values():
                # Try direct outdoor temp keys first
                for key in ("outdoor_drybulb_temperature", "temperature_outdoor"):
                    ot = zone_obs.get(key)
                    if ot is not None:
                        outdoor_temp = float(ot)
                        break
                if outdoor_temp is not None:
                    break
                # Fall back to forecast first hour
                fc_temp = zone_obs.get("forecast_temperature_6h")
                if fc_temp is not None and len(fc_temp) > 0:
                    outdoor_temp = float(fc_temp[0])
                    break

        if outdoor_temp is None:
            tb = "28-31"  # default
        elif outdoor_temp < 28:
            tb = "<28"
        elif outdoor_temp < 31:
            tb = "28-31"
        elif outdoor_temp < 34:
            tb = "31-34"
        else:
            tb = ">=34"

        mode = self._stat_lookup.get((block_index, tb))
        if mode is None:
            # Try adjacent temp buckets
            for fallback_tb in ["28-31", "31-34", "<28", ">=34"]:
                mode = self._stat_lookup.get((block_index, fallback_tb))
                if mode is not None:
                    break
        return mode or "balanced"
