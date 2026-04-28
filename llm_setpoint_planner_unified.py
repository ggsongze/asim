"""Unified BlockPlanner: LLM chooses mode + setpoint together.

Subclasses BlockPlanner from llm_setpoint_planner.py, adding plan_knot_free()
which lets the LLM pick the mode as part of its output.
"""
from __future__ import annotations

import json
import re
from typing import Any

from llm_setpoint_planner import (
    BlockPlanner,
    PlannerConstraints,
    PlannerRequest,
    TransformersSamplingBackend,
    CANDIDATE_MODE_DESCRIPTIONS,
    PMV_EXPLANATION,
    PMV_TOOL_INSTRUCTIONS,
    ZONE_DESCRIPTIONS,
    DEFAULT_ZONE_IDS,
    _as_float,
    _extract_json_payload,
    estimate_zone_pmv,
    joules_to_kwh,
)

# How many minutes per knot (same as parent)
try:
    from llm_setpoint_planner import KNOT_MINUTES
except ImportError:
    KNOT_MINUTES = 10


def _get_pmv_tool_instructions() -> str:
    """Return PMV tool worked-example prompt matching the active model family.

    Switched by env var ``ASIM_TOOL_FORMAT``:
      - ``json`` (default) → Qwen3-8B JSON-style tool_call
      - ``xml``            → Qwen3.5-4B XML-style tool_call
    """
    import os as _os
    if _os.environ.get("ASIM_TOOL_FORMAT", "json").strip().lower() == "xml":
        try:
            from llm_setpoint_planner_qwen35 import PMV_TOOL_INSTRUCTIONS_XML
            return PMV_TOOL_INSTRUCTIONS_XML
        except ImportError:
            pass
    return PMV_TOOL_INSTRUCTIONS


class UnifiedBlockPlanner(BlockPlanner):
    """Extended BlockPlanner that supports free mode selection by the LLM."""

    # ------------------------------------------------------------------
    # Free-mode system prompt
    # ------------------------------------------------------------------

    def _build_knot_free_system_prompt(self) -> str:
        """System prompt for free sampling — LLM chooses mode + setpoint."""
        zone_desc_lines = []
        for zid in self.zone_ids:
            desc = ZONE_DESCRIPTIONS.get(zid, zid)
            zone_desc_lines.append(f"  {zid}: {desc}")
        zone_layout = "\n".join(zone_desc_lines)

        reflection_ctx = (
            self.get_reflection_context(self._current_date)
            if hasattr(self, "_reflection_memory") and self._reflection_memory
            else ""
        )
        block_ref_ctx = (
            self.get_block_reflection_context()
            if hasattr(self, "_block_reflections") and self._block_reflections
            else ""
        )

        mode_choices = "\n".join(
            f"  - {name}: {desc[:120]}"
            for name, desc in CANDIDATE_MODE_DESCRIPTIONS.items()
        )

        import os as _os_pmv
        _pmv_tool_block = (_get_pmv_tool_instructions() + "\n\n") if bool(int(_os_pmv.environ.get("ASIM_ENABLE_PMV_TOOL", "0"))) else ""
        prompt = (
            "You are an expert HVAC controls engineer specializing in PMV-based thermal "
            "comfort and energy optimization for a 2-story 8-zone office building.\n"
            f"Output the next {getattr(self, '_knot_minutes_override', KNOT_MINUTES)}-minute cooling setpoint for each of the {len(self.zone_ids)} zones.\n\n"
            f"{PMV_EXPLANATION}\n\n"
            f"{_pmv_tool_block}"
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
            "Choose ONE of the following strategies based on current conditions:\n"
            f"{mode_choices}\n\n"
            "Return exactly two lines:\n"
            "Line 1: strategy name (cooling, balanced, or energy_saving)\n"
            'Line 2: JSON object {"setpoints": [<float>, ...]}\n\n'
            f"Hard bounds: {self.constraints.min_setpoint_c} to {self.constraints.max_setpoint_c} C.\n"
            "PMV hard limits: all occupied zones must stay within [-0.5, +0.5].\n"
            "Round to nearest 0.1 C."
        )
        return prompt

    # ------------------------------------------------------------------
    # Setpoint-only prompts
    # ------------------------------------------------------------------

    def _build_setpoint_only_system_prompt(self) -> str:
        """System prompt for Stage 2 setpoint-only sampling."""
        zone_desc_lines = []
        for zid in self.zone_ids:
            desc = ZONE_DESCRIPTIONS.get(zid, zid)
            zone_desc_lines.append(f"  {zid}: {desc}")
        zone_layout = "\n".join(zone_desc_lines)

        reflection_ctx = (
            self.get_reflection_context(self._current_date)
            if hasattr(self, "_reflection_memory") and self._reflection_memory
            else ""
        )
        block_ref_ctx = (
            self.get_block_reflection_context()
            if hasattr(self, "_block_reflections") and self._block_reflections
            else ""
        )

        import os as _os_pmv
        _pmv_tool_block = (_get_pmv_tool_instructions() + "\n\n") if bool(int(_os_pmv.environ.get("ASIM_ENABLE_PMV_TOOL", "0"))) else ""
        prompt = (
            "You are an expert HVAC controls engineer specializing in PMV-based thermal "
            "comfort and energy optimization for a 2-story 8-zone office building.\n"
            f"Output the next {getattr(self, '_knot_minutes_override', KNOT_MINUTES)}-minute cooling setpoint for each of the {len(self.zone_ids)} zones.\n\n"
            f"{PMV_EXPLANATION}\n\n"
            f"{_pmv_tool_block}"
            "Zone layout:\n"
            f"{zone_layout}\n\n"
            "Optimize only the actual actuator values: cooling setpoints in Celsius. "
            "Do not output a strategy name, mode label, explanation, or markdown.\n"
            "Use occupancy, PMV, current zone temperatures, PV/net grid, and the 6-hour forecast. "
            "Lower setpoints improve comfort but use more cooling energy; higher setpoints save energy "
            "but can create occupied comfort violations.\n\n"
        )
        if reflection_ctx:
            prompt += reflection_ctx + "\n"
        if block_ref_ctx:
            prompt += block_ref_ctx + "\n"

        if getattr(self, "dict_json_format", False):
            _dict_skel = "{" + ", ".join(f'"{z}": <float>' for z in self.zone_ids) + "}"
            prompt += (
                "Return exactly one JSON object keyed by zone id and nothing else:\n"
                f"{_dict_skel}\n\n"
                f"Include all {len(self.zone_ids)} zones. "
                f"Hard bounds: {self.constraints.min_setpoint_c} to {self.constraints.max_setpoint_c} C.\n"
                "PMV hard limits: all occupied zones must stay within [-0.5, +0.5].\n"
                "Round to nearest 0.1 C."
            )
        else:
            prompt += (
                "Return exactly one JSON object and nothing else:\n"
                '{"setpoints": [<float>, ...]}\n\n'
                f"The setpoints array must have exactly {len(self.zone_ids)} values in the zone order given by the user.\n"
                f"Hard bounds: {self.constraints.min_setpoint_c} to {self.constraints.max_setpoint_c} C.\n"
                "PMV hard limits: all occupied zones must stay within [-0.5, +0.5].\n"
                "Round to nearest 0.1 C."
            )
        if getattr(self, "use_reasoning_template", False):
            zone_order = list(self.zone_ids)
            json_template = "{" + ", ".join(f'"{z}": ?' for z in zone_order) + "}"

            def _json_row(zone_values: dict[str, float]) -> str:
                return "{" + ", ".join(f'"{z}": {zone_values[z]}' for z in zone_order) + "}"

            ex1 = {z: (23.2 if z in ("1FSW", "0FSW") else 23.5) for z in zone_order}
            ex2 = {z: (23.7 if z in ("1FSE", "1FSW") else 24.5) for z in zone_order}
            ex3 = {z: 25.5 for z in zone_order}

            prompt += (
                "\n\n--- STRUCTURED REASONING OVERRIDE ---\n"
                "Before the JSON object, you MUST output exactly these 6 bullet lines, "
                "one per line, each filled from the observation with no extra commentary, "
                "no narrative, no reasoning sentences:\n"
                "- Outdoor temp / forecast 2h: <X>C / <Y>C\n"
                "- Occupancy level: <low | medium | high>\n"
                "- Current PMV status: <mostly cold | mixed | mostly warm>\n"
                "- Dominant constraint: <energy | comfort>\n"
                "- Hotspot zones: <comma-separated list of zone ids that need stronger cooling, "
                "based on per-zone PMV or solar exposure; use 'none' if no hotspot>\n"
                "- Setpoint differentiation: <uniform | mild | strong>, baseline <Z>C\n"
                "Semantics: 'mild' = hotspot zones about 0.3 C below baseline; "
                "'strong' = hotspot zones about 0.8 C below baseline, USE ONLY WHEN outdoor > 33 C; "
                "'uniform' = every zone equals baseline (no hotspot).\n\n"
                "Then output a JSON dict keyed by zone id (not array) on a new line:\n"
                f"JSON format (fill values): {json_template}\n\n"
                "Example 1 (midday peak, mild differentiation on west-facing hotspots):\n"
                "- Outdoor temp / forecast 2h: 31.2C / 32.5C\n"
                "- Occupancy level: high\n"
                "- Current PMV status: mixed\n"
                "- Dominant constraint: comfort\n"
                "- Hotspot zones: 0FSW, 1FSW\n"
                "- Setpoint differentiation: mild, baseline 23.5C\n"
                f"{_json_row(ex1)}\n\n"
                "Example 2 (heat wave >33C, strong differentiation on top-floor south exposure):\n"
                "- Outdoor temp / forecast 2h: 34.5C / 35.0C\n"
                "- Occupancy level: high\n"
                "- Current PMV status: mostly warm\n"
                "- Dominant constraint: comfort\n"
                "- Hotspot zones: 1FSE, 1FSW\n"
                "- Setpoint differentiation: strong, baseline 24.5C\n"
                f"{_json_row(ex2)}\n\n"
                "Example 3 (unoccupied setback, uniform with elevated baseline):\n"
                "- Outdoor temp / forecast 2h: 28.5C / 26.0C\n"
                "- Occupancy level: low\n"
                "- Current PMV status: mostly cold\n"
                "- Dominant constraint: energy\n"
                "- Hotspot zones: none\n"
                "- Setpoint differentiation: uniform, baseline 25.5C\n"
                f"{_json_row(ex3)}\n\n"
                "Output format: 6 bullet lines, then the JSON on a new line. "
                "The JSON must have one key per zone id in the above order. Nothing else."
            )
        return prompt

    def _format_setpoint_exploration_hint(self, hint: Any) -> str:
        if not hint:
            return ""
        if isinstance(hint, dict):
            label = str(hint.get("label", "weather_macro"))
            band = str(hint.get("setpoint_band", ""))
            instruction = str(hint.get("instruction", ""))
            reason = str(hint.get("reason", ""))
            guardrail = str(hint.get("guardrail", ""))
            lines = [
                "\nOptional weather-conditioned setpoint exploration hint:",
                f"- Macro: {label}",
            ]
            if band:
                lines.append(f"- Suggested setpoint band: {band}")
            if instruction:
                lines.append(f"- What to test: {instruction}")
            if reason:
                lines.append(f"- Weather reason: {reason}")
            if guardrail:
                lines.append(f"- Guardrail: {guardrail}")
            lines.append(
                "- This is optional. Ignore or soften it if occupancy, PMV, or current zone temperatures make it unsafe."
            )
            return "\n".join(lines) + "\n"
        return (
            "\nOptional weather-conditioned setpoint exploration hint:\n"
            f"- {hint}\n"
            "- This is optional. Ignore or soften it if occupancy, PMV, or current zone temperatures make it unsafe.\n"
        )

    def _build_setpoint_only_user_prompt(
        self,
        *,
        block_index: int,
        knot_index: int,
        block_start: Any,
        block_end: Any,
        observation: dict[str, dict[str, Any]],
        wallclock: Any,
        setpoint_exploration_hint: Any = None,
    ) -> str:
        """User prompt for setpoint-only Stage 2 planning."""
        prompt = self._build_knot_free_user_prompt(
            block_index=block_index,
            knot_index=knot_index,
            block_start=block_start,
            block_end=block_end,
            observation=observation,
            wallclock=wallclock,
            exploration_mode_hint=None,
        )
        # Strip the trailing strategy/example block from the free-mode prompt
        # so it does not contradict our setpoint-only "Return JSON only. No
        # strategy name." instruction below. The free-mode prompt currently
        # ends with "\nFirst line: choose strategy ...". Older revisions used
        # a "Mode-setpoint consistency reminder:" block; both markers are
        # checked so this strip stays robust to either.
        for marker in ("\nFirst line: choose strategy", "\nMode-setpoint consistency reminder:"):
            if marker in prompt:
                prompt = prompt.split(marker, 1)[0].rstrip() + "\n"
                break

        hint_text = self._format_setpoint_exploration_hint(setpoint_exploration_hint)
        if hint_text:
            prompt += hint_text

        if getattr(self, "dict_json_format", False):
            _dict_skel = "{" + ", ".join(f'"{z}": <float>' for z in self.zone_ids) + "}"
            prompt += (
                "\nReturn JSON only. No strategy name, no mode label, no markdown, no extra text.\n"
                f"Return exactly a JSON dict keyed by zone id: {_dict_skel}\n"
            )
        else:
            prompt += (
                "\nReturn JSON only. No strategy name, no mode label, no markdown, no extra text.\n"
                f"Zone order is {list(self.zone_ids)}.\n"
                f"Return exactly: {{\"setpoints\": [{', '.join(['<float>'] * len(self.zone_ids))}]}}\n"
            )
        return prompt

    # ------------------------------------------------------------------
    # Free-mode user prompt
    # ------------------------------------------------------------------

    def _build_knot_free_user_prompt(
        self,
        *,
        block_index: int,
        knot_index: int,
        block_start: Any,
        block_end: Any,
        observation: dict[str, dict[str, Any]],
        wallclock: Any,
        exploration_mode_hint: str | None = None,
    ) -> str:
        """User prompt for free sampling — same info as regular knot but no fixed mode."""
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
                f"humidity={humidity:.0f}%, occupancy={occupancy:.2f}, PMV={pmv:.2f}"
            )

        first_zone = observation.get(self.zone_ids[0], {})
        pv_kwh = joules_to_kwh(first_zone.get("PV", 0.0))
        facility_kwh = joules_to_kwh(first_zone.get("energy_consumption", 0.0))
        building_kwh = joules_to_kwh(first_zone.get("energy_building", 0.0))
        hvac_kwh = facility_kwh - building_kwh
        net_grid_kwh = hvac_kwh - pv_kwh

        forecast_lines = []
        forecast_available = bool(round(_as_float(first_zone.get("forecast_available", 0.0))))
        if forecast_available:
            # Only first 3 entries are guaranteed valid after rolling shift
            # (forecast is stale up to 3h, horizon=6, so worst case 6-3=3
            # valid entries). Tail zero-padded by the reader; we skip them.
            temp_6h = [float(x) for x in list(first_zone.get("forecast_temperature_6h", []))[:3]]
            humidity_6h = [float(x) for x in list(first_zone.get("forecast_humidity_6h", []))[:3]]
            precip_prob_6h = [float(x) for x in list(first_zone.get("forecast_precip_prob_6h", []))[:3]]
            precip_6h = [float(x) for x in list(first_zone.get("forecast_precip_6h", []))[:3]]
            # Rolling forecast: all values are real (no padding). Shows
            # +1h, +2h, +3h from CURRENT time. Underlying 6h forecast is
            # shifted by reader based on forecast issue time staleness
            # (0-3h), and we only show the 3 guaranteed-real entries.
            cloudcover_6h = [float(x) for x in list(first_zone.get("forecast_cloudcover_6h", []))[:3]]
            def _fmt_6h(vals: list[float], unit: str) -> str:
                parts = [f"+{i+1}h={v:.1f}{unit}" for i, v in enumerate(vals)]
                return ", ".join(parts)
            forecast_lines.append(f"Temp (°C):        {_fmt_6h(temp_6h, '')}")
            forecast_lines.append(f"Humidity (%):     {_fmt_6h(humidity_6h, '')}")
            if cloudcover_6h:
                forecast_lines.append(f"Cloud (%):        {_fmt_6h(cloudcover_6h, '')}")
            forecast_lines.append(f"Precip_prob (%):  {_fmt_6h(precip_prob_6h, '')}")
            forecast_lines.append(f"Precip (mm):      {_fmt_6h(precip_6h, '')}")

        # PMV hints — use full [-0.5, +0.5] range since mode not yet decided
        zone_hints = []
        for zone_id in self.zone_ids:
            zone_obs = observation.get(zone_id, {})
            drybulb = _as_float(zone_obs.get("temperature_drybulb"))
            pmv = estimate_zone_pmv(
                temperature_drybulb=drybulb,
                temperature_radiant=_as_float(zone_obs.get("temperature:radiant")),
                humidity=_as_float(zone_obs.get("humidity")),
            )
            if pmv > 0.45:
                zone_hints.append(f"  {zone_id}: PMV={pmv:+.2f} TOO HIGH → LOWER setpoint")
            elif pmv < -0.45:
                zone_hints.append(f"  {zone_id}: PMV={pmv:+.2f} TOO LOW → HIGHER setpoint")

        # Outdoor weather
        outdoor_temp = _as_float(first_zone.get("outdoor_temp", 0))
        cloud_cover = _as_float(first_zone.get("cloud_cover", 0))

        # Occupancy forecast — Miami office building's Office_OpenOff_Occ schedule
        # (hardcoded in miami_stage2.idf; repeats every weekday, 0 on weekends).
        # Exposing it lets the model precool before occupant arrivals.
        def _occ_at_hour(hhmm: float) -> float:
            if hhmm < 7.0: return 0.0
            if hhmm < 8.0: return 0.25
            if hhmm < 9.0: return 0.5
            if hhmm < 12.0: return 1.0
            if hhmm < 14.0: return 0.75
            if hhmm < 17.0: return 1.0
            if hhmm < 18.0: return 0.5
            if hhmm < 19.0: return 0.25
            return 0.0

        forecast_occ_line = ""
        try:
            import pandas as _pd
            ts = _pd.Timestamp(wallclock)
            if ts.tzinfo is not None or ts.tz is not None:
                ts = ts.tz_localize(None)
            is_weekend = ts.weekday() >= 5
            cur_hour_float = ts.hour + ts.minute / 60.0
            # Show 3 forecast points: +10min (next knot — matches your decision
            # cadence), +30min (3 knots ahead), +60min (6 knots ahead). Compact —
            # enough to see the ramp direction without over-reasoning.
            pts = []
            for k in (1, 3, 6):  # 10, 30, 60 min ahead
                hhmm = (cur_hour_float + k * 10.0 / 60.0) % 24
                occ = 0.0 if is_weekend else _occ_at_hour(hhmm)
                pts.append((k * 10, occ))
            forecast_occ_line = (
                f"Occupancy:        "
                + ", ".join(f"+{mm}min={occ:.2f}" for (mm, occ) in pts)
            )
        except Exception:
            pass

        prompt = (
            f"Block {block_index + 1}: {block_start} to {block_end}, "
            f"Knot {knot_index + 1}\n"
            f"Current time: {wallclock}\n"
            f"Outdoor: {outdoor_temp:.1f}°C, Cloud cover: {cloud_cover*10:.0f}%\n"
            f"PV generation: {pv_kwh:.2f} kWh, HVAC consumption: {hvac_kwh:.2f} kWh\n"
            f"Grid balance: {net_grid_kwh:+.2f} kWh  (positive = buying from grid, negative = PV surplus exporting)\n"
            f"Zone order: {zone_keys}\n"
            "Current zone states:\n"
            f"{chr(10).join(zone_lines)}\n"
        )
        if zone_hints:
            prompt += "PMV warnings (hard limit ±0.5):\n"
            prompt += chr(10).join(zone_hints) + "\n"
        # Unified forecast block. All values are relative to current wallclock.
        # Occupancy: 10-min grain (from IDF schedule). Weather: hourly (from
        # CSV, rolled to current time; only 3 real values, no stale padding).
        if forecast_lines or forecast_occ_line:
            prompt += "Forecast (from current time; all values are real, no padding):\n"
            if forecast_occ_line:
                prompt += "  " + forecast_occ_line + "\n"
            if forecast_lines:
                for ln in forecast_lines:
                    prompt += "  " + ln + "\n"

        # Forecast bias
        cloudcover_6h_vals = list(first_zone.get("forecast_cloudcover_6h", []))
        if forecast_available and cloudcover_6h_vals and cloud_cover > 0:
            forecast_cloud_now = float(cloudcover_6h_vals[0])
            actual_cloud_pct = cloud_cover * 10
            bias = actual_cloud_pct - forecast_cloud_now
            if abs(bias) > 15:
                direction = "cloudier" if bias > 0 else "clearer"
                pv_impact = "lower" if bias > 0 else "higher"
                prompt += (
                    f"Forecast bias: actual cloud {actual_cloud_pct:.0f}% vs forecast {forecast_cloud_now:.0f}% "
                    f"({direction} than predicted → PV may be {pv_impact} than forecast suggests)\n"
                )

        # Previous block results (block-level aggregate)
        if self._prev_block_results:
            prompt += "Previous blocks today:\n"
            for pb in self._prev_block_results:
                prompt += (
                    f"  Block {pb['block_index']+1} ({pb['block_time']}): "
                    f"reward={pb['reward']:+.3f}, "
                    f"HVAC={pb['hvac_kwh']:.0f}kWh, PMV_viol={pb['pmv']:.3f}\n"
                )

        # Previous knot result (fine-grained, only last knot).
        # Shows the model its own most recent action + decomposed physical
        # response so it can do in-context credit assignment:
        #   "last knot I pushed 1FSW to 24.0, PMV_viol went to +0.23 → back off."
        # Avoids relying purely on GRPO advantage to learn action→reward.
        if getattr(self, "_prev_knot_results", None):
            kr = self._prev_knot_results[-1]
            sp = kr["setpoints"]
            sp_arr = "[" + ", ".join(f"{sp.get(z, 24.0):.1f}" for z in self.zone_ids) + "]"
            viols = kr.get("pmv_violation_per_zone") or {}
            if viols:
                viol_str = ", ".join(
                    f"{z}{v:+.2f}" for z, v in sorted(viols.items(), key=lambda kv: -abs(kv[1]))[:3]
                )
                viol_txt = f"PMV_viol: {viol_str}"
            else:
                viol_txt = "PMV_viol: none"
            prompt += (
                f"Previous knot ({kr['wallclock'][-8:-3]}): "
                f"sp={sp_arr} HVAC={kr['hvac_kwh']:.2f}kWh  {viol_txt}\n"
            )

        if exploration_mode_hint:
            prompt += (
                "\nExploration hint for this sampled trajectory:\n"
                f"- Actively consider whether {exploration_mode_hint} is appropriate for this block.\n"
                "- This is a soft exploration hint, not a forced choice. "
                "Choose a different strategy if the observation and forecast contradict it.\n"
            )

        prompt += (
            "\nFirst line: choose strategy (cooling / balanced / energy_saving)\n"
            f"Second line: JSON with \"setpoints\" array of {len(self.zone_ids)} values\n"
            f"Example:\nbalanced\n{{\"setpoints\": [<float>, <float>, ..., <float>]}}\n"
        )
        return prompt

    # ------------------------------------------------------------------
    # Parse free-mode output
    # ------------------------------------------------------------------

    def _parse_knot_free_output(
        self, raw_output: str,
    ) -> tuple[str | None, dict[str, float] | None]:
        """Parse LLM output of 'mode_name\\n{setpoints json}'.

        Returns (mode_name, knot_dict) or (None, None) on failure.
        """
        # Strip thinking tags. Qwen3 in thinking mode sometimes emits the final
        # answer INSIDE the <think>...</think> block; first try the
        # after-</think> substring, and if that is empty fall back to the whole
        # body with just the open/close tags removed so we still see the JSON.
        text = re.sub(r"<think>.*?</think>\s*", "", raw_output, flags=re.DOTALL).strip()
        if not text:
            text = raw_output
            text = re.sub(r"<think>\s*", "", text, flags=re.DOTALL)
            text = text.replace("</think>", "").strip()

        # Try splitting on first newline
        parts = text.split("\n", 1)

        # Extract mode from first part
        mode = None
        for candidate in ("cooling", "balanced", "energy_saving"):
            if candidate in parts[0].lower():
                mode = candidate
                break

        # Extract setpoints from remaining text
        json_text = parts[1] if len(parts) > 1 else text
        knot = None
        try:
            json_payload = _extract_json_payload(json_text)
            data = json.loads(json_payload)
        except Exception:
            # Try parsing entire text as JSON
            try:
                data = json.loads(_extract_json_payload(text))
            except Exception:
                return mode, None

        # Parse setpoints array
        if isinstance(data, dict) and "setpoints" in data:
            arr = data["setpoints"]
            if isinstance(arr, list) and len(arr) == len(self.zone_ids):
                knot = {
                    zid: round(float(arr[i]), 1)
                    for i, zid in enumerate(self.zone_ids)
                }

        return mode, knot

    def _compute_fallback_setpoint(
        self, observation: dict[str, dict[str, Any]] | None
    ) -> float:
        """Pick a sensible fallback setpoint when parser fails.

        Back-compat default: when the optional `fallback_setpoint_low_occ_c`
        and `fallback_setpoint_high_occ_c` constraints are both unset,
        returns the static `fallback_setpoint_c` (24°C by default) — matching
        the pre-2026-04-28 behavior exactly.

        When occ-aware fallback is configured, picks one of three values
        based on the average zone occupancy in the current observation:

          avg_occ <= low_threshold  → fallback_setpoint_low_occ_c   (e.g. 30°C, off AC)
          avg_occ >= high_threshold → fallback_setpoint_high_occ_c  (e.g. 23.5°C, conservative cooling)
          otherwise                  → fallback_setpoint_c          (mid-default)

        The hand-off thresholds avoid the edge case where mid-occ (~0.3) gets
        the wrong fallback. If only one of low/high is configured, the other
        bucket falls through to the static default.
        """
        constraints = self.constraints
        low_c = constraints.fallback_setpoint_low_occ_c
        high_c = constraints.fallback_setpoint_high_occ_c
        # Back-compat fast path: no smart fallback configured.
        if low_c is None and high_c is None:
            return constraints.fallback_setpoint_c

        # Compute average occupancy from observation. If unavailable,
        # default to the static fallback (safest middle ground).
        if not observation:
            return constraints.fallback_setpoint_c
        occs: list[float] = []
        for zone_obs in observation.values():
            if not isinstance(zone_obs, dict):
                continue
            occ_val = zone_obs.get("occupancy")
            if occ_val is None:
                continue
            try:
                occs.append(float(occ_val))
            except (TypeError, ValueError):
                continue
        if not occs:
            return constraints.fallback_setpoint_c
        avg_occ = sum(occs) / len(occs)

        low_thr = constraints.fallback_occ_low_threshold
        high_thr = constraints.fallback_occ_high_threshold
        if avg_occ <= low_thr and low_c is not None:
            return float(low_c)
        if avg_occ >= high_thr and high_c is not None:
            return float(high_c)
        # Mid range or fallback bucket missing → static default.
        return constraints.fallback_setpoint_c

    def _parse_setpoint_only_output(self, raw_output: str) -> dict[str, float] | None:
        """Parse a setpoint-only JSON completion."""
        text = re.sub(r"<think>.*?</think>\s*", "", raw_output, flags=re.DOTALL).strip()
        if not text:
            text = raw_output
            text = re.sub(r"<think>\s*", "", text, flags=re.DOTALL)
            text = text.replace("</think>", "").strip()
        try:
            data = json.loads(_extract_json_payload(text))
        except Exception:
            return None

        if isinstance(data, dict) and "setpoints" in data:
            arr = data["setpoints"]
            if isinstance(arr, list) and len(arr) == len(self.zone_ids):
                try:
                    return {
                        zid: round(float(arr[i]), 1)
                        for i, zid in enumerate(self.zone_ids)
                    }
                except (TypeError, ValueError):
                    return None
        if isinstance(data, dict) and all(zid in data for zid in self.zone_ids):
            try:
                return {
                    zid: round(float(data[zid]), 1)
                    for zid in self.zone_ids
                }
            except (TypeError, ValueError):
                return None
        return None

    def plan_knot_setpoint_only(
        self,
        *,
        block_index: int,
        knot_index: int,
        block_start: Any,
        block_end: Any,
        observation: dict[str, dict[str, Any]] | None = None,
        wallclock: Any = None,
        setpoint_exploration_hint: Any = None,
    ) -> dict[str, Any]:
        """Generate one knot with no mode token, only setpoint JSON."""
        obs = observation if observation is not None else self._last_observation
        wc = wallclock if wallclock is not None else self._last_wallclock
        if obs is None:
            raise ValueError("No observation available for setpoint-only planning.")

        system_prompt = self._build_setpoint_only_system_prompt()
        user_prompt = self._build_setpoint_only_user_prompt(
            block_index=block_index,
            knot_index=knot_index,
            block_start=block_start,
            block_end=block_end,
            observation=obs,
            wallclock=wc,
            setpoint_exploration_hint=setpoint_exploration_hint,
        )
        request = PlannerRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            payload={},
            constraints=self.constraints,
        )

        knot = None
        raw_output = None
        for _attempt in range(self.max_generation_attempts):
            try:
                with self._inference_lock:
                    raw_output = self.backend.generate(request)
                knot = self._parse_setpoint_only_output(str(raw_output))
                if knot is not None:
                    break
            except Exception:
                continue

        if knot is None:
            fallback = self._compute_fallback_setpoint(obs)
            knot = {zone_id: fallback for zone_id in self.zone_ids}

        for zone_id in self.zone_ids:
            value = _as_float(knot.get(zone_id, self.constraints.fallback_setpoint_c))
            value = max(self.constraints.min_setpoint_c, min(self.constraints.max_setpoint_c, value))
            knot[zone_id] = round(float(value), 1)

        return {
            "knot": knot,
            "raw_output": raw_output,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "mode": "setpoint_only",
            "mode_source": "setpoint_only",
            "setpoint_exploration_hint": setpoint_exploration_hint,
            "block_index": block_index,
            "knot_index": knot_index,
        }

    # ------------------------------------------------------------------
    # plan_knot_free: LLM chooses mode + setpoint
    # ------------------------------------------------------------------

    def plan_knot_free(
        self,
        *,
        block_index: int,
        knot_index: int,
        block_start: Any,
        block_end: Any,
        observation: dict[str, dict[str, Any]] | None = None,
        wallclock: Any = None,
        exploration_mode_hint: str | None = None,
    ) -> dict[str, Any]:
        """Generate a knot where the LLM freely chooses mode + setpoint.

        Returns dict with keys: knot, mode, mode_source, raw_output,
        system_prompt, user_prompt, block_index, knot_index.
        """
        obs = observation if observation is not None else self._last_observation
        wc = wallclock if wallclock is not None else self._last_wallclock
        if obs is None:
            raise ValueError("No observation available for free knot planning.")

        system_prompt = self._build_knot_free_system_prompt()
        user_prompt = self._build_knot_free_user_prompt(
            block_index=block_index,
            knot_index=knot_index,
            block_start=block_start,
            block_end=block_end,
            observation=obs,
            wallclock=wc,
            exploration_mode_hint=exploration_mode_hint,
        )
        request = PlannerRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            payload={},
            constraints=self.constraints,
        )

        mode = None
        knot = None
        raw_output = None
        for attempt in range(self.max_generation_attempts):
            try:
                with self._inference_lock:
                    raw_output = self.backend.generate(request)
                mode, knot = self._parse_knot_free_output(str(raw_output))
                if knot is not None:
                    break
            except Exception:
                continue

        # Fallbacks
        if mode is None:
            mode = "balanced"
        if knot is None:
            fallback = self._compute_fallback_setpoint(obs)
            knot = {zone_id: fallback for zone_id in self.zone_ids}

        # Sanitize: clamp to hard bounds
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
            "mode_source": "free",
            "exploration_mode_hint": exploration_mode_hint,
            "block_index": block_index,
            "knot_index": knot_index,
        }
