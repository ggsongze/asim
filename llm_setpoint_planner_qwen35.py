"""Qwen3.5-4B variant of the planner backend.

Same interface as `llm_setpoint_planner.TransformersSamplingBackend`, but uses
Qwen3.5's XML tool-call format (the chat template forces this format and will
NOT accept the JSON form that Qwen3-8B uses).

Optional `test_pmv_range` tool (gated by env var `ASIM_ENABLE_PMV_RANGE_TOOL=1`):
  Lets the model probe a range of setpoints in ONE tool call. Returns a list
  of (temp, pmv) tuples + a `safe_range` text hint identifying the highest
  temp with PMV ≤ 0.4 (model can confirm or override). Saves tool budget when
  refining the safe boundary at 0.1°C precision; without this tool, scanning
  10 temps takes 10 individual `estimate_pmv` calls.

Tool format examples below.

    <tool_call>
    <function=estimate_pmv>
    <parameter=temp>
    24.5
    </parameter>
    <parameter=humidity>
    60
    </parameter>
    <parameter=radiant>
    24.5
    </parameter>
    </function>
    </tool_call>

This module ONLY overrides what differs (parser + worked-example prompt). All
the loop logic — stop_strings, mention-trigger, dup detection, 2-strike forced
finalize, cap-nudge — is inherited unchanged from the parent.

Keep the Qwen3-8B code in `llm_setpoint_planner.py` untouched so we can compare
the two model variants without merge conflicts.
"""
from __future__ import annotations

import re

from llm_setpoint_planner import (
    TransformersSamplingBackend,
    _estimate_pmv_tool,
)

# Range tool config — gated by env var so old experiments stay backward-compat.
PMV_RANGE_MAX_POINTS = 21        # cap to keep response small
PMV_RANGE_MIN_STEP = 0.05        # finest allowed grid
PMV_RANGE_MIN_TEMP = 18.0        # widest valid bound
PMV_RANGE_MAX_TEMP = 32.0


# ---------------------------------------------------------------------------
# Worked-example prompt extension when ASIM_ENABLE_PMV_RANGE_TOOL=1.
# Appended to PMV_TOOL_INSTRUCTIONS_XML at backend init time.
# ---------------------------------------------------------------------------
PMV_RANGE_TOOL_INSTRUCTIONS_XML = (
    "\n"
    "─── BATCH TOOL: test_pmv_range ───\n"
    "Compute PMV for a range of setpoints in ONE call. Use this when refining\n"
    "the safe boundary at 0.1°C precision — saves budget vs estimate_pmv.\n"
    "\n"
    "  <tool_call>\n"
    "  <function=test_pmv_range>\n"
    "  <parameter=temp_min>\n24.0\n</parameter>\n"
    "  <parameter=temp_max>\n26.0\n</parameter>\n"
    "  <parameter=step>\n0.5\n</parameter>\n"
    "  <parameter=humidity>\n55\n</parameter>\n"
    "  <parameter=radiant>\n29.2\n</parameter>\n"
    "  </function>\n"
    "  </tool_call>\n"
    "\n"
    "Response format:\n"
    '  <tool_response>\n'
    '  {"data": [\n'
    '     {"temp": 24.0, "pmv": 0.20},\n'
    '     {"temp": 24.5, "pmv": 0.30},\n'
    '     {"temp": 25.0, "pmv": 0.40},\n'
    '     {"temp": 25.5, "pmv": 0.48},\n'
    '     {"temp": 26.0, "pmv": 0.56}\n'
    '   ],\n'
    '   "safe_range": "25.0°C is the highest temp with PMV ≤ 0.4 (safety buffer)",\n'
    '   "n_points": 5}\n'
    '  </tool_response>\n'
    "\n"
    "How to use the response:\n"
    "  - The 'data' list maps each tested temp to its PMV.\n"
    "  - Pick the HIGHEST temp where PMV ≤ +0.4 (or ≥ -0.4 if cooling needed).\n"
    "  - 'safe_range' is a hint — verify it matches your reading of the data.\n"
    "  - Refine: if the boundary is between two temps, call again with a finer\n"
    "    step around that range (e.g. step=0.1 over a 0.5°C window).\n"
    "\n"
    "Argument constraints:\n"
    "  temp_min < temp_max (both 18-32°C); step ≥ 0.05; max 21 points per call.\n"
    "  Out-of-range or too-many-points calls return an error response.\n"
    "\n"
    "EXAMPLE — refining at 0.1°C: after the 0.5° scan above showed boundary\n"
    "between 25.0 (pmv=0.40) and 25.5 (pmv=0.48), call:\n"
    "  test_pmv_range(temp_min=25.0, temp_max=25.5, step=0.1, humidity=55, radiant=29.2)\n"
    "Returns 6 finer points → pick the highest with pmv ≤ 0.4.\n"
    "─────────────────────────────────────────\n"
)


# ---------------------------------------------------------------------------
# Worked-example prompt for Qwen3.5 XML tool format.
# ---------------------------------------------------------------------------
PMV_TOOL_INSTRUCTIONS_XML = (
    "\n"
    "═══════════════════ PMV CALCULATOR TOOL ═══════════════════\n"
    "The tool syntax is already specified by the system tools block above —\n"
    "DO NOT redefine it; just follow it. The XML form looks like:\n"
    "\n"
    "  <tool_call>\n"
    "  <function=estimate_pmv>\n"
    "  <parameter=temp>\n"
    "  24.5\n"
    "  </parameter>\n"
    "  <parameter=humidity>\n"
    "  60\n"
    "  </parameter>\n"
    "  <parameter=radiant>\n"
    "  24.5\n"
    "  </parameter>\n"
    "  </function>\n"
    "  </tool_call>\n"
    "\n"
    "The system pauses, computes PMV, and injects back:\n"
    '  <tool_response>\n  {"pmv": 0.321}\n  </tool_response>\n'
    "Then you continue thinking with the result.\n"
    "\n"
    "━━━ FULL WORKED EXAMPLE ━━━\n"
    "Observation shows Zone 1FNW: temp=22.0C, radiant=28.0C, humidity=61%, occupancy=0.5, PMV=-0.43.\n"
    "Note: radiant (28.0C) is ~6C higher than drybulb — walls/roof still warm from outside.\n"
    "<think>\n"
    "PMV=-0.43 is slightly cool but within [-0.5, +0.5]. I want to RAISE the setpoint to save\n"
    "energy while keeping PMV in range. Let me test setpoint 23.5°C — air will approach 23.5,\n"
    "but radiant stays ~28.0 (surfaces change slowly), so I pass radiant=28.0 from the obs.\n"
    "<tool_call>\n"
    "<function=estimate_pmv>\n"
    "<parameter=temp>\n23.5\n</parameter>\n"
    "<parameter=humidity>\n61\n</parameter>\n"
    "<parameter=radiant>\n28.0\n</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    '<tool_response>\n{"pmv": -0.090}\n</tool_response>\n'
    "23.5°C → PMV=-0.09, nearly neutral. Let me also try 24°C to save even more energy.\n"
    "<tool_call>\n"
    "<function=estimate_pmv>\n"
    "<parameter=temp>\n24.0\n</parameter>\n"
    "<parameter=humidity>\n61\n</parameter>\n"
    "<parameter=radiant>\n28.0\n</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    '<tool_response>\n{"pmv": 0.030}\n</tool_response>\n'
    "24°C → PMV=+0.03, essentially neutral and still ≤ +0.5. 24°C saves more energy. Pick 24.0.\n"
    "</think>\n"
    "\n"
    '{"setpoints": [24.0, ...]}\n'
    "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "Arguments:\n"
    "  temp     °C — expected indoor dry-bulb after the step (≈ your setpoint).\n"
    "  humidity %  — use the current obs value (not the forecast).\n"
    "  radiant  °C — USE the 'radiant=' value shown in the zone's obs line.\n"
    "                In summer, radiant is often 2-6°C warmer than drybulb;\n"
    "                if you pass radiant=temp you'll get a WRONG (too cool) PMV.\n"
    "Internally fixed: met=1.0, clo=0.5, air_speed=0.1 m/s — match the reward.\n"
    "\n"
    "IMPORTANT RULES:\n"
    "  1. Writing 'let me call the tool' or 'assuming PMV is X' WITHOUT emitting\n"
    "     the literal <tool_call>...</tool_call> XML produces NO result. You will\n"
    "     get NOTHING back. Always emit the XML to get a real PMV value.\n"
    "  2. Do NOT guess the PMV — let the tool give you the real value.\n"
    "  3. SAFETY BUFFER: if the tool returns |PMV| >= 0.4 you're already too\n"
    "     close to the ±0.5 limit (next step's transient may overshoot). Test\n"
    "     a more conservative setpoint until the tool reports |PMV| <= 0.4.\n"
    "  4. DO NOT call with IDENTICAL (temp, humidity, radiant) args twice —\n"
    "     duplicates are reward-penalized.\n"
    "  5. After enough probing, close </think> and output {\"setpoints\": [...]}.\n"
    "     Do NOT continue emitting <tool_call> after </think>.\n"
    "  6. ALWAYS pass the zone's observed radiant value (not drybulb) to the tool.\n"
    "\n"
    "REQUIRED WORKFLOW FOR EVERY KNOT:\n"
    "  Step A (mandatory for occupied zones): CALL estimate_pmv for the worst-PMV\n"
    "         zone with your CANDIDATE setpoint. Wait for <tool_response>. If\n"
    "         PMV is still out of [-0.5, +0.5], adjust and call again.\n"
    "  Step B: Among the verified-safe candidates (|PMV| <= 0.4), CHOOSE based\n"
    "         on the situation — there is no fixed 'best' setpoint. Higher\n"
    "         cooling setpoint = more energy savings but tighter PMV margin\n"
    "         (riskier if the forecast shows heat coming or occupancy is\n"
    "         about to rise). Lower setpoint = larger comfort buffer but\n"
    "         more energy. Weigh: (i) current PMV trend per zone, (ii) the\n"
    "         next 1-3h forecast, (iii) occupancy now and in 30-60 min.\n"
    "         The reward function will signal whether you struck the right\n"
    "         balance — explore both directions across knots.\n"
    "  Step C: Emit </think> then the final JSON answer.\n"
    "\n"
    "BUDGET DISCIPLINE: aim for 7-15 tool calls per knot (hard cap is 30).\n"
    "MANDATORY MINIMUM: emit at least 5 <tool_call> blocks before closing\n"
    "</think>. The PMV calculator's verified values are how you discover the\n"
    "safe energy-saving setpoint per zone group; skipping the tool leads to\n"
    "PMV breaches and bad reward. Test enough candidates to bracket the safe\n"
    "edge for the distinct zone groups (1F vs 0F, north vs south), but do NOT\n"
    "sweep every 0.1°C — stop calling once you have enough info to pick.\n"
    "\n"
    "REASONING STYLE: think CONCISELY over ALL relevant factors before finalizing\n"
    "— do NOT skip reasoning, but do NOT belabor it either. Briefly cover (in\n"
    "order):\n"
    "  (a) which zones are warmest (highest radiant) → set lower setpoint there;\n"
    "  (b) which zones are coolest → can use higher setpoint to save energy;\n"
    "  (c) current PMV per zone (use the obs values, don't recompute);\n"
    "  (d) forecast trend (will it get hotter? consider precooling);\n"
    "  (e) which candidate setpoints to TEST with the tool (typically 1-2 per\n"
    "      zone group).\n"
    "A few short sentences per factor is enough — DON'T write paragraphs.\n"
    "Wasted tokens on verbose deliberation can leave no room for the final JSON.\n"
    "═══════════════════════════════════════════════════════════"
)


# ---------------------------------------------------------------------------
# Backend with XML tool_call parser
# ---------------------------------------------------------------------------
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_BLOCK_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _parse_xml_tool_call_body(tc_body: str) -> tuple[str | None, dict[str, str]]:
    """Parse the inner of <tool_call>...</tool_call> for function name + params.

    Returns (function_name, {param_name: param_value_str}). Function name is
    None if no <function=...> tag is found. Parameter values are stripped of
    surrounding whitespace.
    """
    fn_match = _FUNCTION_BLOCK_RE.search(tc_body)
    if not fn_match:
        return None, {}
    name = fn_match.group(1).strip()
    inner = fn_match.group(2)
    params: dict[str, str] = {}
    for pname, pval in _PARAMETER_RE.findall(inner):
        params[pname.strip()] = pval.strip()
    return name, params


class Qwen35TransformersSamplingBackend(TransformersSamplingBackend):
    """Qwen3.5-4B sampling backend.

    Inherits the full generate() loop from the parent (stop_strings,
    narrative-trigger forcing, dup detection, 2-strike finalize, cap-nudge).
    Overrides only the two methods that touch tool-call format:
      - `_extract_last_tool_call_args`  (used by dup detection)
      - `_handle_pmv_tool_call`          (parses + responds)
    """

    def _extract_last_tool_call_args(
        self, assistant_text: str
    ) -> tuple | None:
        """Parse the most recent XML tool_call's args for dup detection.

        Returns a tool-specific tuple key:
          - estimate_pmv:    (name, temp, humidity, radiant)
          - test_pmv_range:  (name, temp_min, temp_max, step, humidity, radiant)
          - other / unknown: None
        Different tools yield different-length tuples → never collide for dup.
        """
        matches = list(_TOOL_CALL_BLOCK_RE.finditer(assistant_text))
        if not matches:
            return None
        name, params = _parse_xml_tool_call_body(matches[-1].group(1))
        try:
            if name == "estimate_pmv":
                return (
                    "estimate_pmv",
                    round(float(params.get("temp", "0")), 2),
                    round(float(params.get("humidity", params.get("rh", "0"))), 1),
                    round(float(params.get("radiant", params.get("tr", "0"))), 2),
                )
            if name == "test_pmv_range":
                return (
                    "test_pmv_range",
                    round(float(params.get("temp_min", "0")), 2),
                    round(float(params.get("temp_max", "0")), 2),
                    round(float(params.get("step", "0.5")), 2),
                    round(float(params.get("humidity", params.get("rh", "0"))), 1),
                    round(float(params.get("radiant", params.get("tr", "0"))), 2),
                )
        except Exception:
            return None
        return None

    def _handle_pmv_tool_call(
        self, assistant_text: str, *, is_dup: bool = False
    ) -> str:
        """Parse the most recent XML <tool_call>, dispatch by name, return wrapped response."""
        def _wrap(body: str) -> str:
            return (
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"<tool_response>\n{body}\n</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        matches = list(_TOOL_CALL_BLOCK_RE.finditer(assistant_text))
        if not matches:
            return _wrap('{"error": "no tool_call parsed"}')
        name, params = _parse_xml_tool_call_body(matches[-1].group(1))
        if name is None:
            return _wrap('{"error": "no <function=...> tag in tool_call"}')

        # Dispatch by tool name
        if name == "estimate_pmv":
            return _wrap(self._handle_estimate_pmv(params, is_dup=is_dup))
        if name == "test_pmv_range":
            return _wrap(self._handle_test_pmv_range(params, is_dup=is_dup))
        return _wrap(f'{{"error": "unknown tool {name}"}}')

    def _handle_estimate_pmv(self, params: dict[str, str], *, is_dup: bool) -> str:
        """Single-point PMV calculation (original tool body extracted into a helper)."""
        try:
            temp = float(params.get("temp", params.get("temperature", "24")))
            humidity = float(params.get("humidity", params.get("rh", "60")))
            radiant_raw = params.get("radiant", params.get("tr", params.get("radiant_temp")))
            radiant = float(radiant_raw) if radiant_raw is not None else temp
            pmv_val = _estimate_pmv_tool(
                temp=temp, humidity=humidity, radiant=radiant
            )
            warnings: list[str] = []
            if is_dup:
                warnings.append(
                    "DUPLICATE call — you already have this result. "
                    "Either vary (temp, humidity, radiant) or close </think> and "
                    "output the final setpoints JSON."
                )
            # Near-limit buffer: |PMV| in [0.4, 0.5) means next env step's
            # transient PMV may breach ±0.5 and trigger the 50× comfort
            # penalty. Tell the model to leave ≥0.1 buffer.
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
                joined = " | ".join(warnings).replace('"', "'")
                return f'{{"pmv": {pmv_val:.3f}, "warning": "{joined}"}}'
            return f'{{"pmv": {pmv_val:.3f}}}'
        except Exception as exc:
            return f'{{"error": "calc failed: {str(exc)[:80]}"}}'

    def _handle_test_pmv_range(self, params: dict[str, str], *, is_dup: bool) -> str:
        """Batch-PMV calculation across [temp_min, temp_max] with given step.

        Hybrid response format: raw `data` list + `safe_range` text hint.
        Lets the model verify and refine instead of trusting the hint blindly.
        """
        try:
            temp_min = float(params.get("temp_min", "20"))
            temp_max = float(params.get("temp_max", "30"))
            step = float(params.get("step", "0.5"))
            humidity = float(params.get("humidity", params.get("rh", "60")))
            radiant_raw = params.get("radiant", params.get("tr", params.get("radiant_temp")))
            if radiant_raw is None:
                return '{"error": "radiant parameter is required"}'
            radiant = float(radiant_raw)

            # Validate constraints
            if not (PMV_RANGE_MIN_TEMP <= temp_min < temp_max <= PMV_RANGE_MAX_TEMP):
                return (
                    f'{{"error": "temp_min={temp_min} must be < temp_max={temp_max}, '
                    f'both in [{PMV_RANGE_MIN_TEMP}, {PMV_RANGE_MAX_TEMP}]"}}'
                )
            if step < PMV_RANGE_MIN_STEP:
                return f'{{"error": "step={step} too small (min {PMV_RANGE_MIN_STEP})"}}'
            n_points = int(round((temp_max - temp_min) / step)) + 1
            if n_points > PMV_RANGE_MAX_POINTS:
                return (
                    f'{{"error": "{n_points} points exceeds max {PMV_RANGE_MAX_POINTS}; '
                    f'use larger step or narrower range"}}'
                )

            # Compute PMV at each grid temp
            data_pairs: list[tuple[float, float]] = []
            for i in range(n_points):
                t = round(temp_min + i * step, 2)
                if t > temp_max + 1e-6:
                    break
                pmv_val = _estimate_pmv_tool(temp=t, humidity=humidity, radiant=radiant)
                data_pairs.append((t, round(float(pmv_val), 3)))

            # Build safe_range hint. "Safe" band = PMV in [-0.4, +0.4]
            # (the 0.1 buffer below the ±0.5 reward-penalty edge).
            in_band = [(t, p) for t, p in data_pairs if -0.4 <= p <= 0.4]
            too_warm = [(t, p) for t, p in data_pairs if p > 0.4]
            too_cold = [(t, p) for t, p in data_pairs if p < -0.4]

            if in_band:
                # At least one tested temp is in the safe band → highlight extremes
                safe_high = in_band[-1][0]   # highest temp still in safe band
                safe_low = in_band[0][0]
                if too_warm:
                    next_warm = too_warm[0][0]
                    safe_msg = (
                        f"{safe_high}°C is the highest temp with PMV ≤ 0.4 "
                        f"(safety buffer); {next_warm}°C breaches the buffer"
                    )
                elif too_cold:
                    next_cold = too_cold[-1][0]
                    safe_msg = (
                        f"{safe_low}°C is the lowest temp with PMV ≥ -0.4 "
                        f"(safety buffer); {next_cold}°C is too cold"
                    )
                else:
                    # Whole range is in band — model could explore further
                    safe_msg = (
                        f"all tested temps are in safe band [-0.4, +0.4]; "
                        f"range {safe_low}–{safe_high}°C all comfortable. "
                        "Consider testing higher temps for more energy savings."
                    )
            elif too_warm and not too_cold:
                lowest_t = data_pairs[0][0]
                safe_msg = (
                    f"all tested temps too warm (PMV > 0.4); "
                    f"try lower temps below {lowest_t}°C"
                )
            elif too_cold and not too_warm:
                highest_t = data_pairs[-1][0]
                safe_msg = (
                    f"all tested temps too cold (PMV < -0.4); "
                    f"try higher temps above {highest_t}°C"
                )
            else:
                safe_msg = (
                    "tested range straddles both ends — no temp in safe band; "
                    "humidity or radiant may be extreme"
                )

            warnings: list[str] = []
            if is_dup:
                warnings.append(
                    "DUPLICATE range call — you already have these results. "
                    "Either vary the range or close </think> and output the JSON."
                )

            data_json = ", ".join(
                f'{{"temp": {t}, "pmv": {p:.3f}}}' for t, p in data_pairs
            )
            warning_field = (
                f', "warning": "{ " | ".join(warnings).replace(chr(34), chr(39)) }"'
                if warnings else ""
            )
            return (
                f'{{"data": [{data_json}], '
                f'"safe_range": "{safe_msg}", '
                f'"n_points": {len(data_pairs)}'
                f'{warning_field}}}'
            )
        except Exception as exc:
            return f'{{"error": "range calc failed: {str(exc)[:80]}"}}'
