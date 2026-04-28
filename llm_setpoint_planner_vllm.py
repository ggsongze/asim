"""vLLM-backed planner backend for the PMV-tool-calling generate loop.

Same external interface as `Qwen35TransformersSamplingBackend.generate()`, but
uses a vLLM `LLM` engine for generation instead of `model.generate()` /
manual past_key_values forward.

Why vLLM:
  - PagedAttention + continuous batching → 3-5x throughput vs HF generate
  - Built-in prefix caching → tool-call cycle's growing prompt is reused, not
    re-prefilled every cycle (this was the O(N²) bottleneck in the original
    parent backend, also addressed by our cached backend but at lower speedup)

How the tool-call loop works in vLLM:
  - Each cycle, we call `llm.generate([full_prompt_so_far], SamplingParams(
        stop=["</tool_call>"] + narrative_triggers,
        include_stop_str_in_output=True,
    ))`
  - vLLM stops at the first matched stop string. We append the new text to
    `assistant_text`, parse the tool_call, inject the tool_response, and call
    `generate()` again with `prompt + assistant_text` (which now ends with
    the tool_response or the bridge text).
  - vLLM's prefix caching automatically reuses the prefix that's unchanged,
    so the per-cycle cost is only the new tokens since last call.

Activated by env var `ASIM_USE_VLLM_BACKEND=1`. Falls back to cached / parent
backends if not set. The `LLM` engine is constructed externally and passed in
via `__init__(llm_engine=...)`.
"""
from __future__ import annotations

import os
import re
from typing import Any

from llm_setpoint_planner import TransformersSamplingBackend, PlannerRequest
from llm_setpoint_planner_qwen35 import (
    _TOOL_CALL_BLOCK_RE,
    _parse_xml_tool_call_body,
    Qwen35TransformersSamplingBackend,
)


_NARRATIVE_TRIGGERS = ["PMV calculator", "PMV tool", "pmv calculator", "pmv tool"]


class VLLMQwen35Backend(Qwen35TransformersSamplingBackend):
    """vLLM-backed Qwen3.5 XML-format planner.

    Inherits parser overrides from `Qwen35TransformersSamplingBackend`
    (`_extract_last_tool_call_args`, `_handle_pmv_tool_call`). Replaces the
    `generate()` loop with vLLM-driven stop+resume cycles.

    The `model` attribute is set to None — we use `llm_engine` (vLLM `LLM`)
    instead. `_input_device()` is not used because vLLM manages devices.
    """

    def __init__(
        self,
        *,
        llm_engine: Any,
        tokenizer: Any,
        model_name: str | None = None,
        max_output_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        lora_request: Any = None,
        seed: int | None = None,
    ):
        # Call parent init with model=None — we won't use HF model paths.
        super().__init__(
            model=None,
            tokenizer=tokenizer,
            model_name=model_name or "Qwen/Qwen3.5-9B",
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        self.llm = llm_engine
        self.lora_request = lora_request  # vLLM LoRARequest, optional
        # Per-rollout seed: trainer sets `backend.seed = unique_int` so vLLM's
        # multinomial sampling diverges across rollouts. Without this, vLLM's
        # default RNG (shared across requests in the engine) gave identical
        # outputs at different temperatures — GRPO got 0 advantage signal.
        self.seed = seed

    def generate(self, request: PlannerRequest) -> str:
        # Local imports so this module can be imported without vLLM installed
        # (the import only fails at construction time if vLLM is missing).
        from vllm import SamplingParams

        enable_thinking = bool(int(os.environ.get("ASIM_ENABLE_THINKING", "0")))
        enable_pmv_tool = bool(int(os.environ.get("ASIM_ENABLE_PMV_TOOL", "0")))
        max_tool_calls = int(os.environ.get("ASIM_MAX_TOOL_CALLS", "30"))

        user_prompt = self._maybe_disable_qwen_thinking(request.user_prompt)
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Build prompt_text via chat template (identical to parent backend).
        chat_kwargs: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
        if "qwen3" in (self.model_name or "").lower():
            chat_kwargs["enable_thinking"] = enable_thinking
        if enable_pmv_tool:
            tools = [{
                "type": "function",
                "function": {
                    "name": "estimate_pmv",
                    "description": (
                        "Compute the PMV (Predicted Mean Vote) thermal comfort score "
                        "given zone dry-bulb temperature, relative humidity, and mean "
                        "radiant temperature. Fixed parameters: met=1.0, clo=0.5 "
                        "(summer), air_speed=0.1 m/s. Returns float PMV (typically -2 "
                        "to +2; |PMV|<0.5 is comfortable)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temp": {"type": "number"},
                            "humidity": {"type": "number"},
                            "radiant": {"type": "number"},
                        },
                        "required": ["temp", "humidity", "radiant"],
                    },
                },
            }]
            # Optional batch-PMV tool: gated by env var so old runs are
            # backward-compat. Returns a list of (temp, pmv) tuples + a
            # safe_range hint, in ONE tool call (saves 5-20 individual
            # estimate_pmv calls when scanning the safe boundary).
            if bool(int(os.environ.get("ASIM_ENABLE_PMV_RANGE_TOOL", "0"))):
                tools.append({
                    "type": "function",
                    "function": {
                        "name": "test_pmv_range",
                        "description": (
                            "Compute PMV across a range of setpoints in ONE call. "
                            "Returns a list of (temp, pmv) tuples plus a safe_range "
                            "text hint identifying the highest temp with PMV ≤ 0.4. "
                            "Use when refining the safe boundary at fine granularity. "
                            "Constraints: temp_min < temp_max (both 18-32°C); step ≥ 0.05; "
                            "max 21 points per call."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "temp_min": {"type": "number"},
                                "temp_max": {"type": "number"},
                                "step": {"type": "number"},
                                "humidity": {"type": "number"},
                                "radiant": {"type": "number"},
                            },
                            "required": ["temp_min", "temp_max", "humidity", "radiant"],
                        },
                    },
                })
            chat_kwargs["tools"] = tools
        prompt_text = self.tokenizer.apply_chat_template(messages, **chat_kwargs)

        # Sampling parameters constant across all cycles (only stop changes).
        max_output_tokens = int(self.max_output_tokens)

        # Each tool-call cycle gets a unique seed (base seed + cycle index)
        # so within one rollout, model exploration is reproducible but each
        # cycle's RNG is fresh. Across rollouts, base seed differs.
        cycle_counter = [0]

        def _make_sampling_params(remaining: int, stop_strs: list[str]) -> Any:
            sp_kwargs: dict[str, Any] = dict(
                temperature=max(float(self.temperature), 1e-5)
                    if self.temperature > 0.0 else 0.0,
                max_tokens=max(remaining, 1),
                # include_stop_str_in_output: keep "</tool_call>" in output so
                # we can parse the args from assistant_text.
                include_stop_str_in_output=True,
            )
            if self.temperature > 0.0:
                if 0.0 < self.top_p < 1.0:
                    sp_kwargs["top_p"] = float(self.top_p)
                if self.top_k > 0:
                    sp_kwargs["top_k"] = int(self.top_k)
            if self.repetition_penalty and abs(self.repetition_penalty - 1.0) > 1e-6:
                sp_kwargs["repetition_penalty"] = float(self.repetition_penalty)
            if stop_strs:
                sp_kwargs["stop"] = stop_strs
            # Seed: critical for sampling diversity across rollouts. Without
            # an explicit seed, vLLM uses a shared engine RNG that gave
            # bit-identical outputs at different temperatures.
            if self.seed is not None:
                sp_kwargs["seed"] = int(self.seed) + cycle_counter[0]
            cycle_counter[0] += 1
            return SamplingParams(**sp_kwargs)

        assistant_text = ""
        total_completion_tokens = 0
        tool_calls_used = 0
        consecutive_dup_calls = 0
        seen_call_args: set[tuple[float, float, float]] = set()
        # Soft budget warning: when 6 calls remain, inject a system-style nudge
        # to encourage the model to wrap up reasoning and emit JSON before the
        # hard cap. Fires once per generate() call.
        soft_warn_threshold = max(1, max_tool_calls - 6)
        soft_warn_emitted = False
        # Wrap-up mode: triggered by force-finalize (cap reached or 2-strike
        # dup). Once active, vLLM stop_strs becomes ["}", "<|im_end|>"] so
        # generation halts at the first JSON close, preventing the model from
        # continuing in tool_call mode after force-finalize. Also caps the
        # remaining budget to 500 tokens as a backstop. Fixes a 2026-04-27 bug
        # where dup-detection set tool_calls_used = max_tool_calls early
        # (bypassing the soft_warn 24-threshold check < max_tool_calls), then
        # stop_strs went empty (because the standard "tool_calls_used <
        # max_tool_calls" gate flipped False), leaving the model to drift
        # in tool_call style until max_output_tokens — no JSON, fallback used.
        force_finalize_emitted = False

        # Main tool-call loop
        while True:
            remaining_budget = max_output_tokens - total_completion_tokens
            if remaining_budget <= 0:
                break

            # Wrap-up mode (after force-finalize): cap remaining budget so the
            # model can't drift indefinitely if it ignores the nudge text.
            if force_finalize_emitted:
                remaining_budget = min(remaining_budget, 500)

            # Stop strings:
            #   - Wrap-up mode (post force-finalize): stop on JSON close so
            #     vLLM halts as soon as the setpoints JSON ends.
            #   - Otherwise: "</tool_call>" plus narrative triggers (only
            #     before any actual tool call, to avoid a feedback loop).
            if force_finalize_emitted:
                stop_strs = ["}", "<|im_end|>"]
            elif enable_pmv_tool and tool_calls_used < max_tool_calls:
                if tool_calls_used == 0:
                    stop_strs = ["</tool_call>"] + _NARRATIVE_TRIGGERS
                else:
                    stop_strs = ["</tool_call>"]
            else:
                stop_strs = []

            sp = _make_sampling_params(remaining_budget, stop_strs)
            full_input = prompt_text + assistant_text
            gen_kwargs: dict[str, Any] = dict(use_tqdm=False)
            if self.lora_request is not None:
                gen_kwargs["lora_request"] = self.lora_request
            outputs = self.llm.generate([full_input], sp, **gen_kwargs)
            out0 = outputs[0].outputs[0]
            new_text = out0.text
            ntoks = len(out0.token_ids)
            total_completion_tokens += ntoks
            assistant_text += new_text

            # Case 1: tool_call complete (model emitted </tool_call>)
            if (
                enable_pmv_tool
                and tool_calls_used < max_tool_calls
                and new_text.rstrip().endswith("</tool_call>")
            ):
                tool_calls_used += 1
                current_args = self._extract_last_tool_call_args(assistant_text)
                is_dup = current_args is not None and current_args in seen_call_args
                if current_args is not None:
                    seen_call_args.add(current_args)
                consecutive_dup_calls = consecutive_dup_calls + 1 if is_dup else 0
                tool_response = self._handle_pmv_tool_call(assistant_text, is_dup=is_dup)
                assistant_text += tool_response

                # Soft warning: inform model when budget is running low so it
                # can plan its remaining steps. Inject as a system-style note
                # that loops back into assistant turn, keeping the budget
                # check "advisory" (model can still call more tools).
                if (
                    not soft_warn_emitted
                    and tool_calls_used >= soft_warn_threshold
                    and tool_calls_used < max_tool_calls
                ):
                    remaining = max_tool_calls - tool_calls_used
                    nudge_soft = (
                        "<|im_end|>\n"
                        "<|im_start|>user\n"
                        f"Budget reminder: only {remaining} PMV tool calls "
                        f"remain (out of {max_tool_calls}). Wrap up your "
                        "reasoning soon and reserve enough budget to emit the "
                        'final {"setpoints": [...]} JSON. If you have already '
                        "verified safe setpoints, you may skip further tool "
                        "calls and output the JSON now."
                        "<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                    assistant_text += nudge_soft
                    soft_warn_emitted = True

                # Force-finalize on cap or 2-strike dup
                if (
                    tool_calls_used >= max_tool_calls
                    or consecutive_dup_calls >= 2
                ):
                    reason = (
                        f"You have used all {max_tool_calls} PMV tool calls"
                        if tool_calls_used >= max_tool_calls
                        else f"You have made {consecutive_dup_calls + 1} consecutive "
                             "tool calls with IDENTICAL arguments"
                    )
                    nudge = (
                        "<|im_end|>\n"
                        "<|im_start|>user\n"
                        f"{reason}. Do NOT emit more <tool_call> — they will not "
                        "be processed. Close your reasoning with </think> and "
                        'output the final {"setpoints": [...]} JSON now.'
                        "<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                    assistant_text += nudge
                    tool_calls_used = max_tool_calls
                    force_finalize_emitted = True
                continue

            # Case 2: narrative trigger before any tool call
            if (
                enable_pmv_tool
                and tool_calls_used < max_tool_calls
                and any(new_text.rstrip().endswith(trig) for trig in _NARRATIVE_TRIGGERS)
            ):
                assistant_text += ". <tool_call>"
                continue

            # Natural stop (EOS) or no recognized stop → exit
            break

        # Rescue regeneration: if the main loop ended without emitting a
        # parseable setpoints JSON (force-finalize failed, perfectionist
        # tool_call loop, etc.), do ONE clean re-generate with a strong
        # prefix that nudges the model into JSON-emit mode while keeping
        # the obs context. Gated by ASIM_ENABLE_FALLBACK_RESCUE=1 (default
        # off for back-compat).
        #
        # Strategy:
        #   - Take the original prompt_text (system + user with obs)
        #   - Append a clean `<think>...</think>` block + `{"setpoints":[`
        #   - Generate with stop=["]"] and temperature=0 (greedy, decisive)
        #   - If the continuation closes the array, splice into assistant_text
        #     so the parser finds the JSON. Otherwise keep the original
        #     (broken) output and let the planner fall back as before.
        if bool(int(os.environ.get("ASIM_ENABLE_FALLBACK_RESCUE", "0"))):
            answer_region_start = assistant_text.rfind("</think>")
            answer_region = (
                assistant_text[answer_region_start + len("</think>"):]
                if answer_region_start >= 0
                else assistant_text
            )
            if '"setpoints"' not in answer_region:
                # No valid JSON in main output → attempt rescue
                rescue_prefix = (
                    prompt_text
                    + "<think>\n"
                    + "Looking at the observations, I'll commit to a balanced "
                      "setpoint per zone (lower setpoint for high-radiant or "
                      "high-occupancy zones; higher setpoint for low-occupancy "
                      "or low-radiant zones). Outputting the JSON now.\n"
                    + "</think>\n\n"
                    + '{"setpoints": ['
                )
                rescue_sp_kwargs: dict[str, Any] = dict(
                    temperature=0.0,
                    max_tokens=80,
                    stop=["]"],
                    include_stop_str_in_output=True,
                )
                if self.seed is not None:
                    rescue_sp_kwargs["seed"] = int(self.seed) + cycle_counter[0]
                cycle_counter[0] += 1
                rescue_sp = SamplingParams(**rescue_sp_kwargs)
                rescue_gen_kwargs: dict[str, Any] = dict(use_tqdm=False)
                if self.lora_request is not None:
                    rescue_gen_kwargs["lora_request"] = self.lora_request
                try:
                    rescue_outs = self.llm.generate(
                        [rescue_prefix], rescue_sp, **rescue_gen_kwargs
                    )
                    cont = rescue_outs[0].outputs[0].text.strip()
                    # Validate: must end with ] and have 7 commas (8 floats)
                    if cont.endswith("]") and cont.count(",") >= 7:
                        # Splice rescue JSON into assistant_text so parser
                        # finds it. Use a clean </think> separator without
                        # any narrative prefix (parser strips think then
                        # extracts JSON from the remainder; non-JSON prefix
                        # text confuses _extract_json_payload).
                        rescue_json = '{"setpoints": [' + cont + '}'
                        assistant_text += "\n</think>\n\n" + rescue_json
                        if bool(int(os.environ.get("ASIM_DEBUG_THINKING", "0"))):
                            print(f"[VLLM] rescue OK: {rescue_json}", flush=True)
                    elif bool(int(os.environ.get("ASIM_DEBUG_THINKING", "0"))):
                        print(f"[VLLM] rescue FAILED (no valid array close): {cont!r}", flush=True)
                except Exception as exc:
                    if bool(int(os.environ.get("ASIM_DEBUG_THINKING", "0"))):
                        print(f"[VLLM] rescue exception: {exc}", flush=True)

        # Post-process the assistant_text so that the unified planner's
        # non-greedy regex `<think>.*?</think>` correctly strips the entire
        # thinking block (including all tool_call XML and tool_response
        # injections), leaving ONLY the final {"setpoints": ...} JSON.
        #
        # Why this is needed: with vLLM + chat template enable_thinking, the
        # model emits a separate `</think>` BEFORE EACH `<tool_call>` (so 16+
        # `</think>` tags interleaved with tool XML). The output also lacks
        # an opening `<think>` tag (it's in the prompt, not output). Without
        # post-processing, the parser strips only the first think block and
        # then `_extract_json_payload` greedy-matches across tool_call XML
        # → garbage extraction → parser fails → fallback action used.
        #
        # Strategy: find the LAST `</think>` (final answer is after it). Wrap
        # everything before in a single synthetic `<think>...</think>` so the
        # parent parser's regex match captures the whole pre-answer block.
        # If no `</think>` found, leave assistant_text as-is.
        text = assistant_text
        last_close = text.rfind("</think>")
        if last_close >= 0:
            # Strip any extra `</think>` markers inside the pre-answer block
            # so the synthetic `<think>...</think>` wrapper is well-formed.
            pre_close = text[:last_close].replace("</think>", " ")
            post_close = text[last_close + len("</think>"):]
            text = "<think>\n" + pre_close + "\n</think>\n\n" + post_close.lstrip()
        if bool(int(os.environ.get("ASIM_DEBUG_THINKING", "0"))):
            has_close = "</think>" in text
            marker = "HAS_CLOSE" if has_close else "NO_CLOSE"
            tool_marker = f" tool_calls={tool_calls_used}" if enable_pmv_tool else ""
            print(
                f"\n===== GENERATED ({len(text)} chars, {total_completion_tokens} tokens, "
                f"{marker}{tool_marker}) [vllm] =====\n"
                f"{text}\n"
                f"===== END =====\n",
                flush=True,
            )
            jsonl_path = os.environ.get("ASIM_THINKING_JSONL", "")
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
