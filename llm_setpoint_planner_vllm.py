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
            chat_kwargs["tools"] = [{
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

        # Main tool-call loop
        while True:
            remaining_budget = max_output_tokens - total_completion_tokens
            if remaining_budget <= 0:
                break

            # Stop strings: "</tool_call>" always; narrative triggers only
            # before any actual tool call (avoid positive feedback loop).
            if enable_pmv_tool and tool_calls_used < max_tool_calls:
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

        text = assistant_text
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
