"""KV-cache-reuse backend for the PMV-tool-calling generate loop.

Same external interface as `TransformersSamplingBackend.generate()`, but the
inner loop is rewritten so the model's KV cache is **carried across** the
tool-call cycle. The current parent implementation re-tokenizes and re-encodes
`prompt + assistant_text` from scratch every time a tool_call → tool_response
turn is injected, which is O(N²) in total compute (each new cycle re-encodes
all of the previous cycles' growing context).

This backend keeps a running `past_key_values` and only forwards:
  - the initial system+user prompt once (prefill),
  - then one new token at a time during generation,
  - then the small (~50 token) tool_response chunk after each tool call,
  - then resumes single-token generation.

For Qwen3-8B with 20+ tool calls per knot the speedup is ~3-5x in our setup.

Activated by env var `ASIM_USE_CACHED_BACKEND=1`. Without it, the original
`TransformersSamplingBackend` is unchanged.

Implementation notes:

  * Sampling uses `transformers.generation.logits_process` warpers
    (TemperatureLogitsWarper, TopPLogitsWarper, TopKLogitsWarper,
    RepetitionPenaltyLogitsProcessor) so behavior matches `model.generate()`
    closely. `no_repeat_ngram_size` is NOT applied here (we left it off in
    the v11 baseline anyway — it broke tool_call protocol).

  * Stop-string detection is done by decoding the most recent N tokens after
    every step and checking if any stop string is a suffix of the decoded
    text. The initial implementation uses a simple O(K) scan of the full
    suffix; for our short stop strings (e.g. `</tool_call>`, ~5 tokens) the
    overhead is negligible.

  * `_handle_pmv_tool_call` and `_extract_last_tool_call_args` are reused
    from the parent class — only the generate loop changes.
"""
from __future__ import annotations

import os
import re
from typing import Any, Tuple

from llm_setpoint_planner import TransformersSamplingBackend, PlannerRequest


_NARRATIVE_TRIGGERS = ["PMV calculator", "PMV tool", "pmv calculator", "pmv tool"]


def _get_logits_warpers(temperature: float, top_p: float, top_k: int):
    """Build the same logits warpers `model.generate()` would use."""
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopPLogitsWarper,
        TopKLogitsWarper,
    )
    warpers = []
    if temperature > 0.0 and abs(temperature - 1.0) > 1e-6:
        warpers.append(TemperatureLogitsWarper(max(float(temperature), 1e-5)))
    if 0.0 < top_p < 1.0:
        warpers.append(TopPLogitsWarper(float(top_p)))
    if top_k > 0:
        warpers.append(TopKLogitsWarper(int(top_k)))
    return warpers


def _get_repetition_processor(repetition_penalty: float):
    if repetition_penalty and abs(repetition_penalty - 1.0) > 1e-6:
        from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
        return RepetitionPenaltyLogitsProcessor(float(repetition_penalty))
    return None


class CachedTransformersSamplingBackend(TransformersSamplingBackend):
    """Reuses KV cache across tool-call cycles. See module docstring."""

    def generate(self, request: PlannerRequest) -> str:
        import torch

        enable_thinking = bool(int(os.environ.get("ASIM_ENABLE_THINKING", "0")))
        enable_pmv_tool = bool(int(os.environ.get("ASIM_ENABLE_PMV_TOOL", "0")))
        max_tool_calls = int(os.environ.get("ASIM_MAX_TOOL_CALLS", "30"))

        user_prompt = self._maybe_disable_qwen_thinking(request.user_prompt)
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Build prompt_text (identical to parent class).
        if hasattr(self.tokenizer, "apply_chat_template"):
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

        device = self._input_device()
        max_output_tokens = int(self.max_output_tokens)

        # Sampling helpers.
        warpers = _get_logits_warpers(
            float(self.temperature), float(self.top_p), int(self.top_k)
        )
        repetition_processor = _get_repetition_processor(float(self.repetition_penalty))
        do_sample = bool(self.temperature > 0.0 or self.top_p < 1.0 or self.top_k > 0)

        # ----- Initial prefill -----
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)

        with torch.no_grad():
            out = self.model(
                input_ids=prompt_ids,
                use_cache=True,
                return_dict=True,
            )
        past_kv = out.past_key_values
        # Track all assistant-side tokens (for repetition_penalty + stop detection).
        # We only count tokens generated AFTER the prompt; "all_ids" includes prompt
        # because the repetition processor needs the full sequence to score correctly.
        all_ids = prompt_ids  # shape [1, prompt_len]
        last_logits = out.logits[:, -1, :]   # [1, vocab]

        assistant_text = ""
        total_completion_tokens = 0
        tool_calls_used = 0
        consecutive_dup_calls = 0
        seen_call_args: set[Tuple[float, float, float]] = set()

        # Reusable helper: append a chunk of text to context (encode + forward,
        # update past_kv and last_logits). Used both for tool_response injection
        # and for the narrative-trigger ". <tool_call>" bridge.
        #
        # IMPORTANT: forwards tokens ONE AT A TIME via past_key_values. We
        # discovered (Qwen3.5-9B + transformers 4.x): multi-token forward with
        # a NON-EMPTY past_key_values produces broken logits — model emits
        # `<|User|>` instead of the correct continuation. Per-token forward
        # produces the same logits as full re-prefill. Slower in absolute
        # terms (N forward calls for an N-token chunk) but still asymptotically
        # better than the parent's full re-prefill of growing context every
        # cycle, since each per-token forward is O(1) attention against the
        # cached prefix vs O(N) re-attention from scratch.
        def _ingest_text(chunk: str):
            nonlocal past_kv, last_logits, all_ids, assistant_text, total_completion_tokens
            if not chunk:
                return
            chunk_ids = self.tokenizer(
                chunk, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(device)
            if chunk_ids.shape[1] == 0:
                return
            for i in range(chunk_ids.shape[1]):
                tok_i = chunk_ids[:, i:i+1]
                with torch.no_grad():
                    out = self.model(
                        input_ids=tok_i,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                past_kv = out.past_key_values
                last_logits = out.logits[:, -1, :]
                all_ids = torch.cat([all_ids, tok_i], dim=1)
            assistant_text += chunk
            total_completion_tokens += int(chunk_ids.shape[1])

        # Generate tokens one-by-one until a stop condition fires.
        # Returns the (decoded) text generated in this round and the matched
        # stop_string (or None if natural stop / budget hit).
        def _generate_until_stop(stop_strs: list[str]) -> tuple[str, str | None]:
            nonlocal past_kv, last_logits, all_ids, assistant_text, total_completion_tokens
            local_text = ""
            new_token_ids: list[int] = []
            # We tail-decode the LAST `tail_len` chars after each token to check
            # stop strings; large enough to span the longest stop string.
            max_stop_len = max((len(s) for s in stop_strs), default=0)
            tail_len = max(64, max_stop_len * 2)

            while True:
                if total_completion_tokens >= max_output_tokens:
                    return local_text, None
                # Apply repetition penalty + warpers on current logits.
                logits = last_logits
                if repetition_processor is not None:
                    logits = repetition_processor(all_ids, logits)
                for w in warpers:
                    logits = w(all_ids, logits)
                # Sample / argmax.
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                next_id = int(next_token.item())
                new_token_ids.append(next_id)
                # Forward one token to extend KV cache.
                with torch.no_grad():
                    out = self.model(
                        input_ids=next_token,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                past_kv = out.past_key_values
                last_logits = out.logits[:, -1, :]
                all_ids = torch.cat([all_ids, next_token], dim=1)
                total_completion_tokens += 1
                # Decode the newly generated tokens incrementally. We re-decode
                # all `new_token_ids` rather than the single new one because BPE
                # boundaries can shift; this is fast (few hundred tokens max).
                local_text = self.tokenizer.decode(
                    new_token_ids, skip_special_tokens=True
                )
                # EOS check.
                if eos_token_id is not None and next_id == int(eos_token_id):
                    return local_text, None
                # Stop string check (suffix match on decoded tail).
                if stop_strs:
                    tail = local_text[-tail_len:] if len(local_text) > tail_len else local_text
                    for s in stop_strs:
                        if tail.endswith(s):
                            return local_text, s

        # --------- Main loop ---------
        while True:
            if total_completion_tokens >= max_output_tokens:
                break

            if enable_pmv_tool and tool_calls_used < max_tool_calls:
                if tool_calls_used == 0:
                    stop_strs = ["</tool_call>"] + _NARRATIVE_TRIGGERS
                else:
                    stop_strs = ["</tool_call>"]
            else:
                stop_strs = []

            new_text, matched = _generate_until_stop(stop_strs)
            assistant_text += new_text

            # Case 1: complete tool_call emitted.
            if (
                enable_pmv_tool
                and tool_calls_used < max_tool_calls
                and matched == "</tool_call>"
            ):
                tool_calls_used += 1
                current_args = self._extract_last_tool_call_args(assistant_text)
                is_dup = current_args is not None and current_args in seen_call_args
                if current_args is not None:
                    seen_call_args.add(current_args)
                consecutive_dup_calls = consecutive_dup_calls + 1 if is_dup else 0
                tool_response = self._handle_pmv_tool_call(assistant_text, is_dup=is_dup)
                _ingest_text(tool_response)
                # 2-strike or cap finalize.
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
                    _ingest_text(nudge)
                    tool_calls_used = max_tool_calls
                continue

            # Case 2: narrative trigger fired before any real tool call.
            if (
                enable_pmv_tool
                and tool_calls_used < max_tool_calls
                and matched in _NARRATIVE_TRIGGERS
            ):
                _ingest_text(". <tool_call>")
                continue

            # No recognized stop → exit.
            break

        text = assistant_text
        if bool(int(os.environ.get("ASIM_DEBUG_THINKING", "0"))):
            has_close = "</think>" in text
            marker = "HAS_CLOSE" if has_close else "NO_CLOSE"
            tool_marker = f" tool_calls={tool_calls_used}" if enable_pmv_tool else ""
            print(
                f"\n===== GENERATED ({len(text)} chars, {total_completion_tokens} tokens, "
                f"{marker}{tool_marker}) [cached] =====\n"
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


# Optional: Qwen3.5 XML variant (inherits parser overrides from Qwen35TransformersSamplingBackend).
try:
    from llm_setpoint_planner_qwen35 import Qwen35TransformersSamplingBackend

    class CachedQwen35TransformersSamplingBackend(
        CachedTransformersSamplingBackend, Qwen35TransformersSamplingBackend
    ):
        """Qwen3.5 (XML tool format) with cached KV reuse.

        MRO: CachedQwen35 → CachedTransformers (generate, init) →
             Qwen35Transformers (XML _handle_pmv_tool_call,
             _extract_last_tool_call_args) → TransformersSamplingBackend (init).
        """
        pass
except ImportError:
    pass
