"""LlamaIndex CallbackHandler that captures conversation traces and sends them to ADLI /learn.

Registered into the engine's CallbackManager by ADLIWrapper before each
query/chat call, then removed and manually flushed after the call returns.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

from adli_sdk.client import ADLIClient
from adli_sdk.flush_helpers import build_learn_request
from adli_sdk.models import AgentTrace, Message, MessagePart, Usage

logger = logging.getLogger("adli_sdk")

try:
    from llama_index.core.callbacks import CBEventType
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler as _LIBase
except ImportError:
    CBEventType = None  # type: ignore[assignment]
    _LIBase = object  # type: ignore[assignment,misc]


class ADLILlamaIndexHandler(_LIBase):  # type: ignore[misc]
    """LlamaIndex callback handler that assembles an AgentTrace for ADLI /learn.

    Registered into the engine's CallbackManager by ADLIWrapper before each
    query/chat call, then removed and manually flushed after the call returns.
    """

    def __init__(
        self,
        *,
        client: ADLIClient,
        project_id: int,
        agent_name: str,
        adli_trace_id: str,
        user_message: str,
    ) -> None:
        # LlamaIndex BaseCallbackHandler requires these two lists.
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._client = client
        self._project_id = project_id
        self._agent_name = agent_name
        self._adli_trace_id = adli_trace_id
        self._user_message = user_message

        self._lock = threading.Lock()
        self._messages: list[Message] = [
            Message(
                kind="request",
                parts=[MessagePart(part_kind="user-prompt", content=user_message)],
            )
        ]
        self._usage = Usage()
        self._finished = False
        self._system_prompt_captured = False

    # ------------------------------------------------------------------
    # LlamaIndex BaseCallbackHandler interface
    # ------------------------------------------------------------------

    def on_event_start(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if CBEventType is None:
            return event_id
        payload = payload or {}

        if event_type == CBEventType.LLM:
            with self._lock:
                if not self._system_prompt_captured:
                    self._system_prompt_captured = True
                    for msg in payload.get("messages", []):
                        raw_role = getattr(msg, "role", None)
                        role = getattr(raw_role, "value", str(raw_role or ""))
                        if role == "system":
                            content = getattr(msg, "content", "")
                            self._messages.insert(
                                0,
                                Message(
                                    kind="request",
                                    parts=[
                                        MessagePart(
                                            part_kind="system-prompt",
                                            content=str(content),
                                        ),
                                    ],
                                ),
                            )
                            break

        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if CBEventType is None:
            return
        payload = payload or {}

        if event_type == CBEventType.LLM:
            with self._lock:
                self._collect_llm_response(payload.get("response"))

        elif event_type == CBEventType.FUNCTION_CALL:
            output = payload.get("tool_output", "")
            if output:
                with self._lock:
                    self._messages.append(
                        Message(
                            kind="request",
                            parts=[MessagePart(part_kind="tool-return", content=str(output))],
                        )
                    )

    def start_trace(self, trace_id: str | None = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        pass

    # ------------------------------------------------------------------
    # LLM response collection
    # ------------------------------------------------------------------

    def _collect_llm_response(self, response: Any) -> None:
        if response is None:
            return

        # ChatResponse has .message (ChatMessage with .content and .additional_kwargs)
        # CompletionResponse has .text
        msg = getattr(response, "message", None)
        text = getattr(response, "text", None) or (getattr(msg, "content", "") if msg else "")
        tool_calls = []
        if msg:
            additional = getattr(msg, "additional_kwargs", {}) or {}
            tool_calls = additional.get("tool_calls", [])

        parts: list[MessagePart] = []
        # Extract thinking/reasoning from additional_kwargs (extended-thinking models)
        if msg:
            additional = getattr(msg, "additional_kwargs", {}) or {}
            if isinstance(additional, dict):
                reasoning = (
                    additional.get("reasoning_content")
                    or additional.get("thinking_content")
                )
                if reasoning:
                    parts.append(MessagePart(part_kind="thinking", content=str(reasoning)))
        if text:
            parts.append(MessagePart(part_kind="text", content=str(text)))
        for tc in tool_calls:
            func = getattr(tc, "function", None) or (
                tc if isinstance(tc, dict) else {}
            )
            name = getattr(func, "name", "") or (
                func.get("name", "") if isinstance(func, dict) else ""
            )
            raw_args = getattr(func, "arguments", "") or (
                func.get("arguments", "")
                if isinstance(func, dict)
                else ""
            )
            tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
            extra: dict[str, Any] = {}
            if name:
                extra["tool_name"] = name
            if raw_args:
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    extra["args"] = json.dumps(parsed, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    extra["args"] = str(raw_args)
            if tc_id:
                extra["tool_call_id"] = str(tc_id)
            parts.append(MessagePart(part_kind="tool-call", content=None, **extra))

        if parts:
            self._messages.append(Message(kind="response", parts=parts))

        # Token usage — LlamaIndex stores OpenAI-compatible usage in response.raw
        raw = getattr(response, "raw", None)
        usage_data = None
        if isinstance(raw, dict):
            usage_data = raw.get("usage")
        elif raw is not None:
            usage_data = getattr(raw, "usage", None)
        if usage_data is not None:
            self._usage.input_tokens += getattr(usage_data, "prompt_tokens", 0) or 0
            self._usage.output_tokens += getattr(usage_data, "completion_tokens", 0) or 0
            self._usage.requests += 1

    # ------------------------------------------------------------------
    # Manual flush — called by wrapper after query/chat returns
    # ------------------------------------------------------------------

    def _flush(self, output_str: str, *, outcome: str = "success") -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True

        if not self._adli_trace_id:
            logger.debug("Skipping LlamaIndex trace without adli_trace_id")
            return

        try:
            trace = AgentTrace(
                output_str=output_str,
                usage=self._usage,
                messages=self._messages,
            )
            request = build_learn_request(
                agent_name=self._agent_name,
                project_id=self._project_id,
                framework="llama_index",
                adli_trace_id=self._adli_trace_id,
                user_message=self._user_message,
                outcome=outcome,
                trace=trace,
            )
            self._client.learn(request)
        except Exception:
            logger.warning("Failed to flush LlamaIndex trace to ADLI /learn", exc_info=True)
