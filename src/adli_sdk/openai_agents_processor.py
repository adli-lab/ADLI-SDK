"""OpenAI Agents SDK TracingProcessor for ADLI /learn.

Register once at startup::

    adli.instrument_openai_agents()

Then use inject manually and pass the trace IDs via RunConfig metadata::

    inj = adli.inject("query", agent_name="my-agent")
    result = await Runner.run(
        agent, inj.message,
        run_config=RunConfig(metadata={
            "adli_trace_id": inj.adli_trace_id,
            "adli_agent_name": "my-agent",
            "adli_user_message": inj.message,
        }),
    )
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from adli_sdk.client import ADLIClient
from adli_sdk.flush_helpers import build_learn_request
from adli_sdk.models import AgentTrace, Message, MessagePart, Usage

logger = logging.getLogger("adli_sdk")

try:
    from agents.tracing import TracingProcessor
    from agents.tracing.spans import AgentSpanData, FunctionSpanData, GenerationSpanData
except ImportError:
    TracingProcessor = object  # type: ignore[assignment,misc]
    AgentSpanData = FunctionSpanData = GenerationSpanData = None  # type: ignore[assignment]


class _TraceState:
    __slots__ = (
        "messages", "usage", "outcome", "adli_trace_id",
        "user_message", "agent_name", "output_str", "steps_count",
    )

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.usage = Usage()
        self.outcome = "success"
        self.adli_trace_id = ""
        self.user_message = ""
        self.agent_name = ""
        self.output_str = ""
        self.steps_count = 0


class ADLIAgentsProcessor(TracingProcessor):  # type: ignore[misc]
    """OpenAI Agents SDK TracingProcessor that sends traces to ADLI /learn.

    Handles concurrent agent runs via a per-trace-id state dict.
    """

    def __init__(self, client: ADLIClient, project_id: int) -> None:
        self._client = client
        self._project_id = project_id
        self._lock = threading.Lock()
        self._traces: dict[str, _TraceState] = {}

    # ------------------------------------------------------------------
    # TracingProcessor interface
    # ------------------------------------------------------------------

    def on_trace_start(self, trace: Any) -> None:
        state = _TraceState()
        metadata = getattr(trace, "metadata", None) or {}
        state.adli_trace_id = str(metadata.get("adli_trace_id", ""))
        state.agent_name = str(metadata.get("adli_agent_name", ""))
        state.user_message = str(metadata.get("adli_user_message", ""))
        if state.user_message:
            state.messages.append(
                Message(
                    kind="request",
                    parts=[MessagePart(part_kind="user-prompt", content=state.user_message)],
                )
            )
        with self._lock:
            self._traces[str(trace.trace_id)] = state

    def on_trace_end(self, trace: Any) -> None:
        with self._lock:
            state = self._traces.pop(str(trace.trace_id), None)
        if state:
            self._flush(state)

    def on_span_start(self, span: Any) -> None:
        pass

    def on_span_end(self, span: Any) -> None:
        if GenerationSpanData is None:
            return
        tid = str(getattr(span, "trace_id", "") or "")
        with self._lock:
            state = self._traces.get(tid)
        if state is None:
            return
        if getattr(span, "error", None):
            state.outcome = "failure"
        self._process_span(span, state)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Span processing
    # ------------------------------------------------------------------

    def _process_span(self, span: Any, state: _TraceState) -> None:
        data = getattr(span, "span_data", None)
        if data is None:
            return

        if GenerationSpanData and isinstance(data, GenerationSpanData):
            self._collect_generation(data, state)
        elif FunctionSpanData and isinstance(data, FunctionSpanData):
            self._collect_function(data, state)
        elif AgentSpanData and isinstance(data, AgentSpanData):
            if not state.agent_name:
                state.agent_name = str(getattr(data, "name", "") or "")

    def _collect_generation(self, data: Any, state: _TraceState) -> None:
        # Only add input messages from the FIRST generation span.
        # Subsequent generations re-send the full history → duplicates.
        if state.steps_count == 0:
            for msg in getattr(data, "input", []) or []:
                role = str(_field(msg, "role", "user"))
                kind = "response" if role == "assistant" else "request"
                pk_map = {
                    "system": "system-prompt",
                    "user": "user-prompt",
                    "assistant": "text",
                }
                pk = pk_map.get(role, "text")
                content = str(_field(msg, "content", ""))
                state.messages.append(
                    Message(
                        kind=kind,
                        parts=[MessagePart(part_kind=pk, content=content)],
                    )
                )

        for msg in getattr(data, "output", []) or []:
            content = _field(msg, "content", "")
            parts: list[MessagePart] = []

            # Content may be a list of typed blocks or a plain string
            if isinstance(content, list):
                self._parse_content_blocks(content, parts, state)
            elif content:
                text = str(content)
                parts.append(MessagePart(part_kind="text", content=text))
                state.output_str = text

            # Extract tool_calls from output message (OpenAI format)
            raw_tcs = _field(msg, "tool_calls", []) or []
            for tc in raw_tcs:
                tc_extra: dict[str, Any] = {}
                func = _field(tc, "function", {})
                name = _field(func, "name", "")
                raw_args = _field(func, "arguments", "")
                tc_id = _field(tc, "id", "")
                if name:
                    tc_extra["tool_name"] = str(name)
                if raw_args:
                    tc_extra["args"] = str(raw_args)
                if tc_id:
                    tc_extra["tool_call_id"] = str(tc_id)
                parts.append(
                    MessagePart(part_kind="tool-call", content=None, **tc_extra)
                )

            if parts:
                state.messages.append(Message(kind="response", parts=parts))

        usage = getattr(data, "usage", None)
        if usage:
            state.usage.input_tokens += getattr(usage, "input_tokens", 0) or 0
            state.usage.output_tokens += getattr(usage, "output_tokens", 0) or 0
            state.usage.requests += 1
            state.steps_count += 1

    @staticmethod
    def _parse_content_blocks(
        blocks: list, parts: list[MessagePart], state: _TraceState
    ) -> None:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype in ("thinking", "reasoning"):
                thinking_text = (
                    block.get("thinking")
                    or block.get("reasoning")
                    or block.get("content", "")
                )
                if thinking_text:
                    parts.append(
                        MessagePart(part_kind="thinking", content=str(thinking_text))
                    )
            elif btype == "text":
                text = block.get("text") or block.get("content", "")
                if text:
                    parts.append(MessagePart(part_kind="text", content=str(text)))
                    state.output_str = str(text)
            elif btype == "tool_use":
                extra: dict[str, Any] = {}
                name = block.get("name", "")
                if name:
                    extra["tool_name"] = name
                args = block.get("input", "")
                if args:
                    extra["args"] = str(args) if not isinstance(args, str) else args
                tc_id = block.get("id", "")
                if tc_id:
                    extra["tool_call_id"] = str(tc_id)
                parts.append(
                    MessagePart(part_kind="tool-call", content=None, **extra)
                )

    def _collect_function(self, data: Any, state: _TraceState) -> None:
        name = str(getattr(data, "name", "") or "")
        output = str(getattr(data, "output", "") or "")
        if output:
            label = f"{name}: {output}" if name else output
            state.messages.append(
                Message(kind="request", parts=[MessagePart(part_kind="tool-return", content=label)])
            )

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def _flush(self, state: _TraceState) -> None:
        if not state.adli_trace_id:
            logger.debug("Skipping OpenAI Agents trace without adli_trace_id")
            return
        try:
            trace = AgentTrace(
                output_str=state.output_str,
                usage=state.usage,
                messages=state.messages,
            )
            request = build_learn_request(
                agent_name=state.agent_name or "unknown",
                project_id=self._project_id,
                framework="openai_agents",
                adli_trace_id=state.adli_trace_id,
                user_message=state.user_message,
                outcome=state.outcome,
                trace=trace,
            )
            self._client.learn(request)
        except Exception:
            logger.warning("Failed to flush OpenAI Agents trace to ADLI /learn", exc_info=True)


def _field(obj: Any, key: str, default: Any = "") -> Any:
    """Extract a field from a dict or object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
