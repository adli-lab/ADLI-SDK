"""Convert OTel span attributes into ADLI AgentTrace format."""
from __future__ import annotations

import json
import logging
from typing import Any

from adli_sdk.models import AgentTrace, Message, MessagePart, Usage

logger = logging.getLogger("adli_sdk")


class TraceAssembler:
    """Collects spans belonging to a single agent run and assembles an AgentTrace."""

    def __init__(self) -> None:
        self.root_span_attrs: dict[str, Any] = {}
        self.child_spans: list[dict[str, Any]] = []
        self.adli_trace_id: str = ""
        self.user_message: str = ""
        self.agent_name: str = ""
        self.framework: str = ""
        self.outcome: str = "success"

    def add_span(self, span_attrs: dict[str, Any], *, is_root: bool = False) -> None:
        if is_root:
            self.root_span_attrs = span_attrs
        else:
            self.child_spans.append(span_attrs)

    def assemble(self) -> AgentTrace:
        """Build AgentTrace from collected span data."""
        # Try PydanticAI format first (all messages on root span)
        all_messages_json = self.root_span_attrs.get("pydantic_ai.all_messages")
        if all_messages_json:
            return self._assemble_pydantic_ai(all_messages_json)

        # Fall back to generic GenAI OTel format (messages across child spans)
        return self._assemble_genai()

    def _assemble_pydantic_ai(self, all_messages_json: str) -> AgentTrace:
        try:
            raw_messages = json.loads(all_messages_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse pydantic_ai.all_messages JSON")
            raw_messages = []

        messages = _parse_pydantic_ai_messages(raw_messages)
        usage = _extract_usage(self.root_span_attrs)

        # Count tool calls from parsed messages (normalised to "tool-call")
        tool_calls = sum(
            1 for m in messages for p in m.parts
            if p.part_kind == "tool-call"
        )
        usage.tool_calls = tool_calls

        output_str = _extract_output(self.root_span_attrs, messages)

        return AgentTrace(output_str=output_str, usage=usage, messages=messages)

    def _assemble_genai(self) -> AgentTrace:
        messages: list[Message] = []
        total_usage = Usage()

        for span in self.child_spans:
            input_msgs = span.get("gen_ai.input.messages")
            if input_msgs:
                messages.extend(_parse_genai_messages(input_msgs, source="input"))

            output_msgs = span.get("gen_ai.output.messages")
            if output_msgs:
                messages.extend(_parse_genai_messages(output_msgs, source="output"))

            span_usage = _extract_usage(span)
            total_usage.input_tokens += span_usage.input_tokens
            total_usage.output_tokens += span_usage.output_tokens
            total_usage.requests += 1

        output_str = _extract_output(self.root_span_attrs, messages)
        return AgentTrace(output_str=output_str, usage=total_usage, messages=messages)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


_ROLE_TO_KIND: dict[str, str] = {
    "user": "request",
    "system": "request",
    "assistant": "response",
    "request": "request",
    "response": "response",
}

# Normalise OTel GenAI part types to the ADLI-standard names used by
# LangChain/LlamaIndex/OpenAI callbacks so the backend sees a consistent schema.
_PART_TYPE_NORMALISE: dict[str, str] = {
    "tool_call": "tool-call",
    "tool_call_response": "tool-return",
    "tool_return": "tool-return",
    "tool-call-response": "tool-return",
    "user-prompt": "user-prompt",
    "system-prompt": "system-prompt",
    "thinking": "thinking",
}


def _extract_part_content(part: dict[str, Any], part_type: str) -> str | None:
    """Extract content from a message part, handling different OTel GenAI part types."""
    # Direct content field (text, system-prompt, user-prompt, thinking, etc.)
    content = part.get("content")
    if content is not None:
        return str(content) if not isinstance(content, str) else content

    # tool_call / tool-call → synthesise "name(arguments)"
    if part_type in ("tool_call", "tool-call"):
        name = part.get("name", "")
        args = part.get("arguments", part.get("args", ""))
        if name or args:
            return f"{name}({args})" if name else str(args)

    # tool_call_response / tool-return → use result
    if part_type in ("tool_call_response", "tool-call-response", "tool-return", "tool_return"):
        result = part.get("result")
        if result is not None:
            return str(result) if not isinstance(result, str) else result

    return None


def _parse_pydantic_ai_messages(raw: list[dict]) -> list[Message]:
    """Parse PydanticAI OTel GenAI spec messages (role/type format)."""
    messages = []
    for msg in raw:
        # OTel format uses "role"; legacy uses "kind"
        raw_kind = msg.get("role") or msg.get("kind", "unknown")
        kind = _ROLE_TO_KIND.get(raw_kind, "unknown")

        parts = []
        for p in msg.get("parts", []):
            # OTel format uses "type"; legacy uses "part_kind"
            raw_type = p.get("type") or p.get("part_kind", "text")
            content = _extract_part_content(p, raw_type)
            # Normalise to ADLI-standard names (tool-call, tool-return, …)
            part_kind = _PART_TYPE_NORMALISE.get(raw_type, raw_type)
            parts.append(MessagePart(part_kind=part_kind, content=content))

        messages.append(
            Message(
                kind=kind,
                parts=parts,
                timestamp=msg.get("timestamp"),
            )
        )
    return messages


def _parse_genai_messages(raw: Any, source: str) -> list[Message]:
    """Parse GenAI OTel format (gen_ai.input.messages / gen_ai.output.messages)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

    if not isinstance(raw, list):
        return []

    messages = []
    for msg in raw:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        kind_map = {
            "system": "request",
            "user": "request",
            "assistant": "response",
            "tool": "response",
        }
        kind = kind_map.get(role, "response" if source == "output" else "request")

        part_kind_map = {
            "system": "system-prompt",
            "user": "user-prompt",
            "assistant": "text",
            "tool": "tool-return",
        }
        part_kind = part_kind_map.get(role, "text")

        parts = [MessagePart(part_kind=part_kind, content=str(content) if content else None)]
        messages.append(Message(kind=kind, parts=parts))

    return messages


def _extract_usage(attrs: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=int(attrs.get("gen_ai.usage.input_tokens", 0)),
        output_tokens=int(attrs.get("gen_ai.usage.output_tokens", 0)),
        cache_read_tokens=int(attrs.get("gen_ai.usage.details.cache_read_tokens", 0)),
        requests=1 if attrs.get("gen_ai.usage.input_tokens") else 0,
    )


def _extract_output(root_attrs: dict[str, Any], messages: list[Message]) -> str:
    # PydanticAI stores the final result as a dedicated attribute
    final_result = root_attrs.get("final_result")
    if final_result and isinstance(final_result, str):
        return final_result

    direct = root_attrs.get("gen_ai.output.messages") or root_attrs.get("output.value")
    if direct and isinstance(direct, str):
        return direct

    # Walk backwards looking for the last text content in a response message
    for msg in reversed(messages):
        if msg.kind == "response":
            for part in msg.parts:
                if part.content and part.part_kind in ("text", "user-prompt"):
                    return part.content
    return ""
