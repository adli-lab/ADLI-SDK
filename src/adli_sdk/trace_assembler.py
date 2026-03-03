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


def _parse_pydantic_ai_messages(raw: list[dict]) -> list[Message]:
    """Parse PydanticAI message format (parts with part_kind)."""
    messages = []
    for msg in raw:
        parts = []
        for p in msg.get("parts", []):
            parts.append(
                MessagePart(
                    part_kind=p.get("part_kind", "text"),
                    content=p.get("content"),
                )
            )
        messages.append(
            Message(
                kind=msg.get("kind", "unknown"),
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
        requests=1 if attrs.get("gen_ai.usage.input_tokens") else 0,
    )


def _extract_output(root_attrs: dict[str, Any], messages: list[Message]) -> str:
    direct = root_attrs.get("gen_ai.output.messages") or root_attrs.get("output.value")
    if direct and isinstance(direct, str):
        return direct

    for msg in reversed(messages):
        if msg.kind == "response":
            for part in msg.parts:
                if part.content:
                    return part.content
    return ""
