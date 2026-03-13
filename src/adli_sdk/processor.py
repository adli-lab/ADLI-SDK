"""OTel SpanProcessor that captures GenAI spans and forwards traces to ADLI /learn.

This processor handles PydanticAI (via Logfire/OTel) and any other framework
that emits standard ``gen_ai.*`` OTel spans.  LangChain/LangGraph use a
separate callback-based approach — see ``langchain_callback.py``.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import StatusCode

from adli_sdk.client import ADLIClient
from adli_sdk.flush_helpers import build_learn_request
from adli_sdk.trace_assembler import TraceAssembler

logger = logging.getLogger("adli_sdk")


def _span_attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _is_genai_root(span: ReadableSpan) -> bool:
    """Detect root span of a GenAI agent run (PydanticAI or generic OTel GenAI)."""
    attrs = span.attributes or {}
    if "pydantic_ai.all_messages" in attrs:
        return True
    if "gen_ai.operation.name" in attrs:
        parent = getattr(span, "parent", None)
        if parent is None or not parent.is_valid:
            return True
    return False


def _is_genai_span(span: ReadableSpan) -> bool:
    attrs = span.attributes or {}
    for key in attrs:
        if isinstance(key, str) and (
            key.startswith("gen_ai.") or key.startswith("pydantic_ai.")
        ):
            return True
    return False


def _read_trace_id_from_metadata(attrs: dict[str, Any]) -> str:
    """Extract adli_trace_id from span metadata attributes."""
    for key in ("metadata.adli_trace_id", "adli_trace_id"):
        val = attrs.get(key)
        if val:
            return str(val)

    metadata_raw = attrs.get("metadata")
    if isinstance(metadata_raw, str):
        try:
            meta = json.loads(metadata_raw)
            if isinstance(meta, dict) and "adli_trace_id" in meta:
                return str(meta["adli_trace_id"])
        except Exception:
            pass
    elif isinstance(metadata_raw, dict):
        val = metadata_raw.get("adli_trace_id")
        if val:
            return str(val)

    return ""


def _read_adli_metadata(attrs: dict[str, Any]) -> dict[str, str]:
    """Extract adli_agent_name and adli_user_message from span attributes."""
    result: dict[str, str] = {}

    for field in ("agent_name", "user_message"):
        adli_key = f"adli_{field}"
        for prefix in ("metadata.", ""):
            val = attrs.get(f"{prefix}{adli_key}")
            if val:
                result[field] = str(val)
                break

    metadata_raw = attrs.get("metadata")
    if isinstance(metadata_raw, str):
        try:
            meta = json.loads(metadata_raw)
            if isinstance(meta, dict):
                for field in ("agent_name", "user_message"):
                    if field not in result:
                        val = meta.get(f"adli_{field}")
                        if val:
                            result[field] = str(val)
        except Exception:
            pass

    if "agent_name" not in result:
        val = attrs.get("gen_ai.agent.name")
        if val:
            result["agent_name"] = str(val)

    return result


class ADLISpanProcessor(SpanProcessor):
    """Collects GenAI OTel spans and sends assembled traces to ADLI /learn."""

    def __init__(self, client: ADLIClient, project_id: int) -> None:
        self._client = client
        self._project_id = project_id
        self._lock = threading.Lock()
        # trace_id (int) → TraceAssembler
        self._active_traces: dict[int, TraceAssembler] = {}

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        if not _is_genai_span(span):
            return

        otel_trace_id = span.context.trace_id if span.context else 0
        attrs = _span_attrs(span)
        is_root = _is_genai_root(span)

        with self._lock:
            assembler = self._active_traces.get(otel_trace_id)
            if assembler is None:
                assembler = TraceAssembler()
                self._active_traces[otel_trace_id] = assembler

            assembler.add_span(attrs, is_root=is_root)

            if is_root:
                adli_trace_id = _read_trace_id_from_metadata(attrs)
                if adli_trace_id:
                    assembler.adli_trace_id = adli_trace_id

                adli_meta = _read_adli_metadata(attrs)
                if adli_meta.get("user_message"):
                    assembler.user_message = str(adli_meta["user_message"])
                if adli_meta.get("agent_name"):
                    assembler.agent_name = str(adli_meta["agent_name"])

                is_error = (
                    span.status and span.status.status_code == StatusCode.ERROR
                )
                assembler.outcome = "failure" if is_error else "success"
                assembler.framework = self._detect_framework(attrs)

                del self._active_traces[otel_trace_id]

        if is_root:
            self._flush_trace(assembler)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    # ------------------------------------------------------------------

    def _flush_trace(self, assembler: TraceAssembler) -> None:
        if not assembler.adli_trace_id:
            logger.debug("Skipping trace without adli_trace_id")
            return

        try:
            agent_trace = assembler.assemble()
            request = build_learn_request(
                agent_name=assembler.agent_name or "unknown",
                project_id=self._project_id,
                framework=assembler.framework,
                adli_trace_id=assembler.adli_trace_id,
                user_message=assembler.user_message,
                outcome=assembler.outcome,
                trace=agent_trace,
            )
            self._client.learn(request)
        except Exception:
            logger.warning("Failed to flush trace to ADLI /learn", exc_info=True)

    @staticmethod
    def _detect_framework(attrs: dict[str, Any]) -> str:
        if "pydantic_ai.all_messages" in attrs:
            return "pydantic-ai"
        if any(isinstance(k, str) and k.startswith("gen_ai.") for k in attrs):
            return "genai"
        return ""
