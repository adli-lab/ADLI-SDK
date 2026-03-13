"""LangChain CallbackHandler that captures conversation traces and sends them to ADLI /learn.

This is the primary mechanism for LangChain/LangGraph integration — it works
natively with LangChain's callback system, no OTel dependency required.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any
from uuid import UUID

from adli_sdk.client import ADLIClient
from adli_sdk.flush_helpers import build_learn_request, interleave_tool_pairs
from adli_sdk.models import AgentTrace, Message, MessagePart, Usage

logger = logging.getLogger("adli_sdk")


try:
    from langchain_core.callbacks import BaseCallbackHandler as _Base
except ImportError:
    _Base = object  # type: ignore[assignment,misc]


class ADLICallbackHandler(_Base):  # type: ignore[misc]
    """LangChain callback handler that assembles an AgentTrace and sends it to ADLI /learn.

    Captures LLM inputs/outputs, tool calls, and usage metrics through
    LangChain's native callback system.  Works with chains, agents, and
    LangGraph graphs.

    The handler is designed to be created per-invocation (one handler per
    ``invoke``/``stream`` call) so that each trace is independent.
    """

    def __init__(
        self,
        *,
        client: ADLIClient,
        project_id: int,
        agent_name: str,
        adli_trace_id: str,
        user_message: str,
        framework: str = "langchain",
    ) -> None:
        self._client = client
        self._project_id = project_id
        self._agent_name = agent_name
        self._adli_trace_id = adli_trace_id
        self._user_message = user_message
        self._framework = framework

        self._lock = threading.Lock()
        self._messages: list[Message] = [
            Message(
                kind="request",
                parts=[MessagePart(part_kind="user-prompt", content=user_message)],
            )
        ]
        self._usage = Usage()
        self._outcome: str = "success"
        self._root_run_id: UUID | None = None
        self._finished = False
        self._last_system_prompt: str | None = None

    # ------------------------------------------------------------------
    # Chain lifecycle — track root run for flush
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            if self._root_run_id is None:
                self._root_run_id = run_id

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            if run_id == self._root_run_id and not self._finished:
                self._finished = True
                self._flush(outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            if run_id == self._root_run_id and not self._finished:
                self._finished = True
                self._outcome = "failure"
                self._flush(None)

    # ------------------------------------------------------------------
    # Chat model — capture system prompt on first call
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            for msg_batch in messages:
                for msg in msg_batch:
                    if getattr(msg, "type", "") != "system":
                        continue
                    content = str(getattr(msg, "content", ""))
                    if not content or content == self._last_system_prompt:
                        continue
                    self._last_system_prompt = content
                    self._messages.append(
                        Message(
                            kind="request",
                            parts=[
                                MessagePart(
                                    part_kind="system-prompt",
                                    content=content,
                                ),
                            ],
                        ),
                    )

    # ------------------------------------------------------------------
    # LLM completion — capture AI responses and usage
    # ------------------------------------------------------------------

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._collect_llm_response(response)

    def _collect_llm_response(self, response: Any) -> None:
        for gen_list in getattr(response, "generations", []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                content = getattr(msg, "content", None) if msg else getattr(gen, "text", "")
                tool_calls = getattr(msg, "tool_calls", []) if msg else []

                parts: list[MessagePart] = []
                # Use content_blocks when available (extended-thinking models)
                content_blocks = getattr(msg, "content_blocks", None) if msg else None
                if content_blocks and isinstance(content_blocks, list):
                    for block in content_blocks:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type", "")
                        if btype in ("reasoning", "thinking"):
                            text = (
                                block.get("reasoning")
                                or block.get("thinking")
                                or block.get("content", "")
                            )
                            if text:
                                parts.append(MessagePart(part_kind="thinking", content=str(text)))
                        elif btype == "text":
                            text = block.get("text") or block.get("content", "")
                            if text:
                                parts.append(MessagePart(part_kind="text", content=str(text)))
                else:
                    # Fallback: check additional_kwargs.reasoning_content (OpenRouter, etc.)
                    addl = (getattr(msg, "additional_kwargs", {}) or {}) if msg else {}
                    reasoning = (
                        addl.get("reasoning_content") or addl.get("thinking_content")
                        if isinstance(addl, dict)
                        else None
                    )
                    if reasoning:
                        parts.append(MessagePart(part_kind="thinking", content=str(reasoning)))
                    if content:
                        parts.append(MessagePart(part_kind="text", content=str(content)))

                for tc in tool_calls:
                    _name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    _args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    _id = (
                        tc.get("id") or tc.get("tool_call_id")
                        if isinstance(tc, dict)
                        else getattr(tc, "id", None) or getattr(tc, "tool_call_id", None)
                    )
                    extra: dict[str, Any] = {}
                    if _name:
                        extra["tool_name"] = _name
                    if _args:
                        extra["args"] = json.dumps(_args, ensure_ascii=False)
                    if _id:
                        extra["tool_call_id"] = str(_id)
                    parts.append(MessagePart(part_kind="tool-call", content=None, **extra))

                if parts:
                    self._messages.append(Message(kind="response", parts=parts))

                self._collect_usage_from_message(msg)

        self._collect_usage_from_llm_output(response)

    def _collect_usage_from_message(self, msg: Any) -> None:
        """Extract token usage from message.usage_metadata (newer langchain-openai)."""
        usage_meta = getattr(msg, "usage_metadata", None)
        if not usage_meta or not isinstance(usage_meta, dict):
            return
        self._usage.input_tokens += usage_meta.get("input_tokens", 0)
        self._usage.output_tokens += usage_meta.get("output_tokens", 0)
        self._usage.requests += 1

    def _collect_usage_from_llm_output(self, response: Any) -> None:
        """Extract token usage from LLMResult.llm_output (older langchain style)."""
        llm_output = getattr(response, "llm_output", None)
        if not llm_output or not isinstance(llm_output, dict):
            return
        token_usage = llm_output.get("token_usage") or {}
        if not token_usage:
            return
        # Only use if we haven't already captured from message metadata
        if self._usage.requests > 0:
            return
        self._usage.input_tokens += token_usage.get("prompt_tokens", 0)
        self._usage.output_tokens += token_usage.get("completion_tokens", 0)
        self._usage.requests += 1

    # ------------------------------------------------------------------
    # Tools — capture tool returns
    # ------------------------------------------------------------------

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        content, extra = self._parse_tool_output(output)
        with self._lock:
            self._messages.append(
                Message(
                    kind="request",
                    parts=[
                        MessagePart(
                            part_kind="tool-return",
                            content=content,
                            **extra,
                        ),
                    ],
                )
            )

    @staticmethod
    def _parse_tool_output(output: Any) -> tuple[str, dict[str, str]]:
        """Extract clean content and metadata from tool output.

        LangGraph's ToolNode passes ToolMessage objects with .content,
        .name, and .tool_call_id. Plain LangChain passes strings.
        """
        extra: dict[str, str] = {}
        if hasattr(output, "content"):
            content = str(output.content)
            name = getattr(output, "name", None)
            tc_id = getattr(output, "tool_call_id", None)
            if name:
                extra["tool_name"] = str(name)
            if tc_id:
                extra["tool_call_id"] = str(tc_id)
        else:
            content = str(output)
        return content, extra

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._messages.append(
                Message(
                    kind="request",
                    parts=[MessagePart(part_kind="tool-return", content=f"Error: {error}")],
                )
            )

    # ------------------------------------------------------------------
    # Flush assembled trace to ADLI /learn
    # ------------------------------------------------------------------

    def _flush(self, outputs: Any) -> None:
        if not self._adli_trace_id:
            logger.debug("Skipping LangChain trace without adli_trace_id")
            return

        try:
            trace = AgentTrace(
                output_str=self._extract_output(outputs),
                usage=self._usage,
                messages=interleave_tool_pairs(self._messages),
            )
            request = build_learn_request(
                agent_name=self._agent_name,
                project_id=self._project_id,
                framework=self._framework,
                adli_trace_id=self._adli_trace_id,
                user_message=self._user_message,
                outcome=self._outcome,
                trace=trace,
            )
            self._client.learn(request)
        except Exception:
            logger.warning("Failed to flush LangChain trace to ADLI /learn", exc_info=True)

    def _extract_output(self, outputs: Any) -> str:
        if isinstance(outputs, dict):
            # LangGraph returns {"messages": [BaseMessage, ...]}
            messages = outputs.get("messages")
            if isinstance(messages, list) and messages:
                last = messages[-1]
                content = getattr(last, "content", None)
                if content:
                    return str(content)
            # Regular chain outputs
            for key in ("output", "text", "result", "answer"):
                val = outputs.get(key)
                if val:
                    return str(val)
        elif isinstance(outputs, str):
            return outputs

        for msg in reversed(self._messages):
            if msg.kind == "response":
                for part in msg.parts:
                    if part.content and part.part_kind == "text":
                        return part.content
        return ""
