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
from adli_sdk.models import AgentTrace, LearnRequest, Message, MessagePart, Usage

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
        self._system_prompt_captured = False

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
            if self._system_prompt_captured:
                return
            self._system_prompt_captured = True
            for msg_batch in messages:
                for msg in msg_batch:
                    msg_type = getattr(msg, "type", "")
                    if msg_type == "system":
                        content = getattr(msg, "content", "")
                        self._messages.insert(
                            0,
                            Message(
                                kind="request",
                                parts=[MessagePart(part_kind="system-prompt", content=str(content))],
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
                if content:
                    parts.append(MessagePart(part_kind="text", content=str(content)))
                for tc in tool_calls:
                    tc_payload = {"name": tc.get("name", ""), "args": tc.get("args", {})}
                    parts.append(MessagePart(part_kind="tool-call", content=json.dumps(tc_payload)))

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
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._messages.append(
                Message(
                    kind="request",
                    parts=[MessagePart(part_kind="tool-return", content=str(output))],
                )
            )

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
            output_str = self._extract_output(outputs)
            response_count = sum(1 for m in self._messages if m.kind == "response")

            trace = AgentTrace(
                output_str=output_str,
                usage=self._usage,
                messages=self._messages,
            )
            request = LearnRequest(
                agent_name=self._agent_name,
                project_id=self._project_id,
                framework=self._framework,
                adli_trace_id=self._adli_trace_id,
                user_message=self._user_message,
                injected=bool(self._adli_trace_id),
                outcome=self._outcome,
                steps_count=response_count,
                cost_usd=None,
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
