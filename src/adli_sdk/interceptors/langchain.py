"""LangChain interceptor — invoke, ainvoke, stream, astream."""
from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from typing import Any

from adli_sdk.client import ADLIClient
from adli_sdk.models import InjectResult


def make_interceptor(
    wrapped: Any,
    method: Callable,
    method_name: str,
    client: ADLIClient,
    agent_name: str,
    input_key: str,
    project_id: int,
) -> Callable:
    """Return an intercepted version of a LangChain ``Runnable`` method."""
    is_async = method_name.startswith("a") or asyncio.iscoroutinefunction(method)

    def _prepare_config(inj: InjectResult, user_msg: str, kwargs: dict) -> dict:
        from adli_sdk.langchain_callback import ADLICallbackHandler

        handler = ADLICallbackHandler(
            client=client,
            project_id=project_id,
            agent_name=agent_name,
            adli_trace_id=inj.adli_trace_id,
            user_message=inj.message if inj.injected else user_msg,
        )
        config = kwargs.get("config") or {}
        callbacks = list(config.get("callbacks") or [])
        callbacks.append(handler)
        config["callbacks"] = callbacks
        meta = config.get("metadata") or {}
        meta["adli_trace_id"] = inj.adli_trace_id
        config["metadata"] = meta
        kwargs["config"] = config
        return kwargs

    if is_async:

        @functools.wraps(method)
        async def async_wrapper(input: Any = None, *args: Any, **kwargs: Any) -> Any:
            user_msg = _extract_user_message(input, input_key)
            inj = await client.ainject(user_msg, agent_name)
            input = _replace_user_message(input, inj.message, input_key)
            kwargs = _prepare_config(inj, user_msg, kwargs)
            return await method(input, *args, **kwargs)

        return async_wrapper

    @functools.wraps(method)
    def sync_wrapper(input: Any = None, *args: Any, **kwargs: Any) -> Any:
        user_msg = _extract_user_message(input, input_key)
        inj = client.inject(user_msg, agent_name)
        input = _replace_user_message(input, inj.message, input_key)
        kwargs = _prepare_config(inj, user_msg, kwargs)
        return method(input, *args, **kwargs)

    return sync_wrapper


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_user_message(input: Any, input_key: str) -> str:
    if isinstance(input, str):
        return input
    if isinstance(input, dict):
        val = input.get(input_key, "")
        # LangGraph format: {"messages": [HumanMessage(...), ...]}
        if input_key == "messages" and isinstance(val, list) and val:
            return _extract_from_messages(val)
        return str(val)
    return str(input)


def _extract_from_messages(messages: list) -> str:
    """Extract user prompt from last human message in a messages list."""
    for msg in reversed(messages):
        if _is_human_message(msg):
            if isinstance(msg, dict):
                return str(msg.get("content", ""))
            return str(getattr(msg, "content", ""))
    return ""


def _replace_user_message(input: Any, new_message: str, input_key: str) -> Any:
    if isinstance(input, str):
        return new_message
    if isinstance(input, dict):
        val = input.get(input_key)
        # LangGraph format: replace last human message with injected content
        if input_key == "messages" and isinstance(val, list):
            return {**input, input_key: _replace_in_messages(val, new_message)}
        return {**input, input_key: new_message}
    return new_message


def _replace_in_messages(messages: list, new_content: str) -> list:
    """Replace last human message content; append HumanMessage if none found."""
    try:
        from langchain_core.messages import HumanMessage
    except ImportError:
        HumanMessage = None  # type: ignore[assignment]

    def make_human_msg() -> Any:
        if HumanMessage:
            return HumanMessage(content=new_content)
        return {"type": "human", "content": new_content}

    for i in reversed(range(len(messages))):
        if _is_human_message(messages[i]):
            return messages[:i] + [make_human_msg()] + messages[i + 1 :]
    return list(messages or []) + [make_human_msg()]


def _is_human_message(msg: Any) -> bool:
    """Check if a message (dict or object) is a human message."""
    if isinstance(msg, dict):
        msg_type = str(msg.get("type", "")).lower()
        return msg_type in ("human", "humanmessage")
    if hasattr(msg, "type"):
        return getattr(msg, "type", "") == "human"
    return False
