"""Transparent __getattr__ proxy that intercepts agent entry points for inject."""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable

from adli_sdk.client import ADLIClient
from adli_sdk.models import InjectResult

logger = logging.getLogger("adli_sdk")

# Methods to intercept per framework type.
_PYDANTIC_AI_METHODS = frozenset({"run", "run_sync", "run_stream", "iter"})
_LANGCHAIN_METHODS = frozenset({"invoke", "ainvoke", "stream", "astream"})


def _is_pydantic_ai_agent(obj: Any) -> bool:
    cls_name = type(obj).__qualname__
    module = type(obj).__module__ or ""
    return "Agent" in cls_name and "pydantic_ai" in module


def _is_langchain_runnable(obj: Any) -> bool:
    try:
        from langchain_core.runnables import Runnable

        return isinstance(obj, Runnable)
    except ImportError:
        return False


class ADLIWrapper:
    """Transparent proxy around an agent or chain.

    All attributes and methods are delegated to the wrapped object via
    ``__getattr__``.  Only the entry-point methods that accept a user prompt
    are intercepted to call ``/inject`` and attach ``adli_trace_id`` to the
    framework's native metadata.
    """

    def __init__(
        self,
        wrapped: Any,
        *,
        client: ADLIClient,
        agent_name: str,
        input_key: str = "input",
        project_id: int = 0,
    ) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_agent_name", agent_name)
        object.__setattr__(self, "_input_key", input_key)
        object.__setattr__(self, "_project_id", project_id)
        object.__setattr__(self, "_intercept", _detect_intercepts(wrapped))

    # --- transparent delegation -------------------------------------------

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._wrapped, name)
        if name in self._intercept and callable(attr):
            return self._with_inject(attr, name)
        return attr

    def __repr__(self) -> str:
        return f"ADLIWrapper({self._wrapped!r}, agent_name={self._agent_name!r})"

    # --- inject layer -----------------------------------------------------

    def _with_inject(self, method: Callable, method_name: str) -> Callable:
        """Return a thin wrapper that injects before delegating."""

        if _is_pydantic_ai_agent(self._wrapped):
            return self._pydantic_ai_interceptor(method, method_name)
        return self._langchain_interceptor(method, method_name)

    # -- PydanticAI --------------------------------------------------------

    def _pydantic_ai_interceptor(self, method: Callable, method_name: str) -> Callable:
        client = self._client
        agent_name = self._agent_name

        if method_name == "run_sync":

            @functools.wraps(method)
            def sync_wrapper(user_prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
                if user_prompt is not None and isinstance(user_prompt, str):
                    inj = client.inject(user_prompt, agent_name)
                    user_prompt = inj.message
                    meta = kwargs.get("metadata") or {}
                    meta["adli_trace_id"] = inj.adli_trace_id
                    meta["adli_agent_name"] = agent_name
                    meta["adli_user_message"] = inj.message if inj.injected else user_prompt
                    kwargs["metadata"] = meta
                return method(user_prompt, *args, **kwargs)

            return sync_wrapper

        @functools.wraps(method)
        async def async_wrapper(user_prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
            if user_prompt is not None and isinstance(user_prompt, str):
                inj = await client.ainject(user_prompt, agent_name)
                user_prompt = inj.message
                meta = kwargs.get("metadata") or {}
                meta["adli_trace_id"] = inj.adli_trace_id
                meta["adli_agent_name"] = agent_name
                meta["adli_user_message"] = inj.message if inj.injected else user_prompt
                kwargs["metadata"] = meta
            return await method(user_prompt, *args, **kwargs)

        return async_wrapper

    # -- LangChain ---------------------------------------------------------

    def _langchain_interceptor(self, method: Callable, method_name: str) -> Callable:
        client = self._client
        agent_name = self._agent_name
        input_key = self._input_key
        project_id = self._project_id

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_intercepts(wrapped: Any) -> frozenset[str]:
    if _is_pydantic_ai_agent(wrapped):
        return _PYDANTIC_AI_METHODS
    if _is_langchain_runnable(wrapped):
        return _LANGCHAIN_METHODS
    logger.warning(
        "ADLIWrapper: unknown object type %s — no methods will be intercepted",
        type(wrapped).__qualname__,
    )
    return frozenset()


def _extract_user_message(input: Any, input_key: str) -> str:
    if isinstance(input, str):
        return input
    if isinstance(input, dict):
        return str(input.get(input_key, ""))
    return str(input)


def _replace_user_message(input: Any, new_message: str, input_key: str) -> Any:
    if isinstance(input, str):
        return new_message
    if isinstance(input, dict):
        return {**input, input_key: new_message}
    return new_message
