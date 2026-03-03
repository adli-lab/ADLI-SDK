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
_CREWAI_METHODS = frozenset({"kickoff", "kickoff_async"})
_LLAMAINDEX_QUERY_METHODS = frozenset({"query", "aquery"})
_LLAMAINDEX_CHAT_METHODS = frozenset({"chat", "achat"})


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


def _is_crewai_crew(obj: Any) -> bool:
    try:
        from crewai import Crew

        return isinstance(obj, Crew)
    except ImportError:
        return False


def _is_llamaindex_query_engine(obj: Any) -> bool:
    try:
        from llama_index.core import BaseQueryEngine

        return isinstance(obj, BaseQueryEngine)
    except ImportError:
        return False


def _is_llamaindex_chat_engine(obj: Any) -> bool:
    try:
        from llama_index.core.chat_engine.types import BaseChatEngine

        return isinstance(obj, BaseChatEngine)
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
        if _is_crewai_crew(self._wrapped):
            return self._crewai_interceptor(method, method_name)
        if _is_llamaindex_query_engine(self._wrapped) or _is_llamaindex_chat_engine(self._wrapped):
            return self._llamaindex_interceptor(method, method_name)
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


    # -- CrewAI ------------------------------------------------------------

    def _crewai_interceptor(self, method: Callable, method_name: str) -> Callable:
        client = self._client
        agent_name = self._agent_name
        project_id = self._project_id
        crew = self._wrapped

        def _attach(handler: Any) -> list[tuple[Any, list]]:
            """Prepend handler to each agent's LLM callbacks. Returns originals."""
            saved: list[tuple[Any, list]] = []
            for agent in getattr(crew, "agents", []):
                llm = getattr(agent, "llm", None)
                if llm is None or not hasattr(llm, "callbacks"):
                    continue
                orig = list(getattr(llm, "callbacks", None) or [])
                saved.append((llm, orig))
                llm.callbacks = [handler] + orig
            return saved

        def _detach(saved: list[tuple[Any, list]]) -> None:
            for llm, orig in saved:
                llm.callbacks = orig

        def _make_handler(inj: InjectResult, user_msg: str) -> Any:
            from adli_sdk.langchain_callback import ADLICallbackHandler

            return ADLICallbackHandler(
                client=client,
                project_id=project_id,
                agent_name=agent_name,
                adli_trace_id=inj.adli_trace_id,
                user_message=inj.message if inj.injected else user_msg,
                framework="crewai",
            )

        is_async = method_name == "kickoff_async"

        if is_async:

            @functools.wraps(method)
            async def async_wrapper(inputs: Any = None, *args: Any, **kwargs: Any) -> Any:
                user_msg = str(inputs or {})
                inj = await client.ainject(user_msg, agent_name)
                handler = _make_handler(inj, user_msg)
                saved = _attach(handler)
                outcome = "success"
                result = None
                try:
                    result = await method(inputs, *args, **kwargs)
                except Exception:
                    outcome = "failure"
                    raise
                finally:
                    _detach(saved)
                handler._outcome = outcome
                handler._flush({"output": str(result) if result is not None else ""})
                return result

            return async_wrapper

        @functools.wraps(method)
        def sync_wrapper(inputs: Any = None, *args: Any, **kwargs: Any) -> Any:
            user_msg = str(inputs or {})
            inj = client.inject(user_msg, agent_name)
            handler = _make_handler(inj, user_msg)
            saved = _attach(handler)
            outcome = "success"
            result = None
            try:
                result = method(inputs, *args, **kwargs)
            except Exception:
                outcome = "failure"
                raise
            finally:
                _detach(saved)
            handler._outcome = outcome
            handler._flush({"output": str(result) if result is not None else ""})
            return result

        return sync_wrapper

    # -- LlamaIndex --------------------------------------------------------

    def _llamaindex_interceptor(self, method: Callable, method_name: str) -> Callable:
        client = self._client
        agent_name = self._agent_name
        project_id = self._project_id
        engine = self._wrapped

        is_async = method_name.startswith("a")

        def _attach(handler: Any) -> None:
            cm = getattr(engine, "callback_manager", None)
            if cm is not None:
                cm.add_handler(handler, first=True)

        def _detach(handler: Any) -> None:
            cm = getattr(engine, "callback_manager", None)
            if cm is not None:
                try:
                    cm.remove_handler(handler)
                except Exception:
                    pass

        def _make_handler(inj: InjectResult, user_msg: str) -> Any:
            from adli_sdk.llama_index_callback import ADLILlamaIndexHandler

            return ADLILlamaIndexHandler(
                client=client,
                project_id=project_id,
                agent_name=agent_name,
                adli_trace_id=inj.adli_trace_id,
                user_message=inj.message if inj.injected else user_msg,
            )

        if is_async:

            @functools.wraps(method)
            async def async_wrapper(input_str: Any = None, *args: Any, **kwargs: Any) -> Any:
                user_msg = str(input_str or "")
                inj = await client.ainject(user_msg, agent_name)
                injected_input = inj.message if inj.injected else user_msg
                handler = _make_handler(inj, user_msg)
                _attach(handler)
                outcome = "success"
                try:
                    result = await method(injected_input, *args, **kwargs)
                except Exception:
                    outcome = "failure"
                    raise
                finally:
                    _detach(handler)
                handler._flush(_extract_llamaindex_output(result), outcome=outcome)
                return result

            return async_wrapper

        @functools.wraps(method)
        def sync_wrapper(input_str: Any = None, *args: Any, **kwargs: Any) -> Any:
            user_msg = str(input_str or "")
            inj = client.inject(user_msg, agent_name)
            injected_input = inj.message if inj.injected else user_msg
            handler = _make_handler(inj, user_msg)
            _attach(handler)
            outcome = "success"
            try:
                result = method(injected_input, *args, **kwargs)
            except Exception:
                outcome = "failure"
                raise
            finally:
                _detach(handler)
            handler._flush(_extract_llamaindex_output(result), outcome=outcome)
            return result

        return sync_wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_intercepts(wrapped: Any) -> frozenset[str]:
    if _is_pydantic_ai_agent(wrapped):
        return _PYDANTIC_AI_METHODS
    if _is_langchain_runnable(wrapped):
        return _LANGCHAIN_METHODS
    if _is_crewai_crew(wrapped):
        return _CREWAI_METHODS
    if _is_llamaindex_query_engine(wrapped):
        return _LLAMAINDEX_QUERY_METHODS
    if _is_llamaindex_chat_engine(wrapped):
        return _LLAMAINDEX_CHAT_METHODS
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


def _extract_llamaindex_output(result: Any) -> str:
    """Extract string output from a LlamaIndex query/chat response."""
    resp = getattr(result, "response", None)
    if resp:
        return str(resp)
    return str(result)
