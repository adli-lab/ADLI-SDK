"""Transparent __getattr__ proxy that intercepts agent entry points for inject."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from adli_sdk.client import ADLIClient

logger = logging.getLogger("adli_sdk")

# Methods to intercept per framework type.
_PYDANTIC_AI_METHODS = frozenset({"run", "run_sync", "run_stream", "iter"})
_LANGCHAIN_METHODS = frozenset({"invoke", "ainvoke", "stream", "astream"})
_CREWAI_METHODS = frozenset({"kickoff", "kickoff_async"})
_LLAMAINDEX_QUERY_METHODS = frozenset({"query", "aquery"})
_LLAMAINDEX_CHAT_METHODS = frozenset({"chat", "achat"})


# ---------------------------------------------------------------------------
# Framework detection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ADLIWrapper
# ---------------------------------------------------------------------------


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

    # --- inject layer (dispatches to per-framework interceptor) -----------

    def _with_inject(self, method: Callable, method_name: str) -> Callable:
        """Return a thin wrapper that injects before delegating."""
        if _is_pydantic_ai_agent(self._wrapped):
            from adli_sdk.interceptors.pydantic_ai import make_interceptor
        elif _is_crewai_crew(self._wrapped):
            from adli_sdk.interceptors.crewai import make_interceptor
        elif (
            _is_llamaindex_query_engine(self._wrapped)
            or _is_llamaindex_chat_engine(self._wrapped)
        ):
            from adli_sdk.interceptors.llamaindex import make_interceptor
        else:
            from adli_sdk.interceptors.langchain import make_interceptor

        return make_interceptor(
            self._wrapped,
            method,
            method_name,
            self._client,
            self._agent_name,
            self._input_key,
            self._project_id,
        )


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
