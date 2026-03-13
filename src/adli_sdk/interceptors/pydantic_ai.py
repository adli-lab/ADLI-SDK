"""PydanticAI interceptor — run, run_sync, run_stream, iter."""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from adli_sdk.client import ADLIClient
from adli_sdk.interceptors._base import build_adli_metadata, do_inject_async, do_inject_sync


def make_interceptor(
    wrapped: Any,
    method: Callable,
    method_name: str,
    client: ADLIClient,
    agent_name: str,
    input_key: str,
    project_id: int,
) -> Callable:
    """Return an intercepted version of a PydanticAI ``Agent`` method."""
    if method_name == "run_sync":
        return _sync_interceptor(method, client, agent_name)
    if method_name == "iter":
        return _iter_interceptor(method, client, agent_name)
    return _async_interceptor(method, client, agent_name)


# ------------------------------------------------------------------
# Sync (run_sync)
# ------------------------------------------------------------------


def _sync_interceptor(method: Callable, client: ADLIClient, agent_name: str) -> Callable:
    @functools.wraps(method)
    def wrapper(user_prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
        if user_prompt is not None and isinstance(user_prompt, str):
            injected, original, inj = do_inject_sync(client, user_prompt, agent_name)
            user_prompt = injected
            meta = kwargs.get("metadata") or {}
            meta.update(build_adli_metadata(inj, original, agent_name))
            kwargs["metadata"] = meta
        return method(user_prompt, *args, **kwargs)

    return wrapper


# ------------------------------------------------------------------
# Async (run, run_stream)
# ------------------------------------------------------------------


def _async_interceptor(method: Callable, client: ADLIClient, agent_name: str) -> Callable:
    @functools.wraps(method)
    async def wrapper(user_prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
        if user_prompt is not None and isinstance(user_prompt, str):
            injected, original, inj = await do_inject_async(client, user_prompt, agent_name)
            user_prompt = injected
            meta = kwargs.get("metadata") or {}
            meta.update(build_adli_metadata(inj, original, agent_name))
            kwargs["metadata"] = meta
        return await method(user_prompt, *args, **kwargs)

    return wrapper


# ------------------------------------------------------------------
# Iter (async context manager)
# ------------------------------------------------------------------


def _iter_interceptor(method: Callable, client: ADLIClient, agent_name: str) -> Callable:
    @functools.wraps(method)
    def wrapper(user_prompt: Any = None, *args: Any, **kwargs: Any) -> Any:
        return _IterContextWrapper(client, agent_name, method, user_prompt, args, kwargs)

    return wrapper


class _IterContextWrapper:
    """Async context manager that injects the ADLI prompt before entering iter().

    pydantic-ai's ``Agent.iter()`` is a synchronous method returning an async
    context manager.  We wrap it to perform inject inside ``__aenter__`` and
    then delegate to the real context manager.
    """

    def __init__(
        self,
        client: ADLIClient,
        agent_name: str,
        method: Callable,
        user_prompt: Any,
        args: tuple,
        kwargs: dict,
    ) -> None:
        self._client = client
        self._agent_name = agent_name
        self._method = method
        self._user_prompt = user_prompt
        self._args = args
        self._kwargs = kwargs
        self._cm: Any = None

    async def __aenter__(self) -> Any:
        user_prompt = self._user_prompt
        kwargs = dict(self._kwargs)
        if user_prompt is not None and isinstance(user_prompt, str):
            injected, original, inj = await do_inject_async(
                self._client, user_prompt, self._agent_name
            )
            user_prompt = injected
            meta = kwargs.get("metadata") or {}
            meta.update(build_adli_metadata(inj, original, self._agent_name))
            kwargs["metadata"] = meta
        self._cm = self._method(user_prompt, *self._args, **kwargs)
        return await self._cm.__aenter__()

    async def __aexit__(self, *exc_info: Any) -> Any:
        if self._cm is not None:
            return await self._cm.__aexit__(*exc_info)
        return False
