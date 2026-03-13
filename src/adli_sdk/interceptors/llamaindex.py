"""LlamaIndex interceptor — query, aquery, chat, achat."""
from __future__ import annotations

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
    """Return an intercepted version of a LlamaIndex engine method."""
    engine = wrapped
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
            handler._flush(_extract_output(result), outcome=outcome)
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
        handler._flush(_extract_output(result), outcome=outcome)
        return result

    return sync_wrapper


def _extract_output(result: Any) -> str:
    """Extract string output from a LlamaIndex query/chat response."""
    resp = getattr(result, "response", None)
    if resp:
        return str(resp)
    return str(result)
