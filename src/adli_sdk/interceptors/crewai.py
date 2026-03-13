"""CrewAI interceptor — kickoff, kickoff_async."""
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
    """Return an intercepted version of a CrewAI ``Crew`` method."""
    crew = wrapped

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
