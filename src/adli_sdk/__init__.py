"""ADLI SDK — runtime strategy learning for AI agents.

Usage::

    from adli_sdk import ADLI

    adli = ADLI(token="...", project_id=1)
    adli.instrument()

    agent = adli.wrap(Agent(...), agent_name="sql-agent")
    result = await agent.run("user query")
"""
from __future__ import annotations

import logging
from typing import Any

from adli_sdk.client import ADLIClient
from adli_sdk.models import InjectResult
from adli_sdk.wrapper import ADLIWrapper

logger = logging.getLogger("adli_sdk")

_DEFAULT_BASE_URL = "https://api.adli.dev"

# Re-export for manual LangChain usage:  from adli_sdk import ADLICallbackHandler
try:
    from adli_sdk.langchain_callback import ADLICallbackHandler  # noqa: F401
except ImportError:
    pass


class ADLI:
    """Main entry point for the ADLI SDK.

    Parameters
    ----------
    token : str
        ADLI API token.
    project_id : int
        ADLI project identifier.
    base_url : str
        Base URL of the ADLI API (default: ``https://api.adli.dev``).
    """

    def __init__(
        self,
        *,
        token: str,
        project_id: int,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._client = ADLIClient(token=token, project_id=project_id, base_url=base_url)
        self._project_id = project_id
        self._instrumented = False

    # ------------------------------------------------------------------
    # instrument — plug into OTel pipeline
    # ------------------------------------------------------------------

    def instrument(self) -> None:
        """Add :class:`ADLISpanProcessor` to the global OTel ``TracerProvider``.

        Safe to call multiple times — only instruments once.
        """
        if self._instrumented:
            return

        from adli_sdk.processor import ADLISpanProcessor

        processor = ADLISpanProcessor(client=self._client, project_id=self._project_id)

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider

            provider = trace.get_tracer_provider()

            # Unwrap proxy providers (e.g. Logfire wraps the real TracerProvider)
            real_provider = getattr(provider, "_real_tracer_provider", None) or provider
            if hasattr(real_provider, "add_span_processor"):
                real_provider.add_span_processor(processor)
                logger.info("ADLISpanProcessor added to existing TracerProvider")
            else:
                new_provider = TracerProvider()
                new_provider.add_span_processor(processor)
                trace.set_tracer_provider(new_provider)
                logger.info("Created new TracerProvider with ADLISpanProcessor")
        except ImportError:
            logger.warning("opentelemetry-sdk not installed — trace collection disabled")
            return

        self._instrumented = True

    # ------------------------------------------------------------------
    # wrap — transparent proxy for inject
    # ------------------------------------------------------------------

    def wrap(self, obj: Any, *, agent_name: str, input_key: str = "input") -> Any:
        """Wrap an agent or chain with automatic inject.

        The returned object behaves identically to *obj* — all attributes and
        methods are delegated transparently.  Only the entry-point methods
        (``run`` / ``invoke`` etc.) are intercepted to call ``/inject`` and
        attach ``adli_trace_id`` to metadata.

        Parameters
        ----------
        obj
            A PydanticAI ``Agent`` or LangChain ``Runnable``.
        agent_name
            Logical name of this agent in ADLI.
        input_key
            For LangChain dict inputs: which key holds the user message
            (default ``"input"``).
        """
        return ADLIWrapper(
            obj,
            client=self._client,
            agent_name=agent_name,
            input_key=input_key,
            project_id=self._project_id,
        )

    # ------------------------------------------------------------------
    # inject — manual mode
    # ------------------------------------------------------------------

    def inject(self, user_message: str, *, agent_name: str) -> InjectResult:
        """Call ``/inject`` synchronously (for manual mode without wrapper)."""
        return self._client.inject(user_message, agent_name)

    async def ainject(self, user_message: str, *, agent_name: str) -> InjectResult:
        """Call ``/inject`` asynchronously (for manual mode without wrapper)."""
        return await self._client.ainject(user_message, agent_name)

    # ------------------------------------------------------------------
    # langchain_callback — manual mode for LangChain
    # ------------------------------------------------------------------

    def langchain_callback(
        self,
        *,
        agent_name: str,
        adli_trace_id: str,
        user_message: str,
    ) -> Any:
        """Create an :class:`ADLICallbackHandler` for manual LangChain usage.

        Use when you want full control over inject and chain invocation::

            inj = adli.inject("query", agent_name="my-agent")
            handler = adli.langchain_callback(
                agent_name="my-agent",
                adli_trace_id=inj.adli_trace_id,
                user_message=inj.message,
            )
            result = chain.invoke(inj.message, config={"callbacks": [handler]})
        """
        from adli_sdk.langchain_callback import ADLICallbackHandler

        return ADLICallbackHandler(
            client=self._client,
            project_id=self._project_id,
            agent_name=agent_name,
            adli_trace_id=adli_trace_id,
            user_message=user_message,
        )


__all__ = ["ADLI", "InjectResult"]
