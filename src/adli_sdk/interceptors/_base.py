"""Shared inject helpers used by all framework interceptors."""
from __future__ import annotations

from adli_sdk.client import ADLIClient
from adli_sdk.models import InjectResult


def do_inject_sync(
    client: ADLIClient, user_prompt: str, agent_name: str
) -> tuple[str, str, InjectResult]:
    """Synchronous inject. Returns ``(injected_prompt, original_prompt, inject_result)``."""
    inj = client.inject(user_prompt, agent_name)
    return inj.message, user_prompt, inj


async def do_inject_async(
    client: ADLIClient, user_prompt: str, agent_name: str
) -> tuple[str, str, InjectResult]:
    """Async inject. Returns ``(injected_prompt, original_prompt, inject_result)``."""
    inj = await client.ainject(user_prompt, agent_name)
    return inj.message, user_prompt, inj


def build_adli_metadata(
    inj: InjectResult, original_prompt: str, agent_name: str
) -> dict[str, str]:
    """Build the metadata dict with ADLI trace fields."""
    return {
        "adli_trace_id": inj.adli_trace_id,
        "adli_agent_name": agent_name,
        "adli_user_message": inj.message if inj.injected else original_prompt,
    }
