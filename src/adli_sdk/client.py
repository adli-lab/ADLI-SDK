"""HTTP client for ADLI API (/inject, /learn)."""
from __future__ import annotations

import logging
import threading
import time

import httpx

from adli_sdk.models import InjectRequest, InjectResult, LearnRequest

logger = logging.getLogger("adli_sdk")

_DEFAULT_TIMEOUT = 10.0
_LEARN_MAX_RETRIES = 3
_LEARN_BACKOFF_BASE = 1.0


class ADLIClient:
    """Thin HTTP client wrapping ADLI /inject and /learn endpoints."""

    def __init__(self, *, token: str, project_id: int, base_url: str) -> None:
        self._token = token
        self._project_id = project_id
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # /inject
    # ------------------------------------------------------------------

    def inject(self, user_message: str, agent_name: str) -> InjectResult:
        """Synchronous inject call. Returns original message on failure."""
        payload = InjectRequest(
            user_message=user_message,
            agent_name=agent_name,
            project_id=self._project_id,
        )
        try:
            with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
                resp = client.post(
                    f"{self._base_url}/inject",
                    headers=self._headers,
                    content=payload.model_dump_json(),
                )
                resp.raise_for_status()
                data = resp.json()
                return InjectResult(
                    message=data.get("injected_message", user_message),
                    adli_trace_id=data.get("adli_trace_id", ""),
                    injected=data.get("injected", False),
                )
        except Exception:
            logger.warning("ADLI /inject unavailable — using original message", exc_info=True)
            return InjectResult(message=user_message, adli_trace_id="", injected=False)

    async def ainject(self, user_message: str, agent_name: str) -> InjectResult:
        """Async inject call. Returns original message on failure."""
        payload = InjectRequest(
            user_message=user_message,
            agent_name=agent_name,
            project_id=self._project_id,
        )
        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._base_url}/inject",
                    headers=self._headers,
                    content=payload.model_dump_json(),
                )
                resp.raise_for_status()
                data = resp.json()
                return InjectResult(
                    message=data.get("injected_message", user_message),
                    adli_trace_id=data.get("adli_trace_id", ""),
                    injected=data.get("injected", False),
                )
        except Exception:
            logger.warning("ADLI /inject unavailable — using original message", exc_info=True)
            return InjectResult(message=user_message, adli_trace_id="", injected=False)

    # ------------------------------------------------------------------
    # /learn
    # ------------------------------------------------------------------

    def learn(self, request: LearnRequest) -> None:
        """Send trace to /learn in a background thread (fire-and-forget)."""
        thread = threading.Thread(target=self._learn_sync, args=(request,), daemon=False)
        try:
            thread.start()
        except RuntimeError:
            # Interpreter shutdown: cannot create new threads. Fall back to sync call.
            self._learn_sync(request)

    def _learn_sync(self, request: LearnRequest) -> None:
        """Blocking /learn with retries. Runs in background thread."""
        for attempt in range(_LEARN_MAX_RETRIES):
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        f"{self._base_url}/learn",
                        headers=self._headers,
                        content=request.model_dump_json(),
                    )
                    resp.raise_for_status()
                    return
            except Exception:
                wait = _LEARN_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "ADLI /learn attempt %d/%d failed, retrying in %.1fs",
                    attempt + 1,
                    _LEARN_MAX_RETRIES,
                    wait,
                    exc_info=True,
                )
                time.sleep(wait)
        logger.error("ADLI /learn failed after %d attempts — trace lost", _LEARN_MAX_RETRIES)
