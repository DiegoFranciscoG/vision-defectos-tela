"""FastAPI dependencies: settings, DB session, inspector, authentication and rate limiting."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request, Security
from sqlalchemy.orm import Session, sessionmaker

from fabric_inspection.api.security import (
    ApiKeyVerifier,
    RateLimiter,
    api_key_header,
    client_bucket,
)
from fabric_inspection.config import ApiSettings
from fabric_inspection.exceptions import ServiceBusyError
from fabric_inspection.inference.inspector import FabricInspector


@dataclass
class AppState:
    settings: ApiSettings
    session_factory: sessionmaker[Session]
    inspector: FabricInspector
    verifier: ApiKeyVerifier
    key_limiter: RateLimiter
    ip_limiter: RateLimiter
    inference_slots: asyncio.Semaphore


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.services
    return state


State = Annotated[AppState, Depends(get_state)]


def get_session(state: State) -> Iterator[Session]:
    session = state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


DbSession = Annotated[Session, Depends(get_session)]


def require_api_key(
    request: Request,
    state: State,
    api_key: Annotated[str | None, Security(api_key_header)],
) -> str:
    """Deny by default: every /api route depends on this. Unauthenticated attempts are also
    rate limited per client IP, so keys cannot be brute-forced."""
    try:
        fingerprint = state.verifier.verify(api_key)
    except Exception:
        state.ip_limiter.check(client_bucket(request))
        raise
    state.key_limiter.check(fingerprint)
    return fingerprint


ApiKey = Annotated[str, Depends(require_api_key)]


@asynccontextmanager
async def inference_slot(state: AppState) -> AsyncIterator[None]:
    """Bound concurrent inferences (RAM on Render free is 512 MB)."""
    try:
        await asyncio.wait_for(
            state.inference_slots.acquire(), timeout=state.settings.inference_queue_timeout_s
        )
    except TimeoutError as error:
        raise ServiceBusyError("The inference service is busy, retry in a few seconds") from error
    try:
        yield
    finally:
        state.inference_slots.release()
