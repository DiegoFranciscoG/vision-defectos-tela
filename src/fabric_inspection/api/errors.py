"""Exception handlers: stable JSON errors, never stack traces (OWASP API8)."""

import logging
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fabric_inspection.exceptions import (
    BusinessRuleError,
    DomainError,
    ImageTooLargeError,
    InvalidImageError,
    NotFoundError,
    ServiceBusyError,
)

logger = logging.getLogger(__name__)

_STATUS: tuple[tuple[type[DomainError], int], ...] = (
    (NotFoundError, status.HTTP_404_NOT_FOUND),
    (ImageTooLargeError, status.HTTP_413_CONTENT_TOO_LARGE),
    (InvalidImageError, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
    (BusinessRuleError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (ServiceBusyError, status.HTTP_503_SERVICE_UNAVAILABLE),
)


async def _domain_error(_: Request, error: Exception) -> JSONResponse:
    code = next(
        (http for kind, http in _STATUS if isinstance(error, kind)), status.HTTP_400_BAD_REQUEST
    )
    headers = {"Retry-After": "5"} if isinstance(error, ServiceBusyError) else None
    return JSONResponse({"detail": str(error)}, status_code=code, headers=headers)


async def _validation_error(_: Request, error: Exception) -> JSONResponse:
    # Echo field locations and messages, not the submitted values.
    assert isinstance(error, RequestValidationError)  # noqa: S101 - registered for this type only
    details = [
        {"loc": list(item.get("loc", ())), "msg": item.get("msg", "")} for item in error.errors()
    ]
    return JSONResponse(
        {"detail": "Invalid request", "errors": details},
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


async def _unexpected_error(request: Request, error: Exception) -> JSONResponse:
    error_id = uuid.uuid4().hex[:12]
    logger.error(
        "Unhandled error %s on %s %s: %s",
        error_id,
        request.method,
        request.url.path,
        type(error).__name__,
        exc_info=error,
    )
    return JSONResponse(
        {"detail": "Internal server error", "error_id": error_id},
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, _domain_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected_error)
