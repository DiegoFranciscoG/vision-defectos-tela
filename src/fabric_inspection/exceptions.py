"""Domain exceptions. The API maps them to HTTP errors without leaking internals."""


class DomainError(Exception):
    """Base class of expected, user-facing errors."""


class NotFoundError(DomainError):
    """A requested resource does not exist."""


class BusinessRuleError(DomainError):
    """The request violates a business rule (e.g. an unsupported AQL)."""


class InvalidImageError(DomainError):
    """The uploaded file is not an acceptable image."""


class ImageTooLargeError(InvalidImageError):
    """The upload exceeds the size or pixel limit."""


class ServiceBusyError(DomainError):
    """All inference slots are taken; the client should retry later."""
