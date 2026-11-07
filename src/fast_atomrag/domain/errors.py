"""Typed domain errors. Transport layers map them to protocol-specific responses."""


class DomainError(Exception):
    """Base class for all domain-level errors."""


class InvalidAtomError(DomainError):
    """An atom violates a domain invariant (id range, text length, entity limits)."""


class InvalidQueryError(DomainError):
    """A search request violates a domain invariant."""


class RoutingError(DomainError):
    """The router was configured with an inconsistent bundle catalog."""
