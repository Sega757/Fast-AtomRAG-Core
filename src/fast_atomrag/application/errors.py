"""Application-level errors (use-case outcomes, not invariant violations)."""


class ApplicationError(Exception):
    """Base class for application errors."""


class AtomNotFoundError(ApplicationError):
    def __init__(self, atom_id: int) -> None:
        super().__init__(f"atom {atom_id} not found")
        self.atom_id = atom_id


class ServiceNotReadyError(ApplicationError):
    """The in-memory index is not loaded, or was invalidated after an internal failure."""


class StorageError(ApplicationError):
    """The durable store rejected or failed an operation; nothing was changed in memory."""
