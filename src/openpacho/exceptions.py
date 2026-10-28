"""Errors that distinguish missing hardware from unsupported semantics."""


class BackendUnavailableError(RuntimeError):
    """The requested runtime or device cannot be used."""


class UnsupportedDtypeError(TypeError):
    """A dtype is outside the supported set or the device's capabilities."""


class BackendMismatchError(ValueError):
    """An operation combines arrays from different backends/contexts."""
