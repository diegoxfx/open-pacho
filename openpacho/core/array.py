from abc import ABC, abstractmethod
from openpacho.core.backend import OpenPachoBackend

class OpenPachoArray:
    """
    Base class for all OpenPacho array types.
    This class can define a common interface for array-like objects.
    """

    def __init__(self, shape, backend: OpenPachoBackend, dtype=None):
        self.shape = shape
        self.dtype = dtype
        self.backend = backend
        # Possibly store metadata (e.g. strides, data pointer references)
        # but leave it minimal for now.

    def to_host(self):
        """
        If implemented, returns a host-based representation (e.g. NumPy array).
        Must be overridden by child classes.
        """
        raise NotImplementedError("Must be overridden in subclass.")

    def from_host(self, data):
        """
        Optionally load data from host. Must be overridden by child classes.
        """
        raise NotImplementedError("Must be overridden in subclass.")

    def __add__(self, other):
        """Elementwise addition."""
        raise NotImplementedError("Must be overridden in subclass.")

    def __sub__(self, other):
        """Elementwise subtraction."""
        raise NotImplementedError("Must be overridden in subclass.")

    def __mul__(self, other):
        """Elementwise multiplication."""
        raise NotImplementedError("Must be overridden in subclass.")

    def __truediv__(self, other):
        """Elementwise division."""
        raise NotImplementedError("Must be overridden in subclass.")