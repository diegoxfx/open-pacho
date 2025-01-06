class OpenPachoArray:
    """
    Base class for all OpenPacho array types.
    This class can define a common interface for array-like objects.
    """

    def __init__(self, shape, dtype=None):
        self.shape = shape
        self.dtype = dtype
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
