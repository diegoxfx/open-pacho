import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from openpacho.core.array import OpenPachoArray
from openpacho.opencl_backend.opencl_context import OpenCLContextManager

class OpenCLArray(OpenPachoArray):
    """
    An array that resides in the GPU memory and uses OpenCL for computation.
    """

    def __init__(self, host_data=None, dtype=None):
        """
        host_data can be a list or a NumPy array.
        """
        # Convert incoming data to a NumPy array if it isn't one already
        if host_data is not None and not isinstance(host_data, np.ndarray | cl_array.Array | OpenCLArray):
            host_data = np.array(host_data, dtype=dtype if dtype else np.float32)
        
        # Infer shape and dtype if not provided
        if host_data is not None:
            shape = host_data.shape
            dtype = host_data.dtype
        else:
            shape = ()
            if dtype is None:
                dtype = np.float32  # default fallback

        super().__init__(shape=shape, dtype=dtype)

        # Acquire OpenCL context and queue
        self._cl_manager = OpenCLContextManager()
        self._context = self._cl_manager.get_context()
        self._queue = self._cl_manager.get_queue()

        # If 'data' is already an OpenCLArray, just reference its device data
        if isinstance(host_data, OpenCLArray):
            self._cl_buffer = host_data._cl_array
        # If 'data' is a pyopencl.array.Array, store it directly
        elif isinstance(host_data, cl_array.Array):
            self._cl_buffer = host_data
        # If host_data is given, allocate on GPU and copy data over
        elif host_data is not None and host_data.size > 0:
            self._cl_buffer = cl_array.to_device(self._queue, host_data)
        else:
            # Allocate an empty buffer if shape is known but no data provided
            size = np.prod(self.shape) if self.shape else 1
            empty_data = np.zeros(size, dtype=dtype)
            self._cl_buffer = cl_array.to_device(self._queue, empty_data)

    def to_host(self):
        """
        Returns a NumPy array copy of the GPU data.
        """
        return self._cl_buffer.get()

    def from_host(self, data):
        """
        Replaces the GPU data with new host data. Assumes shape/dtype match.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=self.dtype)
        if data.shape != self.shape:
            raise ValueError(f"Incompatible shapes: {data.shape} != {self.shape}")

        # Copy data to GPU buffer
        self._cl_buffer.set(data, queue=self._queue)

    def __add__(self, other):
        """Elementwise addition."""
        other_clarr = self._ensure_opencl_array(other)
        result = self._cl_buffer + other_clarr._cl_buffer
        return OpenCLArray(result)

    def __sub__(self, other):
        """Elementwise subtraction."""
        other_clarr = self._ensure_opencl_array(other)
        result = self._cl_buffer - other_clarr._cl_buffer
        return OpenCLArray(result)

    def __mul__(self, other):
        """Elementwise multiplication."""
        other_clarr = self._ensure_opencl_array(other)
        result = self._cl_buffer * other_clarr._cl_buffer
        return OpenCLArray(result)

    def __truediv__(self, other):
        """Elementwise division."""
        other_clarr = self._ensure_opencl_array(other)
        # Basic check to avoid division by zero; user must handle carefully otherwise
        result = self._cl_buffer / (other_clarr._cl_buffer + 1e-12)
        return OpenCLArray(result)

    def _ensure_opencl_array(self, other):
        """
        Utility to handle the right-hand operand for elementwise ops.
        If 'other' is not an OpenCLArray, convert it.
        """
        if isinstance(other, OpenCLArray):
            return other
        else:
            return OpenCLArray(other)

    def __len__(self):
        """
        Return the 'outer dimension' length for convenience.
        """
        if len(self.shape) == 0:
            return 0
        return self.shape[0]
