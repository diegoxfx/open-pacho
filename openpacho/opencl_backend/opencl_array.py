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
        
        # Infer shape and dtype if not explicitly provided
        if host_data is not None:
            # If host_data is an OpenCLArray, we copy its shape/dtype
            if isinstance(host_data, OpenCLArray):
                shape = host_data.shape
                dtype = host_data.dtype
            # If host_data is a pyopencl Array
            elif isinstance(host_data, cl_array.Array):
                shape = host_data.shape
                dtype = host_data.dtype
            else:
                shape = host_data.shape
                dtype = host_data.dtype
        else:
            shape = ()
            dtype = dtype if dtype else np.float32  # default fallback

        super().__init__(shape=shape, dtype=dtype)

        # Acquire OpenCL context and queue
        self._cl_manager = OpenCLContextManager()
        self._context = self._cl_manager.get_context()
        self._queue = self._cl_manager.get_queue()

        # Metadata for padding
        self._actual_size = int(np.prod(self.shape))  # real number of elements
        self._padded_size = self._round_up_to_multiple_of_8(self._actual_size)

        # Handle different types of host_data
        if isinstance(host_data, OpenCLArray):
            # If the user passed an existing OpenCLArray,
            # we just reference its cl_array, plus copy metadata.
            self._cl_buffer = host_data._cl_buffer
            self._actual_size = host_data._actual_size
            self._padded_size = host_data._padded_size

        elif isinstance(host_data, cl_array.Array):
            # If user passed a pyopencl array, we assume it is already sized appropriately
            # or we might want to copy it into a newly padded buffer.
            # For demonstration, let's simply reference it if it's sized for our shape:
            if host_data.size < self._padded_size:
                # We need to re-allocate a bigger buffer
                host_data = self._pad_and_wrap_cl_array(host_data)
            self._cl_buffer = host_data

        elif host_data is not None and host_data.size > 0:
            # Convert host_data to a padded 1D array (flattened)
            padded_host_data = self._pad_host_data(host_data)
            # Send to GPU
            self._cl_buffer = cl_array.to_device(self._queue, padded_host_data)
        else:
            # No data or shape given; allocate a zeroed buffer (with padding)
            empty_data = np.zeros(self._padded_size, dtype=dtype)
            self._cl_buffer = cl_array.to_device(self._queue, empty_data)

    def _round_up_to_multiple_of_8(self, n):
        return ((n + 7) // 8) * 8

    def _pad_host_data(self, data: np.ndarray) -> np.ndarray:
        """
        Flattens the incoming data and pads it to self._padded_size if needed.
        """
        flattened = data.ravel()  # flatten in row-major order
        if flattened.size == self._padded_size:
            return flattened
        # Otherwise, create a new array with enough space
        new_data = np.zeros(self._padded_size, dtype=self.dtype)
        new_data[:flattened.size] = flattened
        return new_data
    
    def _pad_and_wrap_cl_array(self, cl_arr: cl_array.Array):
        """
        If the existing cl_array is too small (or not padded),
        allocate a new buffer with self._padded_size, copy old data.
        """
        new_buf = cl_array.zeros(self._queue, (self._padded_size,), dtype=self.dtype)
        # Copy the existing data over
        cl.enqueue_copy(self._queue, dest=new_buf.data, src=cl_arr.data, byte_count=cl_arr.nbytes)
        return new_buf

    def to_host(self) -> np.ndarray:
        """
        Returns a NumPy array copy of the GPU data, reshaping to the original shape
        and ignoring padded elements.
        """
        padded_result = self._cl_buffer.get()  # length = self._padded_size
        # Slice to the actual size, then reshape
        real_data = padded_result[:self._actual_size]
        return real_data.reshape(self.shape)

    def from_host(self, data):
        """
        Replaces the GPU data with new host data. Must match shape/dtype (un-padded).
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=self.dtype)

        if data.shape != self.shape:
            raise ValueError(f"Incompatible shapes: {data.shape} != {self.shape}")

        # Flatten and pad
        padded_host_data = self._pad_host_data(data)
        # Copy to GPU buffer
        if padded_host_data.size != self._cl_buffer.size:
            # Need to re-create the buffer
            self._cl_buffer = cl_array.to_device(self._queue, padded_host_data)
        else:
            self._cl_buffer.set(padded_host_data, queue=self._queue)

    def __add__(self, other):
        """Elementwise addition, ignoring padding logic in the user-level API."""
        other_cl = self._ensure_opencl_array(other)
        # Perform direct elementwise on full buffers
        result_buf = self._cl_buffer + other_cl._cl_buffer
        # Create a new OpenCLArray from the result,
        # reusing self's shape/dtype => ensures same padding logic
        res_array = OpenCLArray()
        res_array.shape = self.shape
        res_array.dtype = self.dtype
        res_array._actual_size = self._actual_size
        res_array._padded_size = self._padded_size
        res_array._cl_buffer = result_buf
        return res_array

    def __sub__(self, other):
        other_cl = self._ensure_opencl_array(other)
        result_buf = self._cl_buffer - other_cl._cl_buffer
        res_array = OpenCLArray()
        res_array.shape = self.shape
        res_array.dtype = self.dtype
        res_array._actual_size = self._actual_size
        res_array._padded_size = self._padded_size
        res_array._cl_buffer = result_buf
        return res_array

    def __mul__(self, other):
        other_cl = self._ensure_opencl_array(other)
        result_buf = self._cl_buffer * other_cl._cl_buffer
        res_array = OpenCLArray()
        res_array.shape = self.shape
        res_array.dtype = self.dtype
        res_array._actual_size = self._actual_size
        res_array._padded_size = self._padded_size
        res_array._cl_buffer = result_buf
        return res_array

    def __truediv__(self, other):
        other_cl = self._ensure_opencl_array(other)
        result_buf = self._cl_buffer / (other_cl._cl_buffer + 1e-12)
        res_array = OpenCLArray()
        res_array.shape = self.shape
        res_array.dtype = self.dtype
        res_array._actual_size = self._actual_size
        res_array._padded_size = self._padded_size
        res_array._cl_buffer = result_buf
        return res_array

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
