import numpy as np
import pyopencl.array as cl_array

from openpacho.core.array import OpenPachoArray
from openpacho.core.backend_registry import get_backend
import openpacho as op


class OpenCLArray(OpenPachoArray):
    """
    An array that resides in GPU memory and uses OpenCL for computation.
    It applies transparent padding to ensure each dimension meets the rules above.
    """

    def __init__(self, host_data=None, dtype=None, original_shape=None, padded_shape=None):
        """
        host_data can be a Python list, NumPy array, OpenCLArray, or pyopencl.array.Array.
        """
        # If initializing from an existing OpenCLArray, copy attributes accordingly
        if isinstance(host_data, OpenCLArray):
            self._cl_buffer = host_data.cl_buffer
            self._original_shape = host_data.original_shape
            self._padded_shape = host_data.padded_shape
            self.shape = host_data._padded_shape
            self.dtype = host_data.dtype
            self.backend = host_data.backend
            return  # Initialization complete
        
        # Convert to NumPy if not already
        if host_data is not None and not isinstance(host_data, (np.ndarray, cl_array.Array)):
            host_data = np.array(host_data, dtype=dtype if dtype else np.float32)

        # Infer shape/dtype
        if host_data is not None:
            if isinstance(host_data, cl_array.Array):
                shape = original_shape if original_shape is not None else host_data.shape
                dtype = host_data.dtype

                self._cl_buffer = host_data
            else:
                shape = host_data.shape
                dtype = host_data.dtype

                self._original_shape = shape
        else:
            shape = ()
            dtype = dtype if dtype else np.float32

        # Raise error if shape dim is greater than 2. Only 1D and 2D arrays are supported yet
        if host_data is not None and len(host_data.shape) > 2:
            raise NotImplementedError("Only 1D and 2D arrays are supported.")
        backend = get_backend("opencl")
        super().__init__(shape=shape, dtype=dtype, backend=backend)

        # Raise error if original shape not provided and if _padded_shape not a multiple of preferred_multiple
        if isinstance(host_data, cl_array.Array):
            if original_shape is None:
                preferred_multiple = self.backend.context_manager.get_preferred_work_group_size_multiple()
                if padded_shape is None:
                    padded_shape = shape
                if any(dim % preferred_multiple != 0 for dim in padded_shape):
                    raise ValueError(
                        "Padded shape must be a multiple of the preferred work group size multiple."
                    )
                # TODO: Implement padding logic for cl_array if not already padded
            else:
                self._original_shape = original_shape
                self._padded_shape = padded_shape
        elif isinstance(host_data, np.ndarray):
            self._padded_shape = self._calculate_padded_shape(shape)
            padded_data = self._pad_host_data(host_data, self._original_shape, self._padded_shape)
            self._cl_buffer = cl_array.to_device(self.backend.context_manager.queue, padded_data)

    def _pad_dimension(self, dim):
        """
        Pad a single dimension to the nearest multiple of preferred work group size multiple.
        """
        preferred_multiple = self.backend.context_manager.get_preferred_work_group_size_multiple()
        return ((dim + preferred_multiple - 1) // preferred_multiple) * preferred_multiple

    def _calculate_padded_shape(self, shape):
        """
        Calculate the padded shape.
        """
        return tuple(map(self._pad_dimension, shape))
    
    def _pad_host_data(self, host_data, original_shape, padded_shape):
        """
        Pad the host data with zeros based on the original and padded shapes.
        Supports 1D and 2D arrays.
        """
        slices = tuple(slice(0, n) for n in original_shape)
        if host_data is None:
            # If no data provided, return a zero-filled array with padded shape
            return np.zeros(padded_shape, dtype=self.dtype)
        
        padded_data = np.zeros(padded_shape, dtype=self.dtype)
        padded_data[slices] = host_data[slices]
        return padded_data

    def to_host(self) -> np.ndarray:
        """
        Returns a NumPy array of shape 'self._original_shape', discarding padded elements.
        """
        slices = tuple(slice(0, n) for n in self._original_shape)
        padded_result = self._cl_buffer.get().reshape(self._padded_shape)
        real_data = padded_result[slices]
        return real_data.copy()

    def from_host(self, data):
        """
        Updates the GPU buffer from a new host array matching 'self._actual_shape'.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=self.dtype)

        shape = data.shape
        self._original_shape = shape

        self._padded_shape = self._calculate_padded_shape(shape)
        padded_data = self._pad_host_data(data, self._original_shape, self._padded_shape)
        self._cl_buffer = cl_array.to_device(self.backend.context_manager.queue, padded_data)

    def _is_scalar(self, value):
        """
        Check if the value is a scalar (int, float, or NumPy scalar).
        """
        return np.isscalar(value)

    def _prepare_operand(self, other):
        """
        Prepare the operand for arithmetic operations.
        
        - If 'other' is a scalar, return it directly.
        - If 'other' is an OpenCLArray, ensure shapes match and return its buffer.
        - Otherwise, convert 'other' to an OpenCLArray and ensure shapes match, then return its buffer.
        """
        if self._is_scalar(other):
            return other
        else:
            other_cl = self._ensure_opencl_array(other)
            if self._padded_shape != other_cl._padded_shape:
                raise ValueError("Shapes of both OpenCLArrays must match for this operation.")
            return other_cl._cl_buffer

    def __add__(self, other):
        operand = self._prepare_operand(other)
        result_buf = self._cl_buffer + operand
        return OpenCLArray(
            host_data=result_buf,
            original_shape=self._original_shape,
            padded_shape=self._padded_shape,
        )

    def __sub__(self, other):
        operand = self._prepare_operand(other)
        result_buf = self._cl_buffer - operand
        return OpenCLArray(
            host_data=result_buf,
            original_shape=self._original_shape,
            padded_shape=self._padded_shape,
        )
    
    def __rsub__(self, other):
        operand = self._prepare_operand(other)
        result_buf = operand - self._cl_buffer
        return OpenCLArray(
            host_data=result_buf,
            original_shape=self._original_shape,
            padded_shape=self._padded_shape,
        )

    def __mul__(self, other):
        operand = self._prepare_operand(other)
        result_buf = self._cl_buffer * operand
        return OpenCLArray(
            host_data=result_buf,
            original_shape=self._original_shape,
            padded_shape=self._padded_shape,
        )
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        operand = self._prepare_operand(other)
        result_buf = self._cl_buffer / (operand + 1e-12)
        return OpenCLArray(
            host_data=result_buf,
            original_shape=self._original_shape,
            padded_shape=self._padded_shape,
        )
    
    def __matmul__(self, other):
        operand = self._ensure_opencl_array(other)
        return op.linalg.dot(self, operand)

    def _ensure_opencl_array(self, other):
        """
        Convert 'other' to an OpenCLArray if needed, using the same dtype as self.
        """
        if isinstance(other, OpenCLArray):
            return other
        else:
            return OpenCLArray(other, dtype=self.dtype)
        
    def _extract_scalar(self, result):
        result = result.to_host()
        if result.size == 1:
            return result.item()
        return result

    def __len__(self):
        """If it's 1D or 2D, return the outer dimension (like NumPy does for 2D)."""
        return self.shape[0]
    
    def transpose(self):
        return op.transpose(self)
    
    def reshape(self, new_shape):
        """
        Reshape the current OpenCLArray to 'new_shape'.

        If total elements match and either the old or new shape
        is effectively 1D (including (N,1) or (1,N)), we do a "metadata-only" reshape.
        Otherwise, we do a device->device copy with a new buffer.
        """
        # Normalize to tuple
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        if len(new_shape) > 2:
            raise NotImplementedError("Only 1D or 2D reshapes are supported.")

        # 1) Check total elements
        old_size = np.prod(self._original_shape)
        new_size = np.prod(new_shape)
        if new_size != old_size:
            raise ValueError(
                f"Cannot reshape array of size {old_size} into shape {new_shape} "
                f"(size {new_size})."
            )

        # 2) If shapes are identical, nothing to do
        if new_shape == self._original_shape:
            return self

        # 3) Check if we can do a "metadata-only" reshape
        #    Condition: If old or new shape is effectively 1D
        #    i.e. shape is (N,) or (N,1) or (1,N).
        def is_effectively_1d(shape):
            """Returns True if shape is (N,), (N,1), or (1,N)."""
            # shape could be 1D or 2D
            if len(shape) == 1:
                return True
            elif len(shape) == 2:
                (r, c) = shape
                return (r == 1 or c == 1)
            return False

        can_metadata_reshape = (
            is_effectively_1d(self._original_shape) or
            is_effectively_1d(new_shape)
        )

        if can_metadata_reshape:
            # Also check that we fit within the allocated padded capacity
            if new_size <= np.prod(self._padded_shape):
                # => METADATA-ONLY
                self._original_shape = new_shape
                return self
            # If it doesn't fit in the padded shape (very rare in 1D?), we do a copy below.

        # 4) Otherwise, do device->device copy to re-layout physically
        return self._reshape_device_copy(new_shape)