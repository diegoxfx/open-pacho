from __future__ import annotations
import typing
from openpacho.core.backend import OpenPachoBackend
from openpacho.core.backend_registry import register_backend
from openpacho.opencl_backend.opencl_context import OpenCLContextManager
import openpacho.opencl_backend.opencl_kernels as kernel
from openpacho.opencl_backend.opencl_array import OpenCLArray

if typing.TYPE_CHECKING:
    from openpacho.opencl_backend.opencl_array import OpenCLArray


class OpenCLBackend(OpenPachoBackend):
    def __init__(self):
        super().__init__("opencl")

        self.context_manager = OpenCLContextManager()

    def vector_dot(self, x: OpenCLArray, y: OpenCLArray):
        padded_shape = x._padded_shape
        out_arr = kernel.vector_dot_kernel(self.context_manager, x._cl_buffer, y._cl_buffer, shape=padded_shape)
        return out_arr.get()[0]
    
    def matrix_vec(self, x: OpenCLArray, y: OpenCLArray, return_vec=False):
        padded_shape = x._padded_shape
        shape = x._original_shape
        if return_vec:
            out_shape = (shape[0], )
            out_padded_shape = (padded_shape[0], )
        else:
            out_shape = (shape[0], 1)
            out_padded_shape = (padded_shape[0], 1)
        out_arr = kernel.matrix_vector_kernel(self.context_manager, x._cl_buffer, y._cl_buffer, shape=padded_shape)
        out_arr = OpenCLArray(out_arr, original_shape=out_shape, padded_shape=out_padded_shape)
        return out_arr

    def matrix_matrix(self, x: OpenCLArray, y: OpenCLArray):
        out_arr = kernel.matrix_matrix_kernel(self.context_manager, x._cl_buffer, y._cl_buffer, a_shape=x._padded_shape, b_shape=y._padded_shape)
        out_arr = OpenCLArray(out_arr, original_shape=(x.shape[0], y.shape[1]), padded_shape=(x._padded_shape[0], y._padded_shape[1]))
        return out_arr
    
    def transpose(self, x: OpenCLArray):
        out_arr = kernel.matrix_transpose_kernel(self.context_manager, x._cl_buffer, shape=x._padded_shape)
        out_arr = OpenCLArray(out_arr, original_shape=(x.shape[1], x.shape[0]), padded_shape=(x._padded_shape[1], x._padded_shape[0]))
        return out_arr