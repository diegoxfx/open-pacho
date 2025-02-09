from typing import Tuple
import pyopencl as cl
import pyopencl.array as cl_array
import openpacho.opencl_backend.opencl_utils as cl_utils
from openpacho.opencl_backend.opencl_context import OpenCLContextManager
import numpy as np
import os

def _load_kernel_source(filename):
    # Helper function to load the .cl file contents
    base_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_path, "opencl_kernels", filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def vector_dot_kernel(context_manager: OpenCLContextManager, x_arr: cl_array.Array, y_arr: cl_array.Array, shape: Tuple[int, int]):
    # Load the kernel from vector_dot.cl
    group_size = cl_utils.pick_optimal_group_size_vector_dot(shape[0])
    kernel_src = _load_kernel_source("vector_dot.cl")
    prg = cl.Program(context_manager.context, kernel_src).build(options=[f"-DWG_SIZE={group_size}"])

    out_buf = cl_array.zeros(context_manager.queue, (1,), dtype=np.float32).data
    kernel = prg.dot_product_f32_vec4

    global_size = (shape[0], )
    local_size = (group_size, )

    kernel(context_manager.queue, global_size, local_size, x_arr.data, y_arr.data, out_buf, np.int32(shape[0]))
    context_manager.queue.finish()

    return cl_array.Array(context_manager.queue, (1,), dtype=np.float32, data=out_buf)

def matrix_vector_kernel(context_manager: OpenCLContextManager, mat_arr: cl_array, vec_arr: cl_array, shape: Tuple[int, int]):
    # Load from matrix_vector.cl
    group_size = cl_utils.pick_optimal_group_size_vector_dot(shape[0])
    kernel_src = _load_kernel_source("matrix_vector.cl")
    prg = cl.Program(context_manager.context, kernel_src).build()

    out_arr = cl_array.zeros(context_manager.queue, (shape[0],), dtype=np.float32)
    kernel = prg.matvec_f32_vec4

    global_size = (shape[0],)
    local_size = (group_size,)

    kernel(context_manager.queue, global_size, local_size, mat_arr.data, vec_arr.data, out_arr.data, np.int32(shape[1]), np.int32(shape[0]))
    context_manager.queue.finish()

    return out_arr

def matrix_matrix_kernel(context_manager: OpenCLContextManager, a_mat: cl_array, b_mat: cl_array, a_shape: Tuple[int, int], b_shape: Tuple[int, int]):
    # Load from matrix_matrix.cl
    group_size = context_manager.get_preferred_work_group_size_multiple()
    kernel_src = _load_kernel_source("matrix_matrix.cl")
    prg = cl.Program(context_manager.context, kernel_src).build(options=[f"-DTILE_SIZE={group_size}"])

    out_arr = cl_array.zeros(context_manager.queue, (a_shape[0], b_shape[1]), dtype=np.float32)
    kernel = prg.matmul_f32_tiled

    # Corrected: global_size should be (N, M) where N = b_shape[1] and M = a_shape[0]
    global_size = (a_shape[0], b_shape[1])
    local_size = (group_size, group_size)

    kernel(context_manager.queue, global_size, local_size, 
           a_mat.data, b_mat.data, out_arr.data, 
           np.int32(a_shape[0]), np.int32(a_shape[1]), np.int32(b_shape[1]))
    context_manager.queue.finish()

    return out_arr

def matrix_transpose_kernel(context_manager: OpenCLContextManager,
                            input_arr: cl_array.Array,
                            shape: Tuple[int, int]) -> cl_array.Array:
    group_size = context_manager.get_preferred_work_group_size_multiple()
    kernel_src = _load_kernel_source("transpose.cl")
    prg = cl.Program(context_manager.context, kernel_src).build(options=[f"-DTILE_DIM={group_size}"])
    
    out_arr = cl_array.zeros(context_manager.queue, (shape[1], shape[0]), dtype=np.float32)
    kernel = prg.mat_transpose_f32_4
    
    global_size = (shape[0], shape[1])
    local_size = (group_size, group_size)
    
    kernel(context_manager.queue,
           global_size,
           local_size,
           input_arr.data,
           out_arr.data,
           np.int32(shape[0]),
           np.int32(shape[1]))
    context_manager.queue.finish()
    
    return out_arr
