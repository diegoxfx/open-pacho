// Assume TILE_DIM is defined as CL_KERNEL_PREFERRED_WORK_GROUP_SIZE (e.g., 16)
// and that the matrix dimensions are multiples of TILE_DIM.
#ifndef TILE_DIM
#define TILE_DIM 16
#endif

__kernel void mat_transpose_f32_4(__global float* input,
                              __global float* output,
                              const int rows,
                              const int cols) {
    const int current_row = get_global_id(0);
    const int current_col = get_global_id(1);

    const int group_row = get_local_id(0);
    const int group_col = get_local_id(1);

    __local float tile[TILE_DIM][TILE_DIM + 1];

    tile[group_col][group_row] = input[current_row * cols + current_col];
    barrier(CLK_LOCAL_MEM_FENCE);

    int transposed_row = get_group_id(1) * TILE_DIM + group_row;
    int transposed_col = get_group_id(0) * TILE_DIM + group_col;
    output[transposed_row * rows + transposed_col] = tile[group_row][group_col];
}
