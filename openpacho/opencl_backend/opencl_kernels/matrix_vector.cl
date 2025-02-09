#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#define REDUCE_STEP(val, offset)                 \
    if (WG_SIZE >= offset * 2) {                 \
        if (lid < offset) {                      \
            lsum[lid] += lsum[lid + offset];     \
        }                                        \
        barrier(CLK_LOCAL_MEM_FENCE);            \
    }

#define UNROLL_REDUCTION()  \
    REDUCE_STEP(val, 256)   \
    REDUCE_STEP(val, 128)   \
    REDUCE_STEP(val, 64)    \
    REDUCE_STEP(val, 32)    \
    REDUCE_STEP(val, 16)    \
    REDUCE_STEP(val, 8)     \
    REDUCE_STEP(val, 4)     \
    REDUCE_STEP(val, 2)     \
    REDUCE_STEP(val, 1)


__kernel void matvec_f32_vec4(
    __global const float* matrix, // [N*M], row-major
    __global const float* vec,    // [M]
    __global float*       out,    // [N]
    const int             M,      // # of columns (multiple of 8)
    const int             N       // # of rows (multiple of 8)
)
{
    int row = get_global_id(0);
    if (row >= N) {
        return; 
    }

    // Local thread ID in [0, WG_SIZE)
    int lid = get_local_id(0);

    float partialSum = 0.0f;

    __local float4 vval0;
    __local float4 vval1;
    for (int col = 0; col < M; col += 8)
    {

        if (lid == 0) 
        {
            vval0 = vload4(0, &vec[col]);
            vval1 = vload4(0, &vec[col + 4]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float4 mval0 = vload4(0, &matrix[row * M + col]);
        float4 mval1 = vload4(0, &matrix[row * M + col + 4]);

        partialSum += dot(mval0, vval0) + dot(mval1, vval1);
    }
    out[row] = partialSum;
}