#ifndef WG_SIZE
#define WG_SIZE 256
#endif

#define REDUCE_STEP(val, offset)      \
    if (WG_SIZE >= offset*2) {        \
        if (lid < offset) {           \
            lsum[lid] += lsum[lid+offset]; \
        }                             \
        barrier(CLK_LOCAL_MEM_FENCE); \
    }

#define UNROLL_REDUCTION()      \
    REDUCE_STEP(val, 256)       \
    REDUCE_STEP(val, 128)       \
    REDUCE_STEP(val, 64)        \
    REDUCE_STEP(val, 32)        \
    REDUCE_STEP(val, 16)        \
    REDUCE_STEP(val, 8)         \
    REDUCE_STEP(val, 4)         \
    REDUCE_STEP(val, 2)         \
    REDUCE_STEP(val, 1)


inline void atomic_add_f32(__global float* addr, float val)
{
    union {
        unsigned int u32;
        float        f32;
    } next, expected, current;

    // Read the old value
    current.f32 = *addr;
    do {
        // Save the old value in expected
        expected.f32 = current.f32;

        // Compute the new value
        next.f32 = expected.f32 + val;

        // Atomically compare and swap the old with the new
        current.u32 = atomic_cmpxchg(
            (volatile __global unsigned int*) addr,
            expected.u32,
            next.u32
        );
        // If current.u32 != expected.u32, another thread changed it;
        // so we retry until it succeeds.
    } while (current.u32 != expected.u32);
}


__kernel void dot_product_f32_vec4(
    __global const float* a,      // Padded float array (length multiple of 4)
    __global const float* b,      // Padded float array (length multiple of 4)
    __global float*       out,    // Single-element buffer for final dot-product
    const int             length   // Guaranteed to be multiple of 4
)
{
    // Work-group info
    int lid       = get_local_id(0);
    int groupSize = get_local_size(0);
    int gid       = get_global_id(0);
    int globalSize = get_global_size(0);

    // Allocate local memory for partial sums
    __local float lsum[256];  // Or use __local float lsum[groupSize] if known

    // Each thread accumulates partial sums in 'partialSum'
    float partialSum = 0.0f;

    // Process the array in strides of (4 * globalSize)
    //   i = 4*gid, 4*gid + 4*globalSize, 4*gid + 8*globalSize, ...
    // Each iteration loads 4 elements (float4).
    for (int i = 4 * gid; i < length; i += 4 * globalSize) {
        // Vector load from a + b
        float4 va = vload4(0, &a[i]);
        float4 vb = vload4(0, &b[i]);

        // Multiply component-wise and sum all 4
        float4 product = va * vb;  // = (va.x * vb.x, va.y * vb.y, ...)
        float sum4 = product.s0 + product.s1 + product.s2 + product.s3;
        partialSum += sum4;
    }

    // Store partial sum in local memory
    lsum[lid] = partialSum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // --------------------------------------------------------------------
    // Intra-group parallel unrolled reduction
    // --------------------------------------------------------------------
    UNROLL_REDUCTION();

    // --------------------------------------------------------------------
    // One thread per group does an atomic add to the global output
    // --------------------------------------------------------------------
    if (lid == 0) {
        atomic_add_f32(out, lsum[0]);
    }
}
