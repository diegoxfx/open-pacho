// matmul_kernel.cl

// Define the tile size; this should be set to the device's preferred work-group size multiple.
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

// Kernel for matrix multiplication using tiling.
// A is an MxK matrix, B is a KxN matrix, and C is the resulting MxN matrix.
__kernel void matmul_f32_tiled(
    __global const float* A,  // Matrix A stored in row-major order (M x K)
    __global const float* B,  // Matrix B stored in row-major order (K x N)
    __global float* C,        // Output matrix C stored in row-major order (M x N)
    const int M,              // Number of rows in A and C
    const int K,              // Number of columns in A and rows in B
    const int N)              // Number of columns in B and C
{
    // Global indices for the output element (row, col)
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    // Local indices within the work-group
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    // Allocate local memory for sub-tiles of A and B.
    __local float Asub[TILE_SIZE][TILE_SIZE];
    __local float Bsub[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over all the tiles. Since K is a multiple of TILE_SIZE, no remainder is left.
    const int numTiles = K / TILE_SIZE;
    for (int t = 0; t < numTiles; t++)
    {
        // Load one tile of A into local memory.
        Asub[localRow][localCol] = A[globalRow * K + t * TILE_SIZE + localCol];

        // Load one tile of B into local memory, transposing the tile.
        Bsub[localCol][localRow] = B[(t * TILE_SIZE + localRow) * N + globalCol];

        // Synchronize to make sure the tile is fully loaded.
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial product for this tile.
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 4)
        {
            float4 a_vec = vload4(0, &Asub[localRow][k]);
            float4 b_vec = vload4(0, &Bsub[localCol][k]);
            sum += dot(a_vec, b_vec);
        }

        // Synchronize to ensure that computation is done before loading the next tile.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the computed value to the output matrix C.
    C[globalRow * N + globalCol] = sum;
}
