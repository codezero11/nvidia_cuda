#include <stdio.h>
#include <stdint.h>
#include <inttypes.h> // For printing 64-bit integers

__global__ void traverse_range(uint64_t start, uint64_t end) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; // Unique index for each thread
    uint64_t stride = blockDim.x * gridDim.x;            // Total threads in the grid

    for (uint64_t i = start + idx; i <= end; i += stride) {
        printf("Thread %llu: 0x%016llx\n", (long long unsigned)idx, (long long unsigned)i);
    }
}

int main() {
    // Define the range using split parts (workaround for large numbers)
    uint64_t start = 0x1000000000000000ULL;  // Upper part represented as a 64-bit constant
    uint64_t end = 0x1FFFFFFFFFFFFFFFULL;    // Upper part represented as a 64-bit constant

    // CUDA kernel configuration
    int threadsPerBlock = 1024;
    int numBlocks = 5888;  // Adjusted to better use an RTX 4070

    // Launch the kernel
    traverse_range<<<numBlocks, threadsPerBlock>>>(start, end);

    // Wait for all threads to finish
    cudaDeviceSynchronize();

    return 0;
}
