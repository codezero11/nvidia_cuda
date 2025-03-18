#include <stdio.h>
#include <stdint.h>
#include <curand_kernel.h> // For generating random numbers in CUDA

__global__ void generate_random_hex(uint64_t range_start, uint64_t range_end, int num_values) {
    uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x; // Calculate thread's unique ID
    uint64_t total_threads = blockDim.x * gridDim.x;           // Total threads in the grid

    // Initialize the random number generator using a unique seed for each thread
    curandState state;
    curand_init(clock64() + thread_id, thread_id, 0, &state);

    for (int i = thread_id; i < num_values; i += total_threads) {
        // Generate a random 64-bit number scaled to the range
        uint64_t random_offset = curand(&state); // Generate a random 32-bit number
        random_offset = (random_offset << 32) | curand(&state); // Extend it to 64 bits
        uint64_t random_value = range_start + (random_offset % (range_end - range_start + 1));
        printf("Thread %llu: 0x%016llx\n", (long long unsigned)thread_id, (long long unsigned)random_value);
    }
}

int main() {
    // Define the 64-bit range using constants
    uint64_t range_start = 0x1000000000000000ULL; // Start of the range
    uint64_t range_end = 0x1FFFFFFFFFFFFFFFULL;   // End of the range
    int num_values = 1000; // Total random numbers to generate per iteration

    // Configuration for the CUDA kernel
    int threads_per_block = 1024;  // Number of threads per block
    int number_of_blocks = 5888;  // Adjusted for optimal GPU usage

    // Infinite loop for continuous generation
    while (true) {
        // Launch the kernel
        generate_random_hex<<<number_of_blocks, threads_per_block>>>(range_start, range_end, num_values);

        // Ensure all threads complete execution before launching again
        cudaDeviceSynchronize();
    }

    return 0; // Technically, this will never be reached due to the infinite loop
}

