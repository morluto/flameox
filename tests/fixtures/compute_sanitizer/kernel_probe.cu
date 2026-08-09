#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void write_values(int *values, int count, bool out_of_bounds) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index + (out_of_bounds ? count : 0)] = index;
    }
}

int main(int argc, char **argv) {
    constexpr int count = 32;
    int *values = nullptr;
    if (cudaMalloc(&values, count * sizeof(int)) != cudaSuccess) {
        return 2;
    }
    write_values<<<1, count>>>(values, count, argc > 1);
    cudaError_t status = cudaDeviceSynchronize();
    cudaFree(values);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        return 3;
    }
    return 0;
}
