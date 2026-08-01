#include <cuda_runtime.h>

#include <cstdio>

__global__ void add_one(const float* input, float* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index] + 1.0f;
  }
}

int main() {
  constexpr int count = 1024;
  constexpr size_t bytes = count * sizeof(float);
  float* host = new float[count]{};
  float* device_input = nullptr;
  float* device_output = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  cudaSetDevice(0);
  cudaMalloc(&device_input, bytes);
  cudaMalloc(&device_output, bytes);
  cudaStreamCreate(&stream);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMemcpyAsync(device_input, host, bytes, cudaMemcpyHostToDevice, stream);
  cudaEventRecord(start, stream);
  for (int index = 0; index < 3; ++index) {
    add_one<<<(count + 255) / 256, 256, 0, stream>>>(device_input, device_output, count);
  }
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  add_one<<<(count + 255) / 256, 256, 0, stream>>>(device_input, device_output, count);
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  for (int index = 0; index < 2; ++index) {
    cudaGraphLaunch(graph_exec, stream);
  }
  cudaStreamSynchronize(stream);

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(device_output);
  cudaFree(device_input);
  delete[] host;
  std::puts("nsight smoke complete");
  return 0;
}
