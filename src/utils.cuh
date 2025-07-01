#ifndef UTILS_CUH
#define UTILS_CUH

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

namespace memoryUtil {
template<typename DataT>
cudaError_t cpyToHost(DataT*& src_data, DataT* dst_data, std::size_t count) {
  return cudaMemcpy(dst_data, src_data, sizeof(DataT) * count, cudaMemcpyDeviceToHost);
}

template<typename DataT>
cudaError_t cpyToDevice(DataT* src_data, DataT*& dst_data, std::size_t count) {
  return cudaMemcpy(dst_data, src_data, sizeof(DataT) * count, cudaMemcpyHostToDevice);
}

template<typename DataT>
cudaError_t deviceAlloc(DataT*& src_data, std::size_t count) {
  return cudaMalloc((void**)&src_data, sizeof(DataT) * count);
}

template<typename DataT, typename ByteT>
cudaError_t deviceSet(DataT*& src_data, std::size_t count, ByteT value) {
  return cudaMemset(src_data, value, sizeof(DataT) * count);
}

template<typename DataT>
cudaError_t deviceFree(DataT* src_data) {
  return cudaFree(src_data);
}
}  // namespace memoryUtil

#define CHECK_ERROR(call)                                                               \
  do {                                                                                  \
    cudaError_t err = call;                                                             \
    if (err != cudaSuccess) {                                                           \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  } while (0)

#define LANEID_REVERSED(laneId) (31 - laneId)

__device__ __forceinline__ unsigned lane_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

class GpuTimer {
private:
  cudaEvent_t start, stop;
  float temp_time;

 public:
  GpuTimer() : temp_time(0.0f) {}

  void timerStart() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
  }

  void timerStop() {
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  float getTime() { return temp_time; }
};

#endif
