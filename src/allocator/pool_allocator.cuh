#ifndef POOL_ALLOCATOR_CUH
#define POOL_ALLOCATOR_CUH

#include <stdio.h>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.cuh"

class PoolAllocator {
 public:
  PoolAllocator() : d_pool(nullptr), d_count(nullptr) {}
  ~PoolAllocator() {}
  void init() {
    if (d_pool == nullptr) {
    CHECK_ERROR(memoryUtil::deviceAlloc(d_pool, MAX_SIZE));
    CHECK_ERROR(memoryUtil::deviceSet(d_pool, MAX_SIZE, 0x00));
    }
    if (d_count == nullptr) {
    CHECK_ERROR(memoryUtil::deviceAlloc(d_count, 1));
    CHECK_ERROR(memoryUtil::deviceSet(d_count, uint32_t(1), 0x00));
    }
  }
  void free() {
    if (d_pool != nullptr) {
    CHECK_ERROR(memoryUtil::deviceFree(d_pool));
      d_pool = nullptr;
    }
    if (d_count != nullptr) {
    CHECK_ERROR(memoryUtil::deviceFree(d_count));
      d_count = nullptr;
    }
  }

  double compute_usage() {
    uint32_t allocations_count;
    CHECK_ERROR(memoryUtil::cpyToHost(d_count, &allocations_count, 1));
    double num_bytes = double(allocations_count) * NODE_SIZE * sizeof(uint32_t);
    return num_bytes / (1u << 30);
  }

  PoolAllocator& operator=(const PoolAllocator& rhs) {
    if (this != &rhs) {
      free();
    d_pool = rhs.d_pool;
    d_count = rhs.d_count;
    }
    return *this;
  }

  template<typename AddressT = uint32_t>
  __device__ __forceinline__ AddressT allocate() {
    #if defined(__CUDA_ARCH__)
      return static_cast<AddressT>(atomicAdd(d_count, static_cast<uint32_t>(1)));
    #else
      return AddressT{};
    #endif
  }

  template<typename AddressT = uint32_t>
  __device__ __forceinline__ uint32_t* getAddressPtr(AddressT& address) {
    return d_pool + address * 32;
  }

  template<typename AddressT = uint32_t>
  __device__ __forceinline__ void freeAddress(AddressT& address) {}

  __host__ __device__ uint32_t getCapacity() { return MAX_SIZE; }

  __host__ __device__ uint32_t getOffset() { return *d_count; }

 private:
  uint32_t* d_pool;

  static constexpr uint64_t NODE_SIZE = 32;
  static constexpr uint64_t MAX_NODES = 1 << 25;
  static constexpr uint64_t MAX_SIZE = MAX_NODES * NODE_SIZE;
  uint32_t* d_count;
};

#endif