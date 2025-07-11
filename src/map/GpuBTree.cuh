﻿
#pragma once

#include <stdio.h>

#include <cstdint>

namespace GpuBTree {

template<typename KeyT,
         typename ValueT,
         typename SizeT = KeyT,
         typename AllocatorT = PoolAllocator>
class GpuBTreeMap {
 private:
  static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;
  static constexpr uint32_t DELETED_KEY = 0xFFFFFFFF;
  static constexpr uint32_t BLOCKSIZE_BUILD_ = 128;
  static constexpr uint32_t BLOCKSIZE_SEARCH_ = 1024;

  SizeT _num_keys;
  int _device_id;
  uint32_t* _d_root;
  AllocatorT _mem_allocator;

  cudaError_t initBTree(uint32_t*& root, cudaStream_t stream_id = 0);
  cudaError_t insertKeys(uint32_t*& root,
                         KeyT*& d_keys,
                         ValueT*& d_values,
                         SizeT& count,
                         cudaStream_t stream_id = 0);
  cudaError_t searchKeys(uint32_t*& root,
                         KeyT*& d_queries,
                         ValueT*& d_results,
                         SizeT& count,
                         cudaStream_t stream_id = 0);
  cudaError_t compactTree(uint32_t*& root,
                          KeyT*& d_tree,
                          SizeT*& d_num_nodes,
                          cudaStream_t stream_id = 0);
  cudaError_t deleteKeys(uint32_t*& root,
                         KeyT*& d_queries,
                         SizeT& count,
                         cudaStream_t stream_id = 0);
  cudaError_t rangeQuery(uint32_t*& root,
                         KeyT*& d_queries_lower,
                         KeyT*& d_queries_upper,
                         ValueT*& d_range_results,
                         SizeT& count,
                         SizeT& range_lenght,
                         cudaStream_t stream_id = 0);
  bool _handle_memory;

 public:
  GpuBTreeMap(AllocatorT* mem_allocator = nullptr, int device_id = 0) {
    if (mem_allocator) {
      _mem_allocator = *mem_allocator;
      _handle_memory = false;
    } else {
      PoolAllocator allocator;
      _mem_allocator = allocator;
      _mem_allocator.init();
      CHECK_ERROR(memoryUtil::deviceAlloc(_d_root, 1));
      _handle_memory = true;
    }
    _device_id = device_id;
    CHECK_ERROR(cudaSetDevice(_device_id));
    initBTree(_d_root);
  }
  cudaError_t init(AllocatorT mem_allocator, uint32_t* root_, int deviceId = 0) {
    _device_id = deviceId;
    _mem_allocator = mem_allocator;
    _d_root = root_;
    return cudaSuccess;
  }
  ~GpuBTreeMap() {}
  void free() {
    if (_handle_memory) {
      CHECK_ERROR(cudaDeviceSynchronize());
      _mem_allocator.free();
    }
  }

  __host__ __device__ AllocatorT* getAllocator() { return &_mem_allocator; }
  __host__ __device__ uint32_t* getRoot() { return _d_root; }
  cudaError_t insertKeys(KeyT* keys,
                         ValueT* values,
                         SizeT count,
                         SourceT source = SourceT::DEVICE) {
    KeyT* d_keys;
    ValueT* d_values;
    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, count));
      CHECK_ERROR(memoryUtil::deviceAlloc(d_values, count));
      CHECK_ERROR(memoryUtil::cpyToDevice(keys, d_keys, count));
      CHECK_ERROR(memoryUtil::cpyToDevice(values, d_values, count));
    } else {
      d_keys = keys;
      d_values = values;
    }

    CHECK_ERROR(insertKeys(_d_root, d_keys, d_values, count));

    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceFree(d_keys));
      CHECK_ERROR(memoryUtil::deviceFree(d_values));
    }

    return cudaSuccess;
  }

  cudaError_t searchKeys(KeyT* queries,
                         ValueT* results,
                         SizeT count,
                         SourceT source = SourceT::DEVICE) {
    KeyT* d_queries;
    ValueT* d_results;
    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceAlloc(d_queries, count));
      CHECK_ERROR(memoryUtil::deviceAlloc(d_results, count));

      CHECK_ERROR(memoryUtil::cpyToDevice(queries, d_queries, count));
    } else {
      d_queries = queries;
      d_results = results;
    }

    CHECK_ERROR(searchKeys(_d_root, d_queries, d_results, count));

    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::cpyToHost(d_results, results, count));
      CHECK_ERROR(memoryUtil::deviceFree(d_queries));
      CHECK_ERROR(memoryUtil::deviceFree(d_results));
    }

    return cudaSuccess;
  }

  cudaError_t compactTree(KeyT*& btree,
                          SizeT max_nodes,
                          SizeT& num_nodes,
                          SourceT source = SourceT::DEVICE) {
    KeyT* d_tree;
    SizeT* d_num_nodes;
    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceAlloc(d_tree, std::size_t(max_nodes) * NODE_WIDTH));
      CHECK_ERROR(memoryUtil::deviceAlloc(d_num_nodes, 1));
    } else {
      d_tree = btree;
      d_num_nodes = &num_nodes;
    }

    CHECK_ERROR(compactTree(_d_root, d_tree, d_num_nodes));

    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::cpyToHost(d_num_nodes, &num_nodes, 1));
      CHECK_ERROR(memoryUtil::deviceFree(d_num_nodes));

      CHECK_ERROR(
          memoryUtil::cpyToHost(d_tree, btree, std::size_t(max_nodes) * NODE_WIDTH));
      CHECK_ERROR(memoryUtil::deviceFree(d_tree));
    }

    return cudaSuccess;
  }
  cudaError_t deleteKeys(KeyT* queries, SizeT count, SourceT source = SourceT::DEVICE) {
    KeyT* d_queries;
    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceAlloc(d_queries, count));
      CHECK_ERROR(memoryUtil::cpyToDevice(queries, d_queries, count));
    } else {
      d_queries = queries;
    }

    CHECK_ERROR(deleteKeys(_d_root, d_queries, count));

    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceFree(d_queries));
    }

    return cudaSuccess;
  }
  cudaError_t rangeQuery(KeyT* queries_lower,
                         KeyT* queries_upper,
                         ValueT* results,
                         SizeT average_length,
                         SizeT count,
                         SourceT source = SourceT::DEVICE) {
    KeyT* d_queries_lower;
    KeyT* d_queries_upper;
    KeyT* d_results;
    auto total_range_lenght = count * average_length * 2;
    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::deviceAlloc(d_queries_lower, count));
      CHECK_ERROR(memoryUtil::deviceAlloc(d_queries_upper, count));
      CHECK_ERROR(memoryUtil::deviceAlloc(d_results, total_range_lenght));
      CHECK_ERROR(memoryUtil::cpyToDevice(queries_lower, d_queries_lower, count));
      CHECK_ERROR(memoryUtil::cpyToDevice(queries_upper, d_queries_upper, count));
    } else {
      d_queries_lower = queries_lower;
      d_queries_upper = queries_upper;
      d_results = results;
    }

    CHECK_ERROR(rangeQuery(
        _d_root, d_queries_lower, d_queries_upper, d_results, count, average_length));

    if (source == SourceT::HOST) {
      CHECK_ERROR(memoryUtil::cpyToHost(d_results, results, total_range_lenght));
      CHECK_ERROR(memoryUtil::deviceFree(d_results));
      CHECK_ERROR(memoryUtil::deviceFree(d_queries_lower));
      CHECK_ERROR(memoryUtil::deviceFree(d_queries_upper));
    }

    return cudaSuccess;
  }

  double compute_usage() { return _mem_allocator.compute_usage(); }
};
};  // namespace GpuBTree
