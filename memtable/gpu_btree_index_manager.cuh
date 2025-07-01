#ifndef GPU_BTREE_INDEX_MANAGER_CUH
#define GPU_BTREE_INDEX_MANAGER_CUH

#include <vector>
#include <utility>
#include <cstdint>
#include <cuda_runtime.h>
#include "GpuBTree.h"
#include "gpu_index_host.h"
#include "utils.cuh"
#include "gpu_btree_index_manager.h"

class GpuBTreeIndexManager {
public:
    
    __host__ GpuBTreeIndexManager();
    __host__ ~GpuBTreeIndexManager();

    
    __host__ void insert_from_input(const GpuBTreeInput& input);
    
    
    __host__ void insert_batch(const std::vector<std::pair<uint32_t, uint32_t>>& entries);

    
    __host__ uint32_t search(uint32_t key) const;

    
    __host__ std::vector<std::pair<uint32_t, uint32_t>> range_query(
        uint32_t lower, uint32_t upper, uint32_t avg_len = 8) const;

    
    __host__ double memory_usage() const;

   
    __host__ void reset();

private:
    GpuBTree::GpuBTreeMap<uint32_t, uint32_t, uint32_t>* btree_;
};


__host__ inline GpuBTreeIndexManager::GpuBTreeIndexManager() {
    btree_ = new GpuBTree::GpuBTreeMap<uint32_t, uint32_t, uint32_t>();
}

__host__ inline GpuBTreeIndexManager::~GpuBTreeIndexManager() {
    if (btree_) {
        btree_->free();
        delete btree_;
        btree_ = nullptr;
    }
}

__host__ inline void GpuBTreeIndexManager::insert_from_input(const GpuBTreeInput& input) {
    if (input.empty()) return;

    uint32_t num = input.size();
    
    uint32_t *d_keys = nullptr, *d_values = nullptr;
    CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, num));
    CHECK_ERROR(memoryUtil::deviceAlloc(d_values, num));
    
    std::vector<uint32_t> temp_keys(input.keys);
    std::vector<uint32_t> temp_values(input.values);
    
    CHECK_ERROR(memoryUtil::cpyToDevice(temp_keys.data(), d_keys, num));
    CHECK_ERROR(memoryUtil::cpyToDevice(temp_values.data(), d_values, num));

    btree_->insertKeys(d_keys, d_values, num, SourceT::DEVICE);

    cudaFree(d_keys);
    cudaFree(d_values);
}

__host__ inline void GpuBTreeIndexManager::insert_batch(const std::vector<std::pair<uint32_t, uint32_t>>& entries) {
    if (entries.empty()) return;

    uint32_t num = entries.size();
    std::vector<uint32_t> keys(num), values(num);
    
    for (uint32_t i = 0; i < num; ++i) {
        keys[i] = entries[i].first;
        values[i] = entries[i].second;
    }

    uint32_t *d_keys = nullptr, *d_values = nullptr;
    CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, num));
    CHECK_ERROR(memoryUtil::deviceAlloc(d_values, num));
    
    CHECK_ERROR(memoryUtil::cpyToDevice(keys.data(), d_keys, num));
    CHECK_ERROR(memoryUtil::cpyToDevice(values.data(), d_values, num));

    btree_->insertKeys(d_keys, d_values, num, SourceT::DEVICE);

    cudaFree(d_keys);
    cudaFree(d_values);
}

__host__ inline uint32_t GpuBTreeIndexManager::search(uint32_t key) const {
    uint32_t query = key;
    uint32_t result = 0;
    
    btree_->searchKeys(&query, &result, 1, SourceT::HOST);
    
    return result;
}

__host__ inline std::vector<std::pair<uint32_t, uint32_t>> GpuBTreeIndexManager::range_query(
    uint32_t lower, uint32_t upper, uint32_t avg_len) const {
    
    std::vector<uint32_t> keys_lower = {lower};
    std::vector<uint32_t> keys_upper = {upper};
    uint32_t len = avg_len * 2;
    std::vector<uint32_t> results(len);

    btree_->rangeQuery(keys_lower.data(), keys_upper.data(), results.data(), avg_len, 1, SourceT::HOST);

    std::vector<std::pair<uint32_t, uint32_t>> filtered;
    for (uint32_t i = 0; i + 1 < len; i += 2) {
        if (results[i] == 0xFFFFFFFF || results[i + 1] == 0xFFFFFFFF) break;
        filtered.emplace_back(results[i], results[i + 1]);
    }
    
    return filtered;
}

__host__ inline double GpuBTreeIndexManager::memory_usage() const {
    return btree_->compute_usage();
}

__host__ inline void GpuBTreeIndexManager::reset() {
    if (btree_) {
        btree_->free();
        delete btree_;
    }
    btree_ = new GpuBTree::GpuBTreeMap<uint32_t, uint32_t, uint32_t>();
}

#endif
