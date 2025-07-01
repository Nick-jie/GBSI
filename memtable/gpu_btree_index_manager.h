#ifndef GPU_BTREE_INDEX_MANAGER_H
#define GPU_BTREE_INDEX_MANAGER_H


#include <cuda_runtime.h>
#include <cuda.h>


#include "global.cuh"
#include "utils.cuh"


#include "allocator/pool_allocator.cuh"


#include <vector>
#include <utility>
#include <cstdint>


#include "gpu_index_host.h"


namespace GpuBTree {
    template<typename K, typename V, typename S, typename A>
    class GpuBTreeMap;
}

class GpuBTreeIndexManager {
public:
    
    GpuBTreeIndexManager();
    ~GpuBTreeIndexManager();

    
    void insert_from_input(const GpuBTreeInput& input);
    
    
    void insert_batch(const std::vector<std::pair<uint32_t, uint32_t>>& entries);

    
    uint32_t search(uint32_t key) const;

    
    std::vector<std::pair<uint32_t, uint32_t>> range_query(
        uint32_t lower,
        uint32_t upper
    ) const;

    
    double memory_usage() const;

    
    void reset();

    
    void print_tree_content(bool detailed = false) const;

private:
    GpuBTree::GpuBTreeMap<uint32_t, uint32_t, uint32_t, PoolAllocator>* btree_;
};

#endif 