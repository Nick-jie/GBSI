#include "gpu_btree_index_manager.h"
#include "GpuBTree.h"
#include <iostream>
#include <iomanip>
#include <set>
#include <unordered_set>


namespace std {
    template<>
    struct hash<std::pair<uint32_t, uint32_t>> {
        size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
            
            return hash<uint64_t>()((static_cast<uint64_t>(p.first) << 32) | p.second);
        }
    };
}

GpuBTreeIndexManager::GpuBTreeIndexManager() {
    btree_ = new GpuBTree::GpuBTreeMap<uint32_t, uint32_t, uint32_t, PoolAllocator>();
}

GpuBTreeIndexManager::~GpuBTreeIndexManager() {
    if (btree_) {
        btree_->free();
        delete btree_;
        btree_ = nullptr;
    }
}

void GpuBTreeIndexManager::insert_from_input(const GpuBTreeInput& input) {
    if (input.empty()) return;

    uint32_t num = input.size();
    
    uint32_t *d_keys = nullptr, *d_values = nullptr;
    CHECK_ERROR(memoryUtil::deviceAlloc(d_keys, num));
    CHECK_ERROR(memoryUtil::deviceAlloc(d_values, num));
    
    std::vector<uint32_t> temp_keys(input.keys);
    std::vector<uint32_t> temp_values(input.values);
    
    CHECK_ERROR(memoryUtil::cpyToDevice(temp_keys.data(), d_keys, num));
    CHECK_ERROR(memoryUtil::cpyToDevice(temp_values.data(), d_values, num));

    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA data copy synchronization error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_keys);
        cudaFree(d_values);
        return;
    }


    btree_->insertKeys(d_keys, d_values, num, SourceT::DEVICE);

    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA insert synchronization error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_keys);
    cudaFree(d_values);
}

void GpuBTreeIndexManager::insert_batch(const std::vector<std::pair<uint32_t, uint32_t>>& entries) {
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

uint32_t GpuBTreeIndexManager::search(uint32_t key) const {
    uint32_t query = key;
    uint32_t result = 0;
    
    btree_->searchKeys(&query, &result, 1, SourceT::HOST);
    
    return result;
}

std::vector<std::pair<uint32_t, uint32_t>> GpuBTreeIndexManager::range_query(
    uint32_t lower, uint32_t upper) const {
    
    
    const uint32_t MAX_RESULTS = 1024;  
    std::vector<uint32_t> results(MAX_RESULTS * 2, 0xFFFFFFFF);
    
    
    std::vector<uint32_t> keys_lower = {lower};
    std::vector<uint32_t> keys_upper = {upper};

    
    uint32_t* d_results = nullptr;
    CHECK_ERROR(memoryUtil::deviceAlloc(d_results, MAX_RESULTS * 2));
    CHECK_ERROR(cudaMemset(d_results, 0xFF, MAX_RESULTS * 2 * sizeof(uint32_t)));

    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA memory operation synchronization error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_results);
        return {};
    }

    
    btree_->rangeQuery(keys_lower.data(), keys_upper.data(), d_results, 
                      MAX_RESULTS, 1, SourceT::HOST);

    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA query synchronization error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_results);
        return {};
    }

    
    CHECK_ERROR(memoryUtil::cpyToHost(d_results, results.data(), MAX_RESULTS * 2));
    cudaFree(d_results);

    
    std::unordered_set<std::pair<uint32_t, uint32_t>> unique_pairs;
    std::vector<std::pair<uint32_t, uint32_t>> ordered_pairs;
    ordered_pairs.reserve(MAX_RESULTS);  

    for (uint32_t i = 0; i + 1 < results.size(); i += 2) {
        
        if (results[i] == 0xFFFFFFFF) {
            break;
        }
        
        if (results[i] >= lower && results[i] <= upper) {
            auto pair = std::make_pair(results[i], results[i + 1]);
            
            if (unique_pairs.insert(pair).second) {
                ordered_pairs.push_back(pair);
            }
        }
    }

    
    if (ordered_pairs.size() >= MAX_RESULTS) {
        std::cerr << "Warning: Range query results may be truncated, current limit is " << MAX_RESULTS 
                  << " records. Query range: [" << lower << ", " << upper << "]\n";
    }
    
    return ordered_pairs;
}

double GpuBTreeIndexManager::memory_usage() const {
    return btree_->compute_usage();
}

void GpuBTreeIndexManager::reset() {
    if (btree_) {
        btree_->free();
        delete btree_;
    }
    btree_ = new GpuBTree::GpuBTreeMap<uint32_t, uint32_t, uint32_t, PoolAllocator>();
}

void GpuBTreeIndexManager::print_tree_content(bool detailed) const {
    if (!btree_) {
        std::cout << "\n[GPU B-tree] Tree is empty\n";
        return;
    }

    std::cout << "\n[GPU B-tree] Content statistics:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Memory usage: " << std::fixed << std::setprecision(3) 
              << memory_usage() << " GiB\n";

    
    uint32_t min_key = 0;
    uint32_t max_key = std::numeric_limits<uint32_t>::max();
    
    
    auto all_pairs = range_query(min_key, max_key);
    
    std::cout << "Total number of key-value pairs: " << all_pairs.size() << "\n\n";

    if (detailed) {
        std::cout << "All key-value pairs:\n";
        for (const auto& [key, value] : all_pairs) {
            std::cout << "  " << (key / 10000.0) << " -> " << value << "\n";
        }
    } else {
        
        std::cout << "First 5 key-value pairs:\n";
        for (size_t i = 0; i < std::min(size_t(5), all_pairs.size()); ++i) {
            std::cout << "  " << (all_pairs[i].first / 10000.0) << " -> " 
                     << all_pairs[i].second << "\n";
        }
        
        if (all_pairs.size() > 10) {
            std::cout << "  ...\n";
            std::cout << "Last 5 key-value pairs:\n";
            for (size_t i = all_pairs.size() - std::min(size_t(5), all_pairs.size()); 
                 i < all_pairs.size(); ++i) {
                std::cout << "  " << (all_pairs[i].first / 10000.0) << " -> " 
                         << all_pairs[i].second << "\n";
            }
        }
    }
    std::cout << "----------------------------------------\n";
} 