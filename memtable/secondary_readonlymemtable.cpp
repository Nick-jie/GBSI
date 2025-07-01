#include "secondary_readonlymemtable.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "secondary_memtable_iterator.h"

SecondaryReadOnlyMemTable::SecondaryReadOnlyMemTable(std::vector<std::pair<uint32_t, uint32_t>>&& data)
    : data_(std::move(data)) {}

std::size_t SecondaryReadOnlyMemTable::size() const {
    return data_.size();
}

void SecondaryReadOnlyMemTable::print() const {
    std::cout << "\n[SecondaryReadOnlyMemTable] total = " << data_.size() << "\n";
    std::cout << std::fixed << std::setprecision(4);

    for (auto it = begin(); it != end(); ++it) {
        auto [k, v] = *it;
        std::cout << "  " << (k / 10000.0) << "  ->  " << v << "\n";
    }

    std::cout << "---------------------------------------\n";
}

std::vector<uint32_t> SecondaryReadOnlyMemTable::get(uint32_t key) const {
    std::vector<uint32_t> result;

    for (auto it = begin(); it != end(); ++it) {
        auto [k, v] = *it;
        if (k == key) {
            result.push_back(v);
        } else if (k > key) {
            break;
        }
    }

    return result;
}

std::vector<uint32_t> SecondaryReadOnlyMemTable::range_query(uint32_t lower, uint32_t upper) const {
    std::vector<uint32_t> result;

    for (auto it = begin(); it != end(); ++it) {
        auto [k, v] = *it;
        if (k >= lower && k <= upper) {
            result.push_back(v);
        } else if (k > upper) {
            break;
        }
    }

    return result;
}

GpuBTreeInput SecondaryReadOnlyMemTable::flush() const {
    GpuBTreeInput gpu_input;
    
    
    gpu_input.keys.reserve(data_.size());
    gpu_input.values.reserve(data_.size());
    
    
    for (const auto& [key, value] : data_) {
        gpu_input.keys.push_back(key);
        gpu_input.values.push_back(value);
    }
    
    return gpu_input;
}

secidx::SecondaryReadOnlyMemTableIterator SecondaryReadOnlyMemTable::begin() const {
    return secidx::SecondaryReadOnlyMemTableIterator(data_.begin());
}

secidx::SecondaryReadOnlyMemTableIterator SecondaryReadOnlyMemTable::end() const {
    return secidx::SecondaryReadOnlyMemTableIterator(data_.end());
}
