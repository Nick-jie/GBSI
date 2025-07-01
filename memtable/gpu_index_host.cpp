#include "secondary_readonlymemtable.h"
#include "gpu_index_host.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>



void GpuBTreeInput::print(bool detailed) const {
    std::cout << "\n[GpuBTreeInput] Data statistics:\n";
    std::cout << "Total number of key-value pairs: " << size() << "\n";
    
    if (detailed) {
        std::cout << "\nDetailed data:\n";
        std::cout << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < size(); ++i) {
            std::cout << "  " << (keys[i] / 10000.0) << " -> " << values[i] << "\n";
        }
    } else {
        
        std::cout << "\nFirst 5 key-value pairs:\n";
        for (size_t i = 0; i < std::min(size_t(5), size()); ++i) {
            std::cout << "  " << (keys[i] / 10000.0) << " -> " << values[i] << "\n";
        }
        
        if (size() > 10) {
            std::cout << "  ...\n";
            std::cout << "Last 5 key-value pairs:\n";
            for (size_t i = size() - std::min(size_t(5), size()); i < size(); ++i) {
                std::cout << "  " << (keys[i] / 10000.0) << " -> " << values[i] << "\n";
            }
        }
    }
    std::cout << "\n";
}

void GpuBTreeInput::print_stats() const {
    std::cout << "\n[GpuBTreeInput] Detailed statistics:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Data volume statistics:\n";
    std::cout << "  Total number of key-value pairs: " << size() << "\n";
    std::cout << "  Memory usage: " << (size() * sizeof(uint32_t) * 2) / 1024.0 << " KB\n";
    
    if (!empty()) {
        auto [min_key, max_key] = get_key_range();
        auto [min_val, max_val] = get_value_range();
        
        std::cout << "\nValue range:\n";
        std::cout << "  Key range: [" << (min_key / 10000.0) << ", " << (max_key / 10000.0) << "]\n";
        std::cout << "  Value range: [" << min_val << ", " << max_val << "]\n";
    }
    
    std::cout << "----------------------------------------\n";
}

size_t GpuBTreeInput::size() const {
    return keys.size();
}

bool GpuBTreeInput::empty() const {
    return keys.empty();
}

std::pair<uint32_t, uint32_t> GpuBTreeInput::get_key_range() const {
    if (empty()) {
        return {0, 0};
    }
    return {
        *std::min_element(keys.begin(), keys.end()),
        *std::max_element(keys.begin(), keys.end())
    };
}

std::pair<uint32_t, uint32_t> GpuBTreeInput::get_value_range() const {
    if (empty()) {
        return {0, 0};
    }
    return {
        *std::min_element(values.begin(), values.end()),
        *std::max_element(values.begin(), values.end())
    };
}

bool GpuBTreeInput::validate() const {
    
    if (keys.size() != values.size()) {
        std::cerr << "Error: keys and values size mismatch\n";
        return false;
    }
    
    
    if (empty()) {
        std::cerr << "Warning: data is empty\n";
        return true;  
    }
    
    
    for (size_t i = 1; i < keys.size(); ++i) {
        if (keys[i] < keys[i-1]) {
            std::cerr << "Warning: keys are not sorted\n";
            break;
        }
    }
    
    return true;
}
