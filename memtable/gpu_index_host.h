#ifndef GPU_INDEX_CONVERTER_H
#define GPU_INDEX_CONVERTER_H

#include <vector>
#include <cstdint>
#include <string>
#include <utility>

class SecondaryReadOnlyMemTable;


struct GpuBTreeInput {
    std::vector<uint32_t> keys;    
    std::vector<uint32_t> values;  

    
    void print(bool detailed = false) const;  
    void print_stats() const;                 
    size_t size() const;                      
    bool empty() const;                       
    
    
    std::pair<uint32_t, uint32_t> get_key_range() const;    
    std::pair<uint32_t, uint32_t> get_value_range() const;  
    
    
    bool validate() const;  
};



#endif  // GPU_INDEX_CONVERTER_H
