#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "leveldb/db.h"
#include "host_secondary_index_manager.h"
#include "json_val.h"

class SecondaryIndexEngine {
public:
    
    struct Record {
        uint32_t primary_key;
        std::unordered_map<std::string, uint32_t> attributes;
    };

    
    
    SecondaryIndexEngine(const std::string& db_path, 
                        const std::string& indexed_attribute_name,
                        size_t flush_threshold = 5);
    ~SecondaryIndexEngine();

    
    bool insert_temperature(uint32_t primary_key, uint32_t temperature);

    
    [[deprecated("Use insert_temperature() instead")]]
    bool insert(uint32_t primary_key, 
               const std::unordered_map<std::string, uint32_t>& attributes);

    
    std::vector<Record> query_by_attribute(uint32_t attribute_value);

    
    std::vector<Record> range_query_by_attribute(uint32_t attr_lower, uint32_t attr_upper);

    
    void print_status() const;

    
    const std::string& get_indexed_attribute_name() const { return indexed_attribute_name_; }

private:
    
    Record fetch_record_from_db(uint32_t primary_key);

    std::unique_ptr<leveldb::DB> db_;
    std::unique_ptr<HostSecondaryIndexManager> manager_;
    std::string indexed_attribute_name_;  
}; 