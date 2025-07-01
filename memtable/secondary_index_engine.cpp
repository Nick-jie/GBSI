#include "secondary_index_engine.h"
#include <iostream>
#include <iomanip>
#include "data_gen.h"

SecondaryIndexEngine::SecondaryIndexEngine(
    const std::string& db_path, 
    const std::string& indexed_attribute_name,
    size_t flush_threshold) 
    : indexed_attribute_name_(indexed_attribute_name) {
    
    leveldb::Options options;
    options.create_if_missing = true;
    
    leveldb::DB* db_raw;
    leveldb::Status status = leveldb::DB::Open(options, db_path, &db_raw);
    if (!status.ok()) {
        throw std::runtime_error("Failed to open LevelDB database: " + status.ToString());
    }
    db_.reset(db_raw);

    
    manager_ = std::make_unique<HostSecondaryIndexManager>(flush_threshold);
}

SecondaryIndexEngine::~SecondaryIndexEngine() = default;

bool SecondaryIndexEngine::insert_temperature(uint32_t primary_key, uint32_t temperature) {
    try {
        
        std::string db_key = std::to_string(primary_key);
        std::string value = std::to_string(temperature);
        leveldb::Status s = db_->Put(leveldb::WriteOptions(), db_key, value);
        if (!s.ok()) {
            return false;
        }

        
        manager_->insert(temperature, primary_key);
        return true;

    } catch (const std::exception& e) {
        return false;
    }
}

bool SecondaryIndexEngine::insert(
    uint32_t primary_key, 
    const std::unordered_map<std::string, uint32_t>& attributes) {
    try {
        
        auto it = attributes.find(indexed_attribute_name_);
        if (it == attributes.end()) {
            return false;
        }

        
        return insert_temperature(primary_key, it->second);

    } catch (const std::exception& e) {
        return false;
    }
}

SecondaryIndexEngine::Record SecondaryIndexEngine::fetch_record_from_db(uint32_t primary_key) {
    Record record;
    record.primary_key = primary_key;
    
    std::string db_key = std::to_string(primary_key);
    std::string value;
    leveldb::Status s = db_->Get(leveldb::ReadOptions(), db_key, &value);
    
    if (s.ok() && !value.empty()) {
        try {
            uint32_t attribute_value = std::stoul(value);
            record.attributes[indexed_attribute_name_] = attribute_value;
        } catch (const std::exception& e) {
            std::cerr << "Error converting value: " << e.what() << std::endl;
        }
    }
    
    return record;
}

std::vector<SecondaryIndexEngine::Record> SecondaryIndexEngine::query_by_attribute(uint32_t attribute_value) {
    std::vector<Record> results;
    
    
    auto primary_keys = manager_->query(attribute_value);
    
    
    for (const auto& pkey : primary_keys) {
        results.push_back(fetch_record_from_db(pkey));
    }
    
    return results;
}

std::vector<SecondaryIndexEngine::Record> SecondaryIndexEngine::range_query_by_attribute(
    uint32_t attr_lower, uint32_t attr_upper) {
    std::vector<Record> results;
    
    
    auto primary_keys = manager_->range_query(attr_lower, attr_upper);
    
    
    for (const auto& pkey : primary_keys) {
        results.push_back(fetch_record_from_db(pkey));
    }
    
    return results;
}

void SecondaryIndexEngine::print_status() const {
    std::cout << "\n[SecondaryIndexEngine Status]\n";
    std::cout << "Indexed attribute: " << indexed_attribute_name_ << "\n";
    if (manager_) {
        manager_->print_status();
    }
} 