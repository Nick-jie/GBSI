#include "data_export.h"
#include "json_val.h"
#include <iostream>
#include <memory>
#include <memory> 

void print_parsed_json(const std::string& value_str) {
    Val val = parser(value_str);
    std::cout << "  Temp : " << val["Temp"] << "\n";
    std::cout << "  Press: " << val["Press"] << "\n";
}

void export_all_forward(leveldb::DB* db) {
    std::cout << "=== Export Forward (Head to Tail) ===\n";
    leveldb::ReadOptions opts;
    std::unique_ptr<leveldb::Iterator> iter(db->NewIterator(opts));

    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
        std::string key = iter->key().ToString();
        std::string value = iter->value().ToString();

        std::cout << "Key: " << key << "\n";
        std::cout << "Value (Raw): " << value << "\n";
        print_parsed_json(value);
        std::cout << "-----------------------------\n";
    }
}

void export_all_reverse(leveldb::DB* db) {
    std::cout << "=== Export Reverse (Tail to Head) ===\n";
    leveldb::ReadOptions opts;
    std::unique_ptr<leveldb::Iterator> iter(db->NewIterator(opts));

    for (iter->SeekToLast(); iter->Valid(); iter->Prev()) {
        std::string key = iter->key().ToString();
        std::string value = iter->value().ToString();

        std::cout << "Key: " << key << "\n";
        std::cout << "Value (Raw): " << value << "\n";
        print_parsed_json(value);
        std::cout << "-----------------------------\n";
    }
}

void export_from_key_reverse(leveldb::DB* db, const std::string& start_key) {
    std::cout << "=== Export from Key \"" << start_key << "\" Reverse ===\n";
    leveldb::ReadOptions opts;
    std::unique_ptr<leveldb::Iterator> iter(db->NewIterator(opts));

    for (iter->Seek(start_key); iter->Valid(); iter->Prev()) {
        std::string key = iter->key().ToString();
        std::string value = iter->value().ToString();

        std::cout << "Key: " << key << "\n";
        std::cout << "Value (Raw): " << value << "\n";
        print_parsed_json(value);
        std::cout << "-----------------------------\n";
    }
}

int count_entries(leveldb::DB* db) {
    int count = 0;
    leveldb::ReadOptions opts;
    std::unique_ptr<leveldb::Iterator> iter(db->NewIterator(opts));

    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
        ++count;
    }

    return count;
}
