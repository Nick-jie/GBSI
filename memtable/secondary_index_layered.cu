#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include "leveldb/db.h"
#include "leveldb/options.h"
#include "leveldb/env.h"
#include "leveldb/table.h"
#include <chrono>


#include "gpu_btree_index_manager.h"


uint32_t str2u32(const std::string& s) {
    
    size_t pos = s.find_first_of("0123456789");
    if (pos == std::string::npos) return 0;
    return static_cast<uint32_t>(std::stoul(s.substr(pos)));
}


double InsertData(leveldb::DB* db, int num_records) {
    auto start = std::chrono::high_resolution_clock::now();
    leveldb::WriteOptions write_options;
    for (int i = 0; i < num_records; ++i) {
        
        char key_buf[100];
        snprintf(key_buf, sizeof(key_buf), "pkey%06d", i);
        
        char skey_buf[100];
        snprintf(skey_buf, sizeof(skey_buf), "skey%06d", i % 1000); 
        leveldb::Status status = db->Put(write_options, key_buf, skey_buf);
        assert(status.ok());
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    std::cout << "Inserted " << num_records << " primary->secondary key records into LSM-Tree. Time: " << elapsed << " seconds\n";
    return elapsed;
}


double ForceCompaction(leveldb::DB* db) {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Manually triggering Compaction..." << std::endl;
    db->CompactRange(nullptr, nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    std::cout << "Compaction finished, all data flushed to disk. Time: " << elapsed << " seconds\n";
    return elapsed;
}


double GetSSTableMeta(leveldb::DB* db, std::vector<leveldb::SSTableMetaData>& sstables) {
    auto start = std::chrono::high_resolution_clock::now();
    db->GetCurrentSSTableMeta(&sstables);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    std::cout << "Found " << sstables.size() << " SSTable files. Metadata read time: " << elapsed << " seconds\n";
    for (const auto& meta : sstables) {
        std::cout << "  Level: " << meta.level << " File number: " << meta.number
                  << " Key range: [" << meta.smallest_key << ", " << meta.largest_key << "]\n";
    }
    return elapsed;
}


double ParseSSTable(const std::string& db_path, uint64_t file_number,
                  std::vector<std::pair<std::string, std::string>>& kvs) {
    auto start = std::chrono::high_resolution_clock::now();
    char file_buf[100];
    snprintf(file_buf, sizeof(file_buf), "%06llu.ldb", (unsigned long long)file_number);
    std::string file_path = db_path + "/" + file_buf;

    leveldb::RandomAccessFile* file = nullptr;
    leveldb::Status status = leveldb::Env::Default()->NewRandomAccessFile(file_path, &file);
    if (!status.ok()) {
        std::cerr << "  - Failed to open file: " << status.ToString() << std::endl;
        return 0.0;
    }
    uint64_t file_size;
    status = leveldb::Env::Default()->GetFileSize(file_path, &file_size);
    if (!status.ok()) {
        std::cerr << "  - Failed to get file size: " << status.ToString() << std::endl;
        delete file;
        return 0.0;
    }
    leveldb::Table* table = nullptr;
    status = leveldb::Table::Open(leveldb::Options(), file, file_size, &table);
    if (!status.ok()) {
        std::cerr << "  - Failed to parse SSTable: " << status.ToString() << std::endl;
        delete file;
        return 0.0;
    }
    leveldb::Iterator* iter = table->NewIterator(leveldb::ReadOptions());
    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
        leveldb::Slice internal_key = iter->key();
        
        leveldb::Slice user_key(internal_key.data(), internal_key.size() - 8);
        kvs.emplace_back(user_key.ToString(), iter->value().ToString());
    }
    delete iter;
    delete table;
    delete file;
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    return elapsed;
}


double BuildGpuBTreePerLevel(
    const std::map<int, std::vector<std::pair<std::string, std::string>>>& level_kvs)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::map<int, GpuBTreeIndexManager> gpu_btrees;

    for (const auto& [level, kvs] : level_kvs) {
        std::cout << "Building GPU B+Tree for Level " << level << ", data size: " << kvs.size() << std::endl;
        std::vector<std::pair<uint32_t, uint32_t>> entries;
        entries.reserve(kvs.size());
        for (const auto& kv : kvs) {
            uint32_t skey = str2u32(kv.second);
            uint32_t pkey = str2u32(kv.first);
            entries.emplace_back(skey, pkey);
        }
        auto gpu_start = std::chrono::high_resolution_clock::now();
        gpu_btrees[level].insert_batch(entries);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        double gpu_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count() / 1000.0;
        std::cout << "  Inserted secondary key->primary key batch into GPU B+Tree for Level " << level << ". Time: " << gpu_elapsed << " seconds\n";
        std::cout << "  [Level " << level << "] GPU B+Tree memory usage: "
                  << gpu_btrees[level].memory_usage() << " GiB\n";
        std::cout << "  [Level " << level << "] First 5 items in GPU B+Tree sample:\n";
        auto sample = gpu_btrees[level].range_query(0, 100000);
        for (size_t i = 0; i < std::min(sample.size(), size_t(5)); ++i) {
            std::cout << "    skey=" << sample[i].first << " -> pkey=" << sample[i].second << "\n";
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    std::cout << "Total build time for all levels of GPU B+Tree: " << elapsed << " seconds\n";
    return elapsed;
}

int main() {
    
    std::string db_path = "/opt/Leveldb_DB_DOC/secondary_index_layered_db";
    leveldb::Options options;
    options.create_if_missing = true;
    leveldb::DestroyDB(db_path, leveldb::Options());
    leveldb::DB* db = nullptr;
    leveldb::Status status = leveldb::DB::Open(options, db_path, &db);
    assert(status.ok());
    std::cout << "Database opened: " << db_path << std::endl;

    
    double insert_time = InsertData(db, 500000); 

    
    double compaction_time = ForceCompaction(db);

    
    std::vector<leveldb::SSTableMetaData> sstables;
    double meta_time = GetSSTableMeta(db, sstables);

    /
    std::map<int, std::vector<std::pair<std::string, std::string>>> level_kvs;
    double read_time = 0.0;
    for (const auto& meta : sstables) {
        std::vector<std::pair<std::string, std::string>> kvs;
        double t = ParseSSTable(db_path, meta.number, kvs);
        read_time += t;
        
        auto& vec = level_kvs[meta.level];
        vec.insert(vec.end(), kvs.begin(), kvs.end());
    }
    std::cout << "Total SSTable parse time: " << read_time << " seconds\n";

    
    double gpu_time = BuildGpuBTreePerLevel(level_kvs);

    
    delete db;
    std::cout << "All processes completed, database closed.\n";
    std::cout << "\n==== Summary ====\n";
    std::cout << "Insert time: " << insert_time << " seconds\n";
    std::cout << "Compaction time: " << compaction_time << " seconds\n";
    std::cout << "SSTable metadata read time: " << meta_time << " seconds\n";
    std::cout << "Total SSTable data parse time: " << read_time << " seconds\n";
    std::cout << "Total GPU B+Tree build time: " << gpu_time << " seconds\n";
    return 0;
} 