#ifndef HOST_SECONDARY_INDEX_MANAGER_H
#define HOST_SECONDARY_INDEX_MANAGER_H

#include <vector>
#include <memory>
#include <cstdint>
#include <future>
#include "secondary_memtable.h"
#include "secondary_readonlymemtable.h"
#include "ThreadPool.h"
#include "gpu_index_host.h"
#include "gpu_btree_index_manager.h"

class HostSecondaryIndexManager {
public:
    
    explicit HostSecondaryIndexManager(std::size_t threshold = 1000, 
                                     size_t thread_pool_size = 4);
    ~HostSecondaryIndexManager();

    
    HostSecondaryIndexManager(const HostSecondaryIndexManager&) = delete;
    HostSecondaryIndexManager& operator=(const HostSecondaryIndexManager&) = delete;

    
    void insert(uint32_t attr, uint32_t pkey);
    
    
    std::vector<uint32_t> query(uint32_t attr) const;  
    std::vector<uint32_t> range_query(uint32_t lower, uint32_t upper) const;  
    
    
    void print_all() const;                    
    std::size_t memtable_size() const;        
    std::size_t readonly_table_count() const;  
    void print_status() const;                 

    
    void force_flush();  

    
    const std::vector<GpuBTreeInput>& get_gpu_inputs() const { return gpu_inputs_; }

    
    const GpuBTreeIndexManager* get_gpu_btree() const { return gpu_btree_.get(); }

    
    std::vector<std::pair<uint32_t, uint32_t>> gpu_range_query(uint32_t lower, uint32_t upper) const {
        if (!gpu_btree_) {
            return std::vector<std::pair<uint32_t, uint32_t>>();
        }
        
        wait_for_pending_operations();
        return gpu_btree_->range_query(lower, upper);
    }

    
    std::vector<std::pair<uint32_t, uint32_t>> get_all_readonly_data() const {
        std::vector<std::pair<uint32_t, uint32_t>> all_data;
        for (const auto& table : readonly_tables_) {
            const auto& table_data = table->get_all_data();
            all_data.insert(all_data.end(), table_data.begin(), table_data.end());
        }
        return all_data;
    }

    
    void wait_for_pending_operations() const {
        
        for (const auto& future : pending_flushes_) {
            if (future.valid()) {
                future.wait();
            }
        }
        
        for (const auto& future : pending_gpu_conversions_) {
            if (future.valid()) {
                future.wait();
            }
        }
        
        for (const auto& future : pending_gpu_inserts_) {
            if (future.valid()) {
                future.wait();
            }
        }
    }

private:
    
    void check_and_trigger_flush();  
    void async_flush(std::shared_ptr<SecondaryMemTable> old_table);  
    void async_convert_to_gpu(std::shared_ptr<SecondaryReadOnlyMemTable> readonly_table);  
    void async_insert_to_gpu_btree(GpuBTreeInput gpu_input);  

    
    std::unique_ptr<ThreadPool> thread_pool_;              
    std::unique_ptr<SecondaryMemTable> active_memtable_;   
    std::vector<std::shared_ptr<SecondaryReadOnlyMemTable>> readonly_tables_;  
    std::size_t flush_threshold_;                          
    std::vector<std::future<void>> pending_flushes_;      
    std::vector<GpuBTreeInput> gpu_inputs_;               
    std::vector<std::future<void>> pending_gpu_conversions_;  
    std::unique_ptr<GpuBTreeIndexManager> gpu_btree_;     
    std::vector<std::future<void>> pending_gpu_inserts_;  
};

#endif
