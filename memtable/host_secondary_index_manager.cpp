#include "host_secondary_index_manager.h"
#include "thread_task_manager.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

HostSecondaryIndexManager::HostSecondaryIndexManager(std::size_t threshold, size_t thread_pool_size)
    : thread_pool_(std::make_unique<ThreadPool>(thread_pool_size)),
      active_memtable_(std::make_unique<SecondaryMemTable>()),
      flush_threshold_(threshold),
      gpu_btree_(std::make_unique<GpuBTreeIndexManager>()) {}

HostSecondaryIndexManager::~HostSecondaryIndexManager() {
    
    for (auto& future : pending_flushes_) {
        if (future.valid()) {
            future.wait();
        }
    }
    for (auto& future : pending_gpu_conversions_) {
        if (future.valid()) {
            future.wait();
        }
    }
    for (auto& future : pending_gpu_inserts_) {
        if (future.valid()) {
            future.wait();
        }
    }
}

void HostSecondaryIndexManager::insert(uint32_t attr, uint32_t pkey) {
    active_memtable_->put(attr, pkey);
    check_and_trigger_flush();
}

void HostSecondaryIndexManager::check_and_trigger_flush() {
    if (active_memtable_->size() >= flush_threshold_) {
        force_flush();
    }
}

void HostSecondaryIndexManager::force_flush() {
    
    auto new_memtable = std::make_unique<SecondaryMemTable>();
    
    auto old_table = std::shared_ptr<SecondaryMemTable>(active_memtable_.release());
    
    active_memtable_ = std::move(new_memtable);
    
    
    async_flush(old_table);
}

void HostSecondaryIndexManager::async_flush(std::shared_ptr<SecondaryMemTable> old_table) {
    
    pending_flushes_.erase(
        std::remove_if(pending_flushes_.begin(), pending_flushes_.end(),
            [](auto& future) {
                return future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            }
        ),
        pending_flushes_.end()
    );

    
    auto future = thread_pool_->enqueue([this, old_table]() {
        
        auto flushed_data = old_table->flush();
        auto readonly_table = std::make_shared<SecondaryReadOnlyMemTable>(std::move(flushed_data));
        
        
        readonly_tables_.push_back(readonly_table);
        
        std::cout << "[Async Flush] Completed a SecondaryIndexMemtable→ReadonlyMemtable conversion, current number of readonly tables: " 
                  << readonly_tables_.size() << std::endl;

        
        old_table->clear();

        
        async_convert_to_gpu(readonly_table);
    });

    
    pending_flushes_.push_back(std::move(future));
}

void HostSecondaryIndexManager::print_all() const {
    std::cout << "\n[HostSecondaryIndexManager] === SecondaryIndexMemtable ===\n";
    active_memtable_->print();

    for (size_t i = 0; i < readonly_tables_.size(); ++i) {
        std::cout << "\n[HostSecondaryIndexManager] === ReadonlyMemtable" << i << " ===\n";
        readonly_tables_[i]->print();
    }
}

std::vector<uint32_t> HostSecondaryIndexManager::query(uint32_t attr) const {
    std::vector<uint32_t> result;

    
    if (active_memtable_) {
        auto memtable_results = active_memtable_->get(attr);
        result.insert(result.end(), memtable_results.begin(), memtable_results.end());
    }
    
    
    for (const auto& table : readonly_tables_) {
        auto readonly_results = table->get(attr);
        result.insert(result.end(), readonly_results.begin(), readonly_results.end());
    }
    
    
    
    for (const auto& future : pending_gpu_inserts_) {
        if (future.valid()) {
            future.wait();
        }
    }
    
    if (gpu_btree_) {
        uint32_t gpu_result = gpu_btree_->search(attr);
        if (gpu_result != 0) {  
            result.push_back(gpu_result);
        }
    }

    return result;
}

std::vector<uint32_t> HostSecondaryIndexManager::range_query(uint32_t lower, uint32_t upper) const {
    std::vector<uint32_t> result;

    
    if (active_memtable_) {
        auto memtable_results = active_memtable_->range_query(lower, upper);
        result.insert(result.end(), memtable_results.begin(), memtable_results.end());
    }
    
    
    for (const auto& table : readonly_tables_) {
        auto readonly_results = table->range_query(lower, upper);
        result.insert(result.end(), readonly_results.begin(), readonly_results.end());
    }
    
    
    
    for (const auto& future : pending_gpu_inserts_) {
        if (future.valid()) {
            future.wait();
        }
    }
    
    if (gpu_btree_) {
        auto gpu_results = gpu_btree_->range_query(lower, upper);
        for (const auto& [key, value] : gpu_results) {
            if (value != 0) {  
                result.push_back(value);
    }
        }
    }

    return result;
}

void HostSecondaryIndexManager::print_status() const {
    std::cout << "\n[HostSecondaryIndexManager] Status Report:\n";
    std::cout << "----------------------------------------\n";
    std::cout << "SecondaryIndexMemtable size: " << memtable_size() << "/" << flush_threshold_ << "\n";
    std::cout << "Number of ReadonlyMemtables: " << readonly_table_count() << "\n";
    std::cout << "Pending flush tasks: " << pending_flushes_.size() << "\n";
    std::cout << "Pending GPU conversion tasks: " << pending_gpu_conversions_.size() << "\n";
    std::cout << "Pending GPU insert tasks: " << pending_gpu_inserts_.size() << "\n";
    std::cout << "GPU B-tree memory usage: " << std::fixed << std::setprecision(3) 
              << gpu_btree_->memory_usage() << " GiB\n";
    std::cout << "----------------------------------------\n";
}

std::size_t HostSecondaryIndexManager::memtable_size() const {
    return active_memtable_->size();
}

std::size_t HostSecondaryIndexManager::readonly_table_count() const {
    return readonly_tables_.size();
}

void HostSecondaryIndexManager::async_convert_to_gpu(std::shared_ptr<SecondaryReadOnlyMemTable> readonly_table) {
    
    pending_gpu_conversions_.erase(
        std::remove_if(pending_gpu_conversions_.begin(), pending_gpu_conversions_.end(),
            [](auto& future) {
                return future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            }
        ),
        pending_gpu_conversions_.end()
    );

    
    auto future = thread_pool_->enqueue([this, readonly_table]() {
        
        GpuBTreeInput gpu_input;
        
        
        const auto& data = readonly_table->get_all_data();
        
        
        gpu_input.keys.reserve(data.size());
        gpu_input.values.reserve(data.size());
        
        
        for (const auto& [key, value] : data) {
            gpu_input.keys.push_back(key);
            gpu_input.values.push_back(value);
        }
        
        
        if (gpu_input.validate()) {
            
            gpu_inputs_.push_back(gpu_input);  
            
            std::cout << "[GPU Conversion] Completed a ReadOnlyMemTable→gpu_index_host conversion, current number of gpu_index_host data blocks: " 
                      << gpu_inputs_.size() << std::endl;
            
           
            gpu_inputs_.back().print_stats();

            
            readonly_tables_.erase(
                std::remove(readonly_tables_.begin(), readonly_tables_.end(), readonly_table),
                readonly_tables_.end()
            );

            
            async_insert_to_gpu_btree(std::move(gpu_input));
        } else {
            std::cerr << "[GPU Conversion] Failed!" << std::endl;
        }
    });

    
    pending_gpu_conversions_.push_back(std::move(future));
}

void HostSecondaryIndexManager::async_insert_to_gpu_btree(GpuBTreeInput gpu_input) {
    
    pending_gpu_inserts_.erase(
        std::remove_if(pending_gpu_inserts_.begin(), pending_gpu_inserts_.end(),
            [](auto& future) {
                return future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            }
        ),
        pending_gpu_inserts_.end()
    );

    
    auto future = thread_pool_->enqueue([this, gpu_input = std::move(gpu_input)]() mutable {
        std::cout << "[GPU B-tree] Start inserting data block, size: " << gpu_input.size() << std::endl;
        
        
        gpu_btree_->insert_from_input(gpu_input);
        
        std::cout << "[GPU B-tree] Data block insertion completed, current memory usage: " 
                  << std::fixed << std::setprecision(3) 
                  << gpu_btree_->memory_usage() << " GiB" << std::endl;

        
        gpu_inputs_.erase(
            std::remove_if(gpu_inputs_.begin(), gpu_inputs_.end(),
                [&gpu_input](const GpuBTreeInput& input) {
                    return input.keys == gpu_input.keys && input.values == gpu_input.values;
                }
            ),
            gpu_inputs_.end()
        );

        
        gpu_input.keys.clear();
        gpu_input.values.clear();
    });

    
    pending_gpu_inserts_.push_back(std::move(future));
}
