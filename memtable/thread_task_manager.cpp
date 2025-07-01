// #include "thread_task_manager.h"

// ThreadTaskManager::ThreadTaskManager(size_t thread_pool_size)
//     : thread_pool_(std::make_unique<ThreadPool>(thread_pool_size)) {}

// ThreadTaskManager::~ThreadTaskManager() = default;

// std::future<std::shared_ptr<SecondaryReadOnlyMemTable>>
// ThreadTaskManager::asyncConvertToReadOnly(std::shared_ptr<SecondaryMemTable> memtable) {
//     return thread_pool_->enqueue([memtable]() {
//         
//         auto data = memtable->flush();
        
//         
//         return std::make_shared<SecondaryReadOnlyMemTable>(std::move(data));
//     });
// }

// std::future<GpuBTreeInput>
// ThreadTaskManager::asyncConvertToGpuFormat(std::shared_ptr<SecondaryReadOnlyMemTable> readonly_table) {
//     return thread_pool_->enqueue([readonly_table]() {
//         
//         return convert_to_btree_input(*readonly_table);
//     });
// }
