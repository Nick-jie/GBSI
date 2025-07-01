// #ifndef THREAD_TASK_MANAGER_H
// #define THREAD_TASK_MANAGER_H

// #include "ThreadPool.h"
// #include "secondary_memtable.h"
// #include "secondary_readonlymemtable.h"
// #include "gpu_index_converter.h"
// #include <memory>
// #include <future>

// class ThreadTaskManager {
// public:
//     
//     explicit ThreadTaskManager(size_t thread_pool_size = 4);
//     ~ThreadTaskManager();

//     
//     ThreadTaskManager(const ThreadTaskManager&) = delete;
//     ThreadTaskManager& operator=(const ThreadTaskManager&) = delete;

//     
//     std::future<std::shared_ptr<SecondaryReadOnlyMemTable>> 
//     asyncConvertToReadOnly(std::shared_ptr<SecondaryMemTable> memtable);

//     
//     std::future<GpuBTreeInput> 
//     asyncConvertToGpuFormat(std::shared_ptr<SecondaryReadOnlyMemTable> readonly_table);

// private:
//     std::unique_ptr<ThreadPool> thread_pool_;
// };

// #endif // THREAD_TASK_MANAGER_H