# project (benchmark_insert)
project (benchmark)


include_directories(/usr/local/include)

add_executable(benchmark_insert
    benchmark_insert.cpp
    json_val.cpp
    data_gen.cpp
    secondary_memtable.cpp
    secondary_memtable_iterator.cpp
    secondary_readonlymemtable.cpp
    secondary_readonlymemtable_iterator.cpp
    host_secondary_index_manager.cpp
    gpu_index_host.cpp
    gpu_btree_index_manager.cu
    secondary_index_engine.cpp
)

add_executable(benchmark_query
    benchmark_query.cpp
    json_val.cpp
    data_gen.cpp
    secondary_memtable.cpp
    secondary_memtable_iterator.cpp
    secondary_readonlymemtable.cpp
    secondary_readonlymemtable_iterator.cpp
    host_secondary_index_manager.cpp
    gpu_index_host.cpp
    gpu_btree_index_manager.cu
    secondary_index_engine.cpp
)

target_link_libraries(benchmark_insert PRIVATE gpu_btree leveldb snappy)
target_link_libraries(benchmark_query PRIVATE gpu_btree leveldb snappy)
