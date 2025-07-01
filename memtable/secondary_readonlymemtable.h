#ifndef SECONDARY_READONLY_MEMTABLE_H
#define SECONDARY_READONLY_MEMTABLE_H

#include <vector>
#include <utility>
#include <cstdint>
#include "iterator.h"
#include "secondary_readonlymemtable_iterator.h"
#include "gpu_index_host.h"

class SecondaryReadOnlyMemTable {
public:
    explicit SecondaryReadOnlyMemTable(std::vector<std::pair<uint32_t, uint32_t>>&& data);

    std::size_t size() const;
    void print() const;

    std::vector<uint32_t> get(uint32_t key) const;
    std::vector<uint32_t> range_query(uint32_t lower, uint32_t upper) const;

    GpuBTreeInput flush() const;

    
    const std::vector<std::pair<uint32_t, uint32_t>>& get_all_data() const { return data_; }

    secidx::SecondaryReadOnlyMemTableIterator begin() const;
    secidx::SecondaryReadOnlyMemTableIterator end() const;

private:
    std::vector<std::pair<uint32_t, uint32_t>> data_;

    friend class secidx::SecondaryReadOnlyMemTableIterator;
};

#endif
