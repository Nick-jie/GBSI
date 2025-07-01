#ifndef SECONDARY_MEMTABLE_H
#define SECONDARY_MEMTABLE_H

#include <vector>
#include <memory>
#include <random>
#include <cstddef>
#include <cstdint>
#include "iterator.h"
#include "secondary_memtable_iterator.h"

class SecondaryMemTable {
public:
    explicit SecondaryMemTable(int max_level = 16);
    ~SecondaryMemTable();

    void put(uint32_t attribute_value, uint32_t primary_key);
    std::size_t size() const;
    void print() const;

    std::vector<uint32_t> range_query(uint32_t lower, uint32_t upper) const;
    std::vector<uint32_t> get(uint32_t key) const;

    std::vector<std::pair<uint32_t, uint32_t>> flush() const;
    void clear();

    secidx::SecondaryMemTableIterator begin() const;
    secidx::SecondaryMemTableIterator end() const;

private:
    struct Node {
        uint32_t key;
        uint32_t value;
        std::vector<std::shared_ptr<Node>> forward;

        Node(uint32_t k, uint32_t v, int level)
            : key(k), value(v), forward(level, nullptr) {}
    };

    int max_level;
    int current_level;
    std::size_t entry_cnt;
    std::shared_ptr<Node> head;

    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    int random_level();

    friend class secidx::SecondaryMemTableIterator;
};

#endif
