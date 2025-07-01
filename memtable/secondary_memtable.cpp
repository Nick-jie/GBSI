#include "secondary_memtable.h"
#include <iostream>
#include <iomanip>
#include "secondary_memtable_iterator.h"

SecondaryMemTable::SecondaryMemTable(int max_lvl)
    : max_level(max_lvl),
      current_level(1),
      entry_cnt(0),
      head(std::make_shared<Node>(0, 0, max_lvl)),
      gen(std::random_device{}()),
      dis(0.0, 1.0) {}

SecondaryMemTable::~SecondaryMemTable() = default;

int SecondaryMemTable::random_level() {
    int lvl = 1;
    while (dis(gen) < 0.5 && lvl < max_level) {
        ++lvl;
    }
    return lvl;
}

void SecondaryMemTable::put(uint32_t attr, uint32_t pkey) {
    std::vector<std::shared_ptr<Node>> update(max_level);
    auto x = head;

    for (int i = current_level - 1; i >= 0; --i) {
        while (x->forward[i] && x->forward[i]->key < attr) {
            x = x->forward[i];
        }
        update[i] = x;
    }

    int lvl = random_level();
    if (lvl > current_level) {
        for (int i = current_level; i < lvl; ++i) {
            update[i] = head;
        }
        current_level = lvl;
    }

    auto new_node = std::make_shared<Node>(attr, pkey, lvl);
    for (int i = 0; i < lvl; ++i) {
        new_node->forward[i] = update[i]->forward[i];
        update[i]->forward[i] = new_node;
    }

    ++entry_cnt;
}

std::size_t SecondaryMemTable::size() const {
    return entry_cnt;
}

void SecondaryMemTable::print() const {
    std::cout << "\n[SecondaryMemTable] total = " << entry_cnt << "\n";
    std::cout << std::fixed << std::setprecision(4);

    for (auto it = begin(); it != end(); ++it) {
        auto [k, v] = *it;
        std::cout << "  " << (k / 10000.0) << "  ->  " << v << '\n';
    }

    std::cout << "---------------------------------------\n";
}

std::vector<uint32_t> SecondaryMemTable::range_query(uint32_t lower, uint32_t upper) const {
    std::vector<uint32_t> results;

    for (auto it = begin(); it != end(); ++it) {
        auto [k, v] = *it;
        if (k >= lower && k <= upper) {
            results.push_back(v);
        } else if (k > upper) {
            break;
        }
    }

    return results;
}

std::vector<uint32_t> SecondaryMemTable::get(uint32_t key) const {
    std::vector<uint32_t> result;
    for (auto it = begin(); it != end(); ++it) {
        auto [k, v] = *it;
        if (k == key) {
            result.push_back(v);
        } else if (k > key) {
            break;
        }
    }
    return result;
}

std::vector<std::pair<uint32_t, uint32_t>> SecondaryMemTable::flush() const {
    std::vector<std::pair<uint32_t, uint32_t>> result;

    for (auto it = begin(); it != end(); ++it) {
        result.push_back(*it);
    }

    return result;
}

secidx::SecondaryMemTableIterator SecondaryMemTable::begin() const {
    return secidx::SecondaryMemTableIterator(head->forward[0]);
}

secidx::SecondaryMemTableIterator SecondaryMemTable::end() const {
    return secidx::SecondaryMemTableIterator();
}

void SecondaryMemTable::clear() {
    
    head = std::make_shared<Node>(0, 0, max_level);
    current_level = 1;
    entry_cnt = 0;
}
