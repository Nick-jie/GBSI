#ifndef SECONDARY_MEMTABLE_ITERATOR_H
#define SECONDARY_MEMTABLE_ITERATOR_H

#include "iterator.h"
#include <memory>
#include <vector>
#include <cstdint>

class SecondaryMemTable; 

namespace secidx {


class SecondaryMemTableIterator : public BaseIterator {
public:
    SecondaryMemTableIterator();
    explicit SecondaryMemTableIterator(std::shared_ptr<void> node_ptr);

    BaseIterator& operator++() override;
    value_type operator*() const override;
    bool operator==(const BaseIterator& other) const override;
    bool operator!=(const BaseIterator& other) const override;

    IteratorType get_type() const override;
    bool is_end() const override;
    bool is_valid() const override;

private:
    struct Node;
    std::shared_ptr<Node> current;
};

} 

#endif
