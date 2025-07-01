#ifndef SECONDARY_READONLY_MEMTABLE_ITERATOR_H
#define SECONDARY_READONLY_MEMTABLE_ITERATOR_H

#include "iterator.h"
#include <vector>
#include <cstdint>

namespace secidx {

class SecondaryReadOnlyMemTableIterator : public BaseIterator {
public:
    using InnerIter = std::vector<std::pair<uint32_t, uint32_t>>::const_iterator;

    SecondaryReadOnlyMemTableIterator();
    explicit SecondaryReadOnlyMemTableIterator(InnerIter it);

    BaseIterator& operator++() override;
    value_type operator*() const override;

    bool operator==(const BaseIterator& other) const override;
    bool operator!=(const BaseIterator& other) const override;

    IteratorType get_type() const override;
    bool is_end() const override;
    bool is_valid() const override;

private:
    InnerIter iter_;
};

} 

#endif
