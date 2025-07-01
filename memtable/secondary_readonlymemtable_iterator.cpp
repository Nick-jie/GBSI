#include "secondary_readonlymemtable_iterator.h"

namespace secidx {

SecondaryReadOnlyMemTableIterator::SecondaryReadOnlyMemTableIterator()
    : iter_() {}

SecondaryReadOnlyMemTableIterator::SecondaryReadOnlyMemTableIterator(InnerIter it)
    : iter_(it) {}

BaseIterator& SecondaryReadOnlyMemTableIterator::operator++() {
    ++iter_;
    return *this;
}

BaseIterator::value_type SecondaryReadOnlyMemTableIterator::operator*() const {
    return *iter_;
}

bool SecondaryReadOnlyMemTableIterator::operator==(const BaseIterator& other) const {
    const auto* o = dynamic_cast<const SecondaryReadOnlyMemTableIterator*>(&other);
    return o && iter_ == o->iter_;
}

bool SecondaryReadOnlyMemTableIterator::operator!=(const BaseIterator& other) const {
    return !(*this == other);
}

IteratorType SecondaryReadOnlyMemTableIterator::get_type() const {
    return IteratorType::SecondaryReadOnlyMemTableIterator;
}

bool SecondaryReadOnlyMemTableIterator::is_end() const {
    return false; 
}

bool SecondaryReadOnlyMemTableIterator::is_valid() const {
    return true; 
}

} 
