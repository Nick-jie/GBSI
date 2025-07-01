#include "secondary_memtable_iterator.h"
#include "secondary_memtable.h"

namespace secidx {


struct SecondaryMemTableIterator::Node {
    uint32_t key;
    uint32_t value;
    std::vector<std::shared_ptr<Node>> forward;
};

SecondaryMemTableIterator::SecondaryMemTableIterator()
    : current(nullptr) {}

SecondaryMemTableIterator::SecondaryMemTableIterator(std::shared_ptr<void> node_ptr)
    : current(std::static_pointer_cast<Node>(node_ptr)) {}

BaseIterator& SecondaryMemTableIterator::operator++() {
    if (current) current = current->forward[0];
    return *this;
}

BaseIterator::value_type SecondaryMemTableIterator::operator*() const {
    return { current->key, current->value };
}

bool SecondaryMemTableIterator::operator==(const BaseIterator& other) const {
    const auto* o = dynamic_cast<const SecondaryMemTableIterator*>(&other);
    return o && current == o->current;
}

bool SecondaryMemTableIterator::operator!=(const BaseIterator& other) const {
    return !(*this == other);
}

IteratorType SecondaryMemTableIterator::get_type() const {
    return IteratorType::SecondaryMemTableIterator;
}

bool SecondaryMemTableIterator::is_end() const {
    return current == nullptr;
}

bool SecondaryMemTableIterator::is_valid() const {
    return current != nullptr;
}

} 
