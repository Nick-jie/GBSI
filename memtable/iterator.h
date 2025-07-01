#pragma once

#include <cstdint>
#include <utility>

namespace secidx {

enum class IteratorType {
  SecondaryMemTableIterator,
  SecondaryReadOnlyMemTableIterator,
};

class BaseIterator {
public:
  
  using value_type = std::pair<uint32_t, uint32_t>;
  using pointer = value_type*;
  using reference = value_type&;

  virtual ~BaseIterator() = default;

  
  virtual BaseIterator& operator++() = 0;

  
  virtual bool operator==(const BaseIterator& other) const = 0;
  virtual bool operator!=(const BaseIterator& other) const = 0;

  
  virtual value_type operator*() const = 0;

  
  virtual IteratorType get_type() const = 0;

  
  virtual bool is_end() const = 0;
  virtual bool is_valid() const = 0;
};

} // namespace secidx
