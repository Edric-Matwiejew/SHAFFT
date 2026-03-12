// array_utils.hpp - Common array utility functions
//
// Provides modern C++17 iterator-based product operations to consolidate
// duplicated inline loops across the codebase.

#ifndef SHAFFT_DETAIL_ARRAY_UTILS_HPP
#define SHAFFT_DETAIL_ARRAY_UTILS_HPP

#include <cstddef>
#include <iterator>

namespace shafft::detail {

//==============================================================================
// Product operations
//==============================================================================

/// Product of elements in range [first, last).
/// @tparam Iter Iterator type
/// @tparam U Accumulator type (defaults to iterator's value_type)
/// @param first Iterator to first element
/// @param last Iterator past last element
/// @param init Initial value (default 1)
/// @return Product of all elements in range, cast to U
template <typename Iter, typename U = typename std::iterator_traits<Iter>::value_type>
constexpr U product(Iter first, Iter last, U init = U{1}) noexcept {
  for (; first != last; ++first)
    init *= static_cast<U>(*first);
  return init;
}

/// Product of first n elements of array.
/// Equivalent to product(arr, arr + n).
/// @tparam T Element type
/// @tparam U Accumulator type (defaults to T)
/// @param arr Pointer to array
/// @param n Number of elements
/// @return Product of first n elements
template <typename T, typename U = T>
constexpr U prodN(const T* arr, int n) noexcept {
  return (n > 0) ? product(arr, arr + n, U{1}) : U{1};
}

/// Product of elements in index range [i0, i1] (inclusive).
/// Returns 1 if i0 > i1.
/// @tparam T Element type
/// @tparam U Accumulator type (defaults to T)
/// @param arr Pointer to array
/// @param i0 Start index (inclusive)
/// @param i1 End index (inclusive)
/// @return Product of elements arr[i0] * arr[i0+1] * ... * arr[i1]
template <typename T, typename U = T>
constexpr U prodRange(const T* arr, int i0, int i1) noexcept {
  return (i0 <= i1) ? product(arr + i0, arr + i1 + 1, U{1}) : U{1};
}

} // namespace shafft::detail

#endif // SHAFFT_DETAIL_ARRAY_UTILS_HPP
