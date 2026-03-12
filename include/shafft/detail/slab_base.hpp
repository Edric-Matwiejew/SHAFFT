/**
 * @file slab_base.hpp
 * @brief Abstract interface for FFT layout tracking.
 *
 * SlabBase provides a unified interface for querying tensor layouts in
 * distributed FFT plans. Uses size_t for element counts (can exceed INT_MAX)
 * and int for axis indices (always small).
 */

#ifndef SHAFFT_DETAIL_SLAB_BASE_HPP
#define SHAFFT_DETAIL_SLAB_BASE_HPP

#include "../shafft_enums.hpp" // FFTType, TensorLayout

#include <cstddef> // size_t

namespace shafft::detail {

/**
 * @brief Abstract interface for FFT layout tracking.
 *
 * Implemented by Slab (ND) and Slab1D.
 */
class SlabBase {
public:
  virtual ~SlabBase() = default;

  /// Number of dimensions (always small, use int)
  [[nodiscard]] virtual int ndim() const noexcept = 0;

  /// Global size in each dimension (use size_t for 1D support)
  /// @param[out] size Array of length ndim()
  virtual void getSize(size_t* size) const noexcept = 0;

  /// Allocation size in complex elements (max across all layouts)
  [[nodiscard]] virtual size_t allocSize() const noexcept = 0;

  /// Get current local subsize (use size_t for 1D support)
  /// @param[out] subsize Array of length ndim()
  virtual void getSubsize(size_t* subsize) const noexcept = 0;

  /// Get current local offset (use size_t for 1D support)
  /// @param[out] offset Array of length ndim()
  virtual void getOffset(size_t* offset) const noexcept = 0;

  /// Number of contiguous (non-distributed) axes
  [[nodiscard]] virtual int nca() const noexcept = 0;

  /// Number of distributed axes
  [[nodiscard]] virtual int nda() const noexcept = 0;

  /// Get current contiguous axis indices (indices are always int)
  /// @param[out] ca Array of length nca()
  virtual void getCa(int* ca) const noexcept = 0;

  /// Get current distributed axis indices (indices are always int)
  /// @param[out] da Array of length nda()
  virtual void getDa(int* da) const noexcept = 0;

  /// Get layout at state i (i=0 is INITIAL, i=numExecSteps() is REDISTRIBUTED)
  /// @param[out] subsize Array of length ndim()
  /// @param[out] offset Array of length ndim()
  /// @param[in] i State index
  /// @return 0 on success, non-zero on error
  virtual int getIthLayout(size_t* subsize, size_t* offset, int i) const noexcept = 0;

  /// Get axes at state i (indices are always int)
  /// @param[out] ca Contiguous axes
  /// @param[out] da Distributed axes
  /// @param[in] i State index
  /// @return 0 on success, non-zero on error
  virtual int getIthAxes(int* ca, int* da, int i) const noexcept = 0;

  /// Number of execution steps (for state indexing, always small)
  [[nodiscard]] virtual int numExecSteps() const noexcept = 0;

  /// Attach data and work buffers
  virtual void setBuffers(void* data, void* work) noexcept = 0;

  /// Retrieve currently attached buffer pointers
  virtual void getBuffers(void** data, void** work) const noexcept = 0;

  /// Update current state (called after execution)
  virtual void setState(int state) noexcept = 0;

  /// Get current state index
  [[nodiscard]] virtual int getState() const noexcept = 0;

  /// Check if this rank is active (has work to do)
  [[nodiscard]] virtual bool isActive() const noexcept = 0;

  /// Get FFT precision type
  [[nodiscard]] virtual shafft::FFTType precision() const noexcept = 0;
};

} // namespace shafft::detail

#endif // SHAFFT_DETAIL_SLAB_BASE_HPP
