// slab_1d.hpp - 1D layout adapter implementing SlabBase
//
// Provides SlabBase-compatible interface for 1D distributed FFTs.
// Stores copies of layout values (does not reference external handles).

#ifndef SHAFFT_DETAIL_SLAB_1D_HPP
#define SHAFFT_DETAIL_SLAB_1D_HPP

#include <shafft/detail/slab_base.hpp>

#include <cstddef> // size_t

namespace shafft::detail {

// 1D layout adapter for SlabBase interface.
class Slab1D : public SlabBase {
private:
  // Global size
  size_t N_ = 0;

  // INITIAL layout
  size_t localNInit_ = 0;
  size_t localStartInit_ = 0;

  // REDISTRIBUTED layout
  size_t localNTrans_ = 0;
  size_t localStartTrans_ = 0;

  // Allocation size
  size_t alloc_size_ = 0;

  // Precision
  FFTType precision_ = FFTType::C2C;

  // Current state (0 = INITIAL, 1 = REDISTRIBUTED)
  int currentState_ = 0;

  // Buffers
  void* data_ = nullptr;
  void* work_ = nullptr;

  bool initialized_ = false;

public:
  Slab1D() = default;
  ~Slab1D() override = default;

  // Copyable (stores values, not pointers)
  Slab1D(const Slab1D&) = default;
  Slab1D& operator=(const Slab1D&) = default;
  Slab1D(Slab1D&&) = default;
  Slab1D& operator=(Slab1D&&) = default;

  // Initialization - copy values from handle

  // Initialize with layout values (typically extracted from FFT1DHandle)
  void init(size_t globalN,
            size_t localNInit,
            size_t localStartInit,
            size_t localNTrans,
            size_t localStartTrans,
            size_t localAllocSize,
            FFTType precision) noexcept {
    N_ = globalN;
    localNInit_ = localNInit;
    localStartInit_ = localStartInit;
    localNTrans_ = localNTrans;
    localStartTrans_ = localStartTrans;
    alloc_size_ = localAllocSize;
    precision_ = precision;
    currentState_ = 0;
    initialized_ = true;
  }

  // Check if Slab1D is properly initialized
  bool isInitialized() const noexcept { return initialized_; }

  // SlabBase interface - dimension queries

  int ndim() const noexcept override { return 1; }

  void getSize(size_t* size) const noexcept override {
    if (size)
      size[0] = N_;
  }

  size_t allocSize() const noexcept override { return alloc_size_; }

  // Current layout (depends on state)

  void getSubsize(size_t* subsize) const noexcept override {
    if (!subsize)
      return;
    subsize[0] = (currentState_ == 0) ? localNInit_ : localNTrans_;
  }

  void getOffset(size_t* offset) const noexcept override {
    if (!offset)
      return;
    offset[0] = (currentState_ == 0) ? localStartInit_ : localStartTrans_;
  }

  // For 1D, the single axis is distributed: nca=0, nda=1
  int nca() const noexcept override { return 0; }
  int nda() const noexcept override { return 1; }

  // No contiguous axes for 1D
  void getCa(int* /*ca*/) const noexcept override {}

  // The single axis (0) is distributed
  void getDa(int* da) const noexcept override {
    if (da)
      da[0] = 0;
  }

  // Layout at specific state (TensorLayout support)

  int getIthLayout(size_t* subsize, size_t* offset, int i) const noexcept override {
    if (!subsize || !offset)
      return -1;

    if (i == 0) { // INITIAL
      subsize[0] = localNInit_;
      offset[0] = localStartInit_;
    } else { // REDISTRIBUTED (any i > 0 maps to redistributed for 1D)
      subsize[0] = localNTrans_;
      offset[0] = localStartTrans_;
    }
    return 0;
  }

  int getIthAxes(int* /*ca*/, int* da, int /*i*/) const noexcept override {
    // No contiguous axes, single distributed axis (0)
    if (da)
      da[0] = 0;
    return 0;
  }

  int numExecSteps() const noexcept override { return 1; }

  // Buffer management

  void setBuffers(void* data, void* work) noexcept override {
    data_ = data;
    work_ = work;
  }

  void getBuffers(void** data, void** work) const noexcept override {
    if (data)
      *data = data_;
    if (work)
      *work = work_;
  }

  // Swap data and work pointers (called after execute to keep result in data_)
  void swapBuffers() noexcept {
    void* tmp = data_;
    data_ = work_;
    work_ = tmp;
  }

  // State management

  void setState(int state) noexcept override { currentState_ = state; }
  int getState() const noexcept override { return currentState_; }

  // Other queries

  bool isActive() const noexcept override {
    // All ranks are active in a 1D distributed FFT.  Even ranks with
    // localN == 0 (no original data) must participate in MPI collectives
    // (MPI_Alltoall) during the Cooley-Tukey / Bluestein algorithm, so
    // they need valid buffers and cannot be skipped.
    return initialized_;
  }

  FFTType precision() const noexcept override { return precision_; }
};

} // namespace shafft::detail

#endif // SHAFFT_DETAIL_SLAB_1D_HPP
