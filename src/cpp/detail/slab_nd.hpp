// slab_nd.hpp - ND tensor decomposition and axes redistribution
//
// Internal implementation detail. Use shafft::FFTND, not SlabND directly.
// Manages MPI communicator layouts, local subarray metadata, buffer attachment
// and staged exchanges for multidimensional FFTs.

#ifndef SHAFFT_DETAIL_SLAB_ND_HPP
#define SHAFFT_DETAIL_SLAB_ND_HPP

#include <shafft/detail/slab_base.hpp>
#include <shafft/shafft_enums.hpp>

#include <cstddef>
#include <mpi.h>

namespace shafft {

// Tensor slab decomposition and axes redistribution (internal).
// Contiguous axes (ca) are the locally contiguous, non-distributed axes.
class SlabND : public detail::SlabBase {
public:
  // Construct with explicit Cartesian process grid.
  SlabND(int ndim,
         int size[],
         int commDims[],
         MPI_Datatype mpiSendtype,
         MPI_Comm comm,
         size_t elemSize = 0);

  // Destructor; releases internal MPI/datatype resources.
  ~SlabND();

  // Get ith stage config: subsize, offset, ca arrays. Returns 0 on success.
  int getIthConfig(int* subsize, int* offset, int* ca, int ith) const;

  // Get ith stage layout: subsize, offset arrays. Returns 0 on success.
  int getIthLayout(int* subsize, int* offset, int ith) const;

  // Get ith stage axes: ca and da arrays. Returns 0 on success.
  // Note: Non-noexcept version with different return type; use getIthAxes override for SlabBase

  // Total buffer size required (in elements).
  size_t localAllocSize() const;

  // Attach data and work buffers (SlabBase interface).
  void setBuffers(void* aData, void* bData) noexcept override;

  // Retrieve currently attached buffer pointers (SlabBase interface).
  void getBuffers(void** aData, void** bData) const noexcept override;

  // Swap data and work buffer roles.
  void swapBuffers();

  // Execute forward axes redistribution. Returns 0 on success.
  int forward();

  // Execute backward axes redistribution. Returns 0 on success.
  int backward();

  // Current exchange stage index.
  int esIndex();

  // Number of exchange stages.
  int nes();

  // Get exchange stage indices into es array.
  void getEs(int* es);

  // Copy global extents into size array.
  void getSize(int* size);

  // Copy current local extents into subsize array.
  void getSubsize(int* subsize);

  // Copy current global offsets into offset array.
  void getOffset(int* offset);

  // Current data buffer pointer.
  void* data();

  // Current work buffer pointer.
  void* work();

  // Number of contiguous (non-distributed) axes.
  int nca() const noexcept override;

  // Number of distributed axes.
  int nda() const noexcept override;

  // True if this rank participates in the FFT.
  bool isActive() const noexcept override;

  // Convert flat index to multi-index coordinates.
  void getIndices(int index, int ndim, int* size, int* offset, int* indices);

  // Convert multi-index to flat index.
  int getIndex(int* indices, int ndim, int* size);

  // SlabBase interface implementation

  // Number of dimensions.
  int ndim() const noexcept override { return ndim_; }

  // Global size in each dimension.
  void getSize(size_t* size) const noexcept override {
    if (!size_) {
      for (int i = 0; i < ndim_; ++i)
        size[i] = 0;
      return;
    }
    for (int i = 0; i < ndim_; ++i)
      size[i] = static_cast<size_t>(size_[i]);
  }

  // Allocation size in complex elements.
  size_t allocSize() const noexcept override;

  // Current local subsize.
  void getSubsize(size_t* subsize) const noexcept override {
    if (!subsize_) {
      for (int i = 0; i < ndim_; ++i)
        subsize[i] = 0;
      return;
    }
    for (int i = 0; i < ndim_; ++i)
      subsize[i] = static_cast<size_t>(subsize_[i]);
  }

  // Current local offset.
  void getOffset(size_t* offset) const noexcept override {
    if (!offset_) {
      for (int i = 0; i < ndim_; ++i)
        offset[i] = 0;
      return;
    }
    for (int i = 0; i < ndim_; ++i)
      offset[i] = static_cast<size_t>(offset_[i]);
  }

  // Current contiguous axis indices.
  void getCa(int* ca) const noexcept override;

  // Current distributed axis indices.
  void getDa(int* da) const noexcept override;

  // Get layout at state i.
  int getIthLayout(size_t* subsize, size_t* offset, int i) const noexcept override;

  // Get axes at state i.
  int getIthAxes(int* ca, int* da, int i) const noexcept override;

  // Number of execution steps.
  int numExecSteps() const noexcept override { return nes_; }

  // Update current state.
  void setState(int state) noexcept override { currentState_ = state; }

  // Get current state index.
  int getState() const noexcept override { return currentState_; }

  // Get FFT precision type.
  FFTType precision() const noexcept override { return precision_; }

  // Set FFT precision type.
  void setPrecision(FFTType prec) noexcept { precision_ = prec; }

private:
  int currentState_ = 0;             // Current state index
  FFTType precision_ = FFTType::Z2Z; // FFT precision type
  int taA_;
  int taB_;
  int nes_;
  int ndim_;
  int* size_;
  int* subsize_;
  int* offset_;
  int* subsizes_;
  int* offsets_;
  int nca_;
  int nda_;
  int* caA_;
  int* caB_;
  int* cas_;
  int* das_;
  int* axesA_;
  int* axesB_;
  int* daA_;
  int* daB_;
  int exchangeIndex_ = 0;
  int* es_;
  int esIndex_ = 0;
  int* subsizeA_;
  int* subsizeB_;
  int exchangeDirection_;
  MPI_Datatype* subarrayA_;
  MPI_Datatype* subarrayB_;
  void* aData_ = nullptr;
  void* bData_ = nullptr;
  MPI_Comm* worldcomm_;
  MPI_Comm* comms_;
  MPI_Comm comm_;
  int maxCommSize_;
  MPI_Datatype* subarrays_;

#if SHAFFT_BACKEND_HIPFFT && !SHAFFT_GPU_AWARE_MPI
  // Host-staging buffers for non-GPU-aware MPI
  size_t elemSize_ = 0;
  void* hostA_ = nullptr;
  void* hostB_ = nullptr;
#endif

  // Prepare metadata for forward axes redistribution.
  void prepareForwardExchange();

  // Prepare metadata for backward axes redistribution.
  void prepareBackwardExchange();

  // Execute axes redistribution.
  void doExchange();
};

} // namespace shafft

#endif // SHAFFT_DETAIL_SLAB_ND_HPP
