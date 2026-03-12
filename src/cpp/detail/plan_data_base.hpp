// plan_data_base.hpp - Abstract base class for FFT plan data
//
// Provides unified interface for ND and 1D FFT plans including
// normalization logic (unitary scaling: 1/sqrt(N) after each execute).

#ifndef SHAFFT_DETAIL_PLAN_DATA_BASE_HPP
#define SHAFFT_DETAIL_PLAN_DATA_BASE_HPP

#include <shafft/detail/slab_base.hpp>
#include <shafft/shafft_types.hpp> // FFTType, TensorLayout, TransformLayout

#include "array_utils.hpp"

#include <cmath>   // std::sqrt
#include <cstddef> // size_t
#include <mpi.h>

namespace shafft::detail {

// Abstract base for FFT plan data. Derived: FFTNDPlan, FFT1DPlan.
class PlanBase {
public:
  virtual ~PlanBase() = default;

  // Layout access (delegated to SlabBase)

  // Get the underlying slab (layout tracker)
  virtual SlabBase* slab() noexcept = 0;
  virtual const SlabBase* slab() const noexcept = 0;

  // Normalization (NVI: base handles inactive-rank guard)

  // Apply normalization to data buffer. Returns 0 on success.
  // Inactive ranks silently reset the exponent and succeed.
  int normalize() noexcept {
    if (activeComm_ == MPI_COMM_NULL) {
      norm_exponent_ = 0;
      return 0;
    }
    return normalizeImpl();
  }

  // Lifecycle management (pure virtual - type-specific cleanup)

  // Release all internal resources.
  virtual void release() noexcept = 0;

  // Execution (NVI: base handles inactive-rank guard)

  // Execute FFT. Returns 0 on success.
  // Inactive ranks silently increment the norm exponent and succeed.
  int execute(FFTDirection direction) noexcept {
    if (activeComm_ == MPI_COMM_NULL) {
      norm_exponent_ += 1;
      return 0;
    }
    return executeImpl(direction);
  }

  // Stream management (backend-specific, pure virtual)

  // Set the execution stream (hipStream_t for GPU, nullptr for CPU)
  virtual void setStream(void* stream) = 0;

  // Get the current execution stream
  virtual void* getStream() const = 0;

  // State queries

  // Check if this plan is active (has work to do).
  // A rank is active when it has a non-zero allocation size.
  // Before configure, all ranks are considered inactive.
  bool isActive() const noexcept { return activeComm_ != MPI_COMM_NULL; }

  // Check if this plan is configured (init succeeded)
  virtual bool isConfigured() const noexcept = 0;

  // Check if this plan is planned (backend plans created)
  virtual bool isPlanned() const noexcept = 0;

  // Get the active subcommunicator (MPI_COMM_NULL on inactive ranks).
  // Valid after configure. getCommunicator() dups this for the user.
  MPI_Comm getComm() const noexcept { return activeComm_; }

  // Create backend FFT plans (called by shafftPlan / plan())
  virtual int createPlans() noexcept = 0;

  // Configure the forward output policy used for layout-query semantics.
  void setOutputLayoutPolicy(TransformLayout policy) noexcept { output_layout_policy_ = policy; }

  // Return configured forward output policy for this plan.
  [[nodiscard]] TransformLayout outputLayoutPolicy() const noexcept {
    return output_layout_policy_;
  }

  // Get allocation size in complex elements
  size_t allocSize() const noexcept {
    const SlabBase* s = slab();
    return s ? s->allocSize() : 0;
  }

  // Get the global FFT size (product of all dimensions)
  size_t globalSize() const noexcept {
    const SlabBase* s = slab();
    if (!s)
      return 0;
    int nd = s->ndim();
    size_t sizes[16]; // Max 16 dimensions should be plenty
    s->getSize(sizes);
    return prodN<size_t, size_t>(sizes, nd);
  }

  // Get number of dimensions
  int ndim() const noexcept {
    const SlabBase* s = slab();
    return s ? s->ndim() : 0;
  }

  // Get FFT precision type
  FFTType precision() const noexcept {
    const SlabBase* s = slab();
    return s ? s->precision() : FFTType::C2C;
  }

  // Buffer management (delegated to SlabBase)

  // Set data and work buffers
  void setBuffers(void* data, void* work) noexcept {
    SlabBase* s = slab();
    if (s)
      s->setBuffers(data, work);
  }

  // Get data and work buffer pointers
  void getBuffers(void** data, void** work) const noexcept {
    const SlabBase* s = slab();
    if (s) {
      s->getBuffers(data, work);
    } else {
      if (data)
        *data = nullptr;
      if (work)
        *work = nullptr;
    }
  }

  // Layout queries (delegated to SlabBase)

  // Get local layout (subsize/offset) for given layout state.
  int getLayout(size_t* subsize, size_t* offset, TensorLayout layout) const noexcept {
    const SlabBase* s = slab();
    if (!s)
      return -1;

    switch (layout) {
    case TensorLayout::CURRENT:
      s->getSubsize(subsize);
      s->getOffset(offset);
      return 0;
    case TensorLayout::INITIAL:
      return s->getIthLayout(subsize, offset, 0);
    case TensorLayout::REDISTRIBUTED:
      return s->getIthLayout(
          subsize,
          offset,
          output_layout_policy_ == TransformLayout::INITIAL ? 0 : s->numExecSteps());
    }
    return 0;
  }

  // Get axis distribution (contiguous/distributed) for given layout state.
  int getAxes(int* ca, int* da, TensorLayout layout) const noexcept {
    const SlabBase* s = slab();
    if (!s)
      return -1;

    switch (layout) {
    case TensorLayout::CURRENT:
      s->getCa(ca);
      s->getDa(da);
      return 0;
    case TensorLayout::INITIAL:
      return s->getIthAxes(ca, da, 0);
    case TensorLayout::REDISTRIBUTED:
      return s->getIthAxes(
          ca, da, output_layout_policy_ == TransformLayout::INITIAL ? 0 : s->numExecSteps());
    }
    return 0;
  }

  /// Create the active subcommunicator via MPI_Comm_split.
  /// Collective on @p origComm.  Active ranks (allocSize > 0) join
  /// colour 0; inactive ranks receive MPI_COMM_NULL.
  /// Call once, after the slab/layout is initialised in configure.
  int initActiveComm(MPI_Comm origComm) noexcept {
    bool active = allocSize() > 0;
    int rank = 0;
    int rc = MPI_Comm_rank(origComm, &rank);
    if (rc != MPI_SUCCESS)
      return rc;
    return MPI_Comm_split(origComm, active ? 0 : MPI_UNDEFINED, rank, &activeComm_);
  }

  /// Free the active subcommunicator (safe to call when already null).
  void freeActiveComm() noexcept {
    if (activeComm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&activeComm_);
      activeComm_ = MPI_COMM_NULL;
    }
  }

protected:
  // Type-specific execute/normalize implementations (called only on active ranks).
  virtual int executeImpl(FFTDirection direction) noexcept = 0;
  virtual int normalizeImpl() noexcept = 0;

  TransformLayout output_layout_policy_ = TransformLayout::REDISTRIBUTED;
  MPI_Comm activeComm_ = MPI_COMM_NULL; ///< Subcommunicator of active ranks.
  int norm_exponent_ = 0;               ///< Pending normalization exponent.
};

} // namespace shafft::detail

#endif // SHAFFT_DETAIL_PLAN_DATA_BASE_HPP
