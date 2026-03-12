#ifndef SHAFFT_INTERNAL_HPP
#define SHAFFT_INTERNAL_HPP

#include "detail/error_macros.hpp"
#include "detail/plan_1d_data.hpp"
#include "detail/plan_data_base.hpp"
#include "detail/plan_nd_data.hpp"
#include "detail/slab_nd.hpp"
#include <shafft/shafft_config.h>

#include <cstdlib>
#include <mpi.h>

#if SHAFFT_BACKEND_HIPFFT
#include <hip/hip_runtime.h>
#endif

struct FFTNDHandle;

namespace shafft::detail {

// FFTND (N-dimensional distributed FFT) functions

#if SHAFFT_BACKEND_HIPFFT
int setStream(shafft::detail::FFTNDPlan* plan, hipStream_t stream) noexcept;
#endif

int planCreate(shafft::detail::FFTNDPlan** plan) noexcept;

template <typename T>
int setBuffers(shafft::detail::FFTNDPlan* plan, T* data, T* work) noexcept {
  try {
    if (!plan)
      return (int)shafft::Status::ERR_NULLPTR;
    if (!plan->slab_)
      return (int)shafft::Status::ERR_PLAN_NOT_INIT;
    const bool inactive = !plan->slab_->isActive() || plan->slab_->allocSize() == 0;
    if (inactive) {
      plan->slab_->setBuffers(nullptr, nullptr);
      return (int)shafft::Status::SUCCESS;
    }
    if (!data || !work)
      return (int)shafft::Status::ERR_NULLPTR;

    plan->slab_->setBuffers(static_cast<void*>(data), static_cast<void*>(work));
    return static_cast<int>(shafft::Status::SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

template <typename T>
int getBuffers(shafft::detail::FFTNDPlan* plan, T** data, T** work) noexcept {
  try {
    if (!plan)
      return (int)shafft::Status::ERR_NULLPTR;
    if (!plan->slab_)
      return (int)shafft::Status::ERR_PLAN_NOT_INIT;
    if (!data || !work)
      return (int)shafft::Status::ERR_NULLPTR;

    const bool inactive = !plan->slab_->isActive() || plan->slab_->allocSize() == 0;
    void* d = nullptr;
    void* w = nullptr;
    plan->slab_->getBuffers(&d, &w);

    // Inactive ranks may have null buffers (consistent with setBuffers behavior)
    if (!inactive && (!d || !w))
      return (int)shafft::Status::ERR_NO_BUFFER;

    *data = static_cast<T*>(d);
    *work = static_cast<T*>(w);
    return static_cast<int>(shafft::Status::SUCCESS);
  }
  SHAFFT_CATCH_RETURN();
}

int configurationND(int ndim,
                    int* size,
                    shafft::FFTType precision,
                    int* commDims,
                    int* nda,
                    int* subsize,
                    int* offset,
                    int* commSize,
                    shafft::DecompositionStrategy strategy,
                    size_t memLimit,
                    MPI_Comm comm) noexcept;

// Configure ND plan: create slab decomposition but defer backend plan creation.
// After this call, allocSize() and layout queries are available.
int planNDConfigure(shafft::detail::FFTNDPlan* plan,
                    int ndim,
                    int commDims[],
                    int dimensions[],
                    shafft::FFTType precision,
                    MPI_Comm comm) noexcept;

// Create backend FFT plans for a configured ND plan.
// Must be called after planNDConfigure() and (for GPU) after setBuffers().
int planNDCreatePlans(shafft::detail::FFTNDPlan* plan) noexcept;

// Note: ND-specific destroy, getLayout, getAxes, getAllocSize, execute, normalize
// functions have been removed in favor of the unified PlanBase* functions below.

// Unified functions (work on both ND and 1D plans via PlanBase*)

// Destroy any plan (ND or 1D) via virtual dispatch
int destroy(shafft::detail::PlanBase** plan) noexcept;

// Execute any plan (ND or 1D) via virtual dispatch
int execute(shafft::detail::PlanBase* plan, shafft::FFTDirection direction) noexcept;

// Normalize any plan (ND or 1D) via virtual dispatch
int normalize(shafft::detail::PlanBase* plan) noexcept;

// Get allocation size for any plan
int getAllocSize(shafft::detail::PlanBase* plan, size_t* localAllocSize) noexcept;

// Get global FFT size (product of dimensions) for any plan
int getGlobalSize(shafft::detail::PlanBase* plan, size_t* globalSize) noexcept;

// Get layout for any plan
int getLayout(shafft::detail::PlanBase* plan,
              size_t* subsize,
              size_t* offset,
              shafft::TensorLayout layout) noexcept;

// Get axes for any plan
int getAxes(shafft::detail::PlanBase* plan, int* ca, int* da, shafft::TensorLayout layout) noexcept;

// Check if any plan is configured
int isConfigured(shafft::detail::PlanBase* plan, int* configured) noexcept;

// Check if any plan is active
int isActive(shafft::detail::PlanBase* plan, int* active) noexcept;

// Get precision of any plan
int getPrecision(shafft::detail::PlanBase* plan, shafft::FFTType* precision) noexcept;

// Set buffers for any plan
int setBuffers(shafft::detail::PlanBase* plan, void* data, void* work) noexcept;

// Get buffers from any plan
int getBuffers(shafft::detail::PlanBase* plan, void** data, void** work) noexcept;

#if SHAFFT_BACKEND_HIPFFT
// Set stream for any plan (GPU only)
int setStream(shafft::detail::PlanBase* plan, hipStream_t stream) noexcept;
#endif

// FFT1D (1-dimensional distributed FFT) functions

// FFT1DPlan is now defined in <shafft/detail/plan_1d_data.hpp>

// Query the layout for a 1D distributed FFT without creating a plan
int configuration1D(size_t globalN,
                    size_t* localN,
                    size_t* localStart,
                    size_t* localAllocSize,
                    shafft::FFTType precision,
                    MPI_Comm comm) noexcept;

// Create a new 1D plan (allocates FFT1DPlan)
int fft1dCreate(shafft::detail::FFT1DPlan** plan) noexcept;

// Configure 1D plan: compute layout and initialize Slab1D, but defer backend plan creation.
// After this call, allocSize() and layout queries are available.
int fft1dConfigure(shafft::detail::FFT1DPlan* plan,
                   size_t globalN,
                   size_t localN,
                   size_t localStart,
                   shafft::FFTType precision,
                   MPI_Comm comm) noexcept;

// Create backend FFT plans for a configured 1D plan.
// Must be called after fft1dConfigure().
int fft1dCreatePlans(shafft::detail::FFT1DPlan* plan) noexcept;

// Note: All other 1D-specific functions have been removed in favor of unified functions:
//   FFT1DDestroy    -> destroy(PlanBase**)
//   FFT1DSetBuffers -> setBuffers(PlanBase*, ...)
//   FFT1DGetBuffers -> getBuffers(PlanBase*, ...)
//   FFT1DExecute    -> execute(PlanBase*, ...)
//   FFT1DNormalize  -> normalize(PlanBase*)
//   FFT1DGetAllocSize -> getAllocSize(PlanBase*, ...)
//   FFT1DGetGlobalSize -> getGlobalSize(PlanBase*, ...)
//   FFT1DGetLayout  -> getLayout(PlanBase*, ...)
//   FFT1DIsInitialized -> isInitialized(PlanBase*, ...)
//   FFT1DIsActive   -> isActive(PlanBase*, ...)
//   FFT1DGetPrecision -> getPrecision(PlanBase*, ...)
//   FFT1DSetStream  -> setStream(PlanBase*, ...)

} // namespace shafft::detail

#endif // SHAFFT_INTERNAL_HPP
