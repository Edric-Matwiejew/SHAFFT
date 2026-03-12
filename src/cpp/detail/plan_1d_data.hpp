// plan_1d_data.hpp - Concrete implementation of PlanBase for 1D FFTs
//
// Holds state for 1D distributed FFT: backend handle, layout adapter, buffers.
// Users should prefer shafft::FFT1D RAII class.

#ifndef SHAFFT_DETAIL_PLAN_1D_DATA_HPP
#define SHAFFT_DETAIL_PLAN_1D_DATA_HPP

#include "plan_data_base.hpp"
#include "slab_1d.hpp"
#include <shafft/shafft_enums.hpp>

#include <mpi.h>

// Forward declaration - FFT1DHandle is defined in backend-specific headers
namespace shafft {
struct FFT1DHandle;
}

namespace shafft::detail {

// Concrete plan data for 1D distributed FFTs.
struct FFT1DPlan : public PlanBase {
  FFT1DHandle* handle = nullptr;    // Backend handle (owned, allocated in plan())
  Slab1D slab1d_;                   // Layout adapter for SlabBase interface
  void* stream_ = nullptr;          // Execution stream (opaque)
  FFTType precision = FFTType::C2C; // FFT precision
  bool configured = false;          // Whether configure() has been called
  bool initialized = false;         // Whether plan() has been called (backend plans created)

  // Stored parameters for deferred plan creation
  size_t globalN_ = 0;            // Global FFT size (set in configure)
  MPI_Comm comm_ = MPI_COMM_NULL; // Owned dup of the input communicator (freed on release).

  // PlanBase interface implementation
  SlabBase* slab() noexcept override { return configured ? &slab1d_ : nullptr; }
  const SlabBase* slab() const noexcept override { return configured ? &slab1d_ : nullptr; }
  bool isConfigured() const noexcept override { return configured; }
  bool isPlanned() const noexcept override { return initialized; }
  int createPlans() noexcept override;
  void setStream(void* stream) noexcept override { stream_ = stream; }
  void* getStream() const noexcept override { return stream_; }

  // Lifecycle and execution (implemented in shafft_internal.cpp)
  void release() noexcept override;
  int executeImpl(FFTDirection direction) noexcept override;
  int normalizeImpl() noexcept override;
};

} // namespace shafft::detail

#endif // SHAFFT_DETAIL_PLAN_1D_DATA_HPP
