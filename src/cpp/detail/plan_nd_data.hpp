// plan_nd_data.hpp - Concrete implementation of PlanBase for ND FFTs
//
// Holds state for N-dimensional distributed FFT: SlabND, backend subplans,
// normalization parameters. Users should prefer shafft::FFTND RAII class.

#ifndef SHAFFT_DETAIL_PLAN_ND_DATA_HPP
#define SHAFFT_DETAIL_PLAN_ND_DATA_HPP

#include "plan_data_base.hpp"
#include <mpi.h>
#include <shafft/shafft_types.hpp>

// Forward declarations to avoid circular includes
namespace shafft {
class SlabND;
}
struct FFTNDHandle;

namespace shafft::detail {

// Concrete plan data for ND distributed FFTs.
struct FFTNDPlan : public PlanBase {
  SlabND* slab_ = nullptr;          // Handles tensor decomposition and axes redistribution.
  int nsubplans = 0;                // Number of backend subplans.
  FFTNDHandle* subplans = nullptr;  // Array of backend subplans.
  FFTType fft_type = FFTType::C2C;  // FFT element/type (C2C or Z2Z).
  long double norm_denominator = 1; // Normalization denominator (scale = (1/den)^exp).
  MPI_Comm comm = MPI_COMM_NULL;    // Owned dup of the input communicator (freed on release).
  bool planned_ = false;            // Whether createPlans() has been called.

  // PlanBase interface implementation
  SlabBase* slab() noexcept override;
  const SlabBase* slab() const noexcept override;

  bool isConfigured() const noexcept override { return slab_ != nullptr; }
  bool isPlanned() const noexcept override { return planned_; }

  int createPlans() noexcept override;

  void setStream(void* stream) noexcept override { stream_ = stream; }

  void* getStream() const noexcept override { return stream_; }

  // Lifecycle and execution (implemented in shafft_internal.cpp)
  void release() noexcept override;
  int executeImpl(FFTDirection direction) noexcept override;
  int normalizeImpl() noexcept override;

private:
  void* stream_ = nullptr; // Execution stream (opaque)
};

} // namespace shafft::detail

#endif // SHAFFT_DETAIL_PLAN_ND_DATA_HPP
