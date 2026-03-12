// Handle and configuration types for distributed 1D FFT (hipFFT backend)

#ifndef SHAFFT_FFT1D_HANDLE_HPP
#define SHAFFT_FFT1D_HANDLE_HPP

#include <shafft/shafft_types.hpp>

#include <cstddef>
#include <hip/hip_runtime.h>
#include <mpi.h>

namespace shafft {

/// Configuration options for distributed 1D FFT path selection.
struct FFT1DConfig {
  /// Padding tolerance: use Path A (Cooley-Tukey) if N'/N <= tau.
  /// Default 1.25 means up to 25% padding overhead is acceptable.
  double tau = 1.25;

  /// If false, force Bluestein (Path B) when padding would change semantics.
  /// Set to false when exact N-point DFT frequencies are required.
  bool allowPadding = false;
};

/// Handle for distributed 1D FFT plan.
struct FFT1DHandle {
  void* internal; ///< Backend-specific internal data

  size_t globalN; ///< Global FFT size (before padding)

  // INITIAL layout (before transform)
  size_t localNInit;     ///< Elements for this rank
  size_t localStartInit; ///< This rank's offset

  // REDISTRIBUTED layout (after transform)
  size_t localNTrans;     ///< Elements for this rank
  size_t localStartTrans; ///< This rank's offset

  size_t localAllocSize; ///< Required allocation per buffer

  FFTType precision; ///< C2C or Z2Z
  MPI_Comm comm = MPI_COMM_NULL;
  hipStream_t stream = nullptr;
};

} // namespace shafft

#endif // SHAFFT_FFT1D_HANDLE_HPP
