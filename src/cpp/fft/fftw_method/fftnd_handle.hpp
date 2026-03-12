#ifndef SHAFFT_FFTW_FFTND_HANDLE_HPP
#define SHAFFT_FFTW_FFTND_HANDLE_HPP

#include <shafft/shafft_types.hpp>

#include <fftw3.h>
#include <mpi.h>

// FFT plan handle for FFTW backend.
struct FFTNDHandle {
  int nsubplans = 1; // Must be odd so upstream swaps buffers once.

  // Unused by FFTW, present for ABI compatibility with HIP backend.
  int* superbatches = nullptr;
  int* superbatches_offset = nullptr;

  // FFTW plans (both directions planned at fftPlan time).
  fftw_plan fwd_dp = nullptr;  // Forward double-precision.
  fftw_plan bwd_dp = nullptr;  // Backward double-precision.
  fftwf_plan fwd_sp = nullptr; // Forward single-precision.
  fftwf_plan bwd_sp = nullptr; // Backward single-precision.

  // Metadata.
  shafft::FFTType fft_type;      // C2C or Z2Z.
  int ndim = 0;                  // Number of dimensions.
  int nta = 0;                   // Rank of contiguous transform block.
  int a = 0, b = -1;             // Range [a..b] of transform axes.
  int nthreads = 1;              // Number of CPU threads used by FFTW.
  unsigned flags = FFTW_MEASURE; // FFTW planning flags.

  // Communicator for diagnostics
  MPI_Comm comm = MPI_COMM_NULL;
};

#endif // SHAFFT_FFTW_FFTND_HANDLE_HPP
