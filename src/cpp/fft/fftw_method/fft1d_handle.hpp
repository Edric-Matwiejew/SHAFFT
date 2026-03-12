#ifndef SHAFFT_FFTW_FFT1D_HANDLE_HPP
#define SHAFFT_FFTW_FFT1D_HANDLE_HPP

#include <cstddef>
#include <fftw3.h>
#include <mpi.h>
#include <shafft/shafft_types.hpp>

namespace shafft {

// Internal state for FFTW-MPI 1D plans.
struct FFTW1DInternal {
  fftw_plan plan_fwd;     // Double precision forward.
  fftw_plan plan_bwd;     // Double precision backward.
  fftwf_plan plan_fwd_sp; // Single precision forward.
  fftwf_plan plan_bwd_sp; // Single precision backward.
  int nthreads;           // Number of threads used.
  unsigned flags;         // FFTW planner flags used.
};

// Handle for a distributed 1D FFT plan (FFTW backend).
struct FFT1DHandle {
  void* internal; // Pointer to FFTW1DInternal.
  size_t globalN; // Global FFT size.

  // INITIAL layout (input to forward FFT / output of backward FFT)
  size_t localNInit;     // Elements for this rank in initial layout.
  size_t localStartInit; // This rank's offset in initial layout.

  // REDISTRIBUTED layout (output of forward FFT / input to backward FFT)
  size_t localNTrans;     // Elements for this rank in transformed layout.
  size_t localStartTrans; // This rank's offset in transformed layout.

  size_t localAllocSize;         // Required allocation per buffer (max of both layouts).
  FFTType precision;             // FFT precision (C2C or Z2Z).
  MPI_Comm comm = MPI_COMM_NULL; // MPI communicator.
};

} // namespace shafft

#endif // SHAFFT_FFTW_FFT1D_HANDLE_HPP
