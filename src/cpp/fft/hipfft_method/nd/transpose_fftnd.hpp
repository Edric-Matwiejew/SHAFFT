// Transpose-based FFT strategy using gpuTT.
// When the superbatch count exceeds a threshold, this module provides an
// alternative execution path: transpose data so FFT axes become trailing,
// execute FFT without superbatch iteration, then transpose back.
// This avoids precision degradation from repeated hipfftExec calls.
//
#ifndef SHAFFT_TRANSPOSE_FFTND_HPP
#define SHAFFT_TRANSPOSE_FFTND_HPP

#include "../fftnd_handle.hpp"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <mpi.h>

// Get the superbatch threshold for transpose-based FFT.
// When superbatch count exceeds this threshold, use transpose-FFT-transpose
// pattern instead of iterating over superbatches.
//
// Thresholds are per-FFT-rank, based on MI250X benchmarks (2026-02-03):
//   1D FFT: threshold = 16
//   2D FFT: threshold = 16
//   3D FFT: threshold = 1 (always use transpose)
//
// Can be overridden by environment variables:
//   SHAFFT_SUPERBATCH_THRESHOLD_1D (default: 16)
//   SHAFFT_SUPERBATCH_THRESHOLD_2D (default: 16)
//   SHAFFT_SUPERBATCH_THRESHOLD_3D (default: 1)
//   SHAFFT_SUPERBATCH_THRESHOLD (legacy, overrides all if set)
//
// Parameters:
//   fft_rank: number of FFT dimensions (1, 2, or 3)
long long getSuperbatchThreshold(int fft_rank);

// Reset the superbatch threshold cache (for testing only).
// After calling this, the next call to getSuperbatchThreshold() will re-read
// environment variables. This allows tests to verify default behavior even if
// other tests in the same binary have already initialized the cache.
void resetSuperbatchThresholdCache();

// Create transpose-based FFT plans for a non-trailing axis block.
// This is used when superbatch count exceeds the threshold.
//
// Strategy:
//   1. Transpose to bring FFT axes [minAxis..maxAxis] to trailing positions
//   2. Execute FFT on contiguous trailing axes (batch over remaining dims)
//   3. Transpose back to original layout
//
// Parameters:
//   ndim: total tensor rank
//   dimensions: tensor dimensions (row-major: dimensions[0] is slowest)
//   nta: number of FFT axes
//   ta: FFT axes (should be contiguous: [minAxis..minAxis+nta-1])
//   fft_type: HIPFFT_C2C or HIPFFT_Z2Z
//   stream: HIP stream for execution
//   comm: MPI communicator for diagnostics (may be MPI_COMM_NULL)
//   info: output TransposeFFTInfo structure
//
// Returns 0 on success, non-zero on error.
// If info.enabled is false on success, the caller should fall back to
// superbatch iteration (either because transpose is not needed, or gpuTT
// cannot handle the dimensions).
int createTransposeFFT(int ndim,
                       const int* dimensions,
                       int nta,
                       const int* ta,
                       hipfftType_t fft_type,
                       hipStream_t stream,
                       MPI_Comm comm,
                       TransposeFFTInfo& info);

// Destroy transpose-based FFT resources.
// Safe to call on uninitialized or partially initialized info.
void destroyTransposeFFT(TransposeFFTInfo& info);

#endif // SHAFFT_TRANSPOSE_FFTND_HPP
