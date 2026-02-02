#ifndef FFTHANDLE_H
#define FFTHANDLE_H

#include <shafft/shafft_types.hpp>

#include <fftw3.h>

/**
 * @brief FFT plan handle for the FFTW (CPU) backend.
 *
 * Mirrors the fields that the upstream SHAFFT code expects
 * while providing FFTW-specific plan storage.
 */
struct fftHandle {
  int nsubplans = 1;  ///< Must be odd so upstream swaps buffers once.

  // Unused by the FFTW backend, but present for ABI compatibility with the HIP backend.
  int* superbatches = nullptr;
  int* superbatches_offset = nullptr;

  // FFTW plans (both directions are planned at fftPlan time).
  fftw_plan fwd_dp = nullptr;   ///< Forward double-precision plan.
  fftw_plan bwd_dp = nullptr;   ///< Backward double-precision plan.
  fftwf_plan fwd_sp = nullptr;  ///< Forward single-precision plan.
  fftwf_plan bwd_sp = nullptr;  ///< Backward single-precision plan.

  // Metadata for optional diagnostics or validation.
  shafft::FFTType fft_type;       ///< Single (C2C) or double (Z2Z) precision.
  int ndim = 0;                   ///< Total number of dimensions.
  int nta = 0;                    ///< Rank of the contiguous transform block.
  int a = 0, b = -1;              ///< Range [a..b] of transform axes.
  int nthreads = 1;               ///< Number of CPU threads used by FFTW.
  unsigned flags = FFTW_MEASURE;  ///< FFTW planning flags.
};

#endif  // FFTHANDLE_H
