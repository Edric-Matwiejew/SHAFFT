// transpose_fft.h
//
// Transpose-based FFT strategy using gpuTT.
// When the superbatch count exceeds a threshold, this module provides an
// alternative execution path: transpose data so FFT axes become trailing,
// execute FFT without superbatch iteration, then transpose back.
// This avoids precision degradation from repeated hipfftExec calls.
//
#ifndef TRANSPOSE_FFT_H
#define TRANSPOSE_FFT_H

#include "ffthandle.h"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

// Get the superbatch threshold for transpose-based FFT.
// When superbatch count exceeds this threshold, use transpose-FFT-transpose
// pattern instead of iterating over superbatches.
// Controlled by SHAFFT_SUPERBATCH_THRESHOLD environment variable (default: 16).
long long get_superbatch_threshold();

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
//   info: output TransposeFFTInfo structure
//
// Returns 0 on success, non-zero on error.
// If info.enabled is false on success, the caller should fall back to
// superbatch iteration (either because transpose is not needed, or gpuTT
// cannot handle the dimensions).
int create_transpose_fft(int ndim, const int* dimensions, int nta, const int* ta,
                         hipfftType_t fft_type, hipStream_t stream, TransposeFFTInfo& info);

// Destroy transpose-based FFT resources.
// Safe to call on uninitialized or partially initialized info.
void destroy_transpose_fft(TransposeFFTInfo& info);

#endif  // TRANSPOSE_FFT_H
