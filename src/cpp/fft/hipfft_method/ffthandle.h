#ifndef FFT_HANDLE_H
#define FFT_HANDLE_H
#include <gputt.h>
#include <hipfft/hipfft.h>

// Information for transpose-based FFT execution (used when superbatch threshold exceeded)
struct TransposeFFTInfo {
  gputtHandle transpose_to_front = nullptr;  // Transpose FFT axes to trailing positions
  gputtHandle transpose_to_back = nullptr;   // Transpose back to original layout
  hipfftHandle transposed_fft = nullptr;     // FFT plan for transposed layout (contiguous trailing)
  int* transposed_dims = nullptr;            // Dimension sizes after transpose
  int rank = 0;                              // Tensor rank for transpose plans
  bool enabled = false;                      // Whether to use transpose-based execution
};

struct fftHandle {
  hipfftHandle* subplans = nullptr;
  int nsubplans = 0;
  // Outer loop (over leading dimensions):
  long long* superbatches = nullptr;         // Count of outer superbatch iterations
  long long* superbatches_offset = nullptr;  // Element offset between outer superbatches
  // Inner loop (over trailing dimensions when batch=1):
  long long* innerbatches = nullptr;  // Count of inner iterations (T, or 1 for trailing blocks)
  long long* innerbatches_offset =
      nullptr;  // Element offset between inner iterations (1 for non-trailing)
  hipfftType_t fft_type;
  hipStream_t stream = nullptr;

  // Transpose-based FFT for high-superbatch subplans
  TransposeFFTInfo* transpose_info = nullptr;  // Per-subplan transpose info (nullptr if not used)
};
#endif  // FFT_HANDLE_H
