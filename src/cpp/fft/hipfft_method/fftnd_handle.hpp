#ifndef FFTND_HANDLE_HPP
#define FFTND_HANDLE_HPP

#include <cstddef> // Required for NULL before gputt.h
#include <gputt.h>
#include <hipfft/hipfft.h>
#include <mpi.h>

// Transform method used for a subplan
enum class FFTMethod {
  STRIDED,   // Regular strided access with superbatch loop
  TRANSPOSE, // Transpose-FFT-transpose optimization via gpuTT
  TRAILING   // FFT on trailing axes (contiguous, no superbatch needed)
};

// Information for transpose-based FFT execution (used when superbatch threshold exceeded)
struct TransposeFFTInfo {
  gputtHandle transpose_to_front = nullptr; // Transpose FFT axes to trailing positions
  gputtHandle transpose_to_back = nullptr;  // Transpose back to original layout
  hipfftHandle transposed_fft = nullptr;    // FFT plan for transposed layout (contiguous trailing)
  int* transposedDims = nullptr;            // Dimension sizes after transpose
  int rank = 0;                             // Tensor rank for transpose plans
  bool enabled = false;                     // Whether to use transpose-based execution
};

struct FFTNDHandle {
  hipfftHandle* subplans = nullptr;
  int nsubplans = 0;
  // Outer loop (over leading dimensions):
  long long* superbatches = nullptr;        // Count of outer superbatch iterations
  long long* superbatches_offset = nullptr; // Element offset between outer superbatches
  // Inner loop (over trailing dimensions when batch=1):
  long long* innerbatches = nullptr; // Count of inner iterations (T, or 1 for trailing blocks)
  long long* innerbatches_offset =
      nullptr; // Element offset between inner iterations (1 for non-trailing)
  hipfftType_t fft_type;
  hipStream_t stream = nullptr;

  // Transpose-based FFT for high-superbatch subplans
  TransposeFFTInfo* transpose_info = nullptr; // Per-subplan transpose info (nullptr if not used)

  // Transform method used for each subplan (for diagnostics/benchmarking)
  FFTMethod* methods = nullptr; // Per-subplan method array

  // Communicator for diagnostics
  MPI_Comm comm = MPI_COMM_NULL;
};

#endif // FFTND_HANDLE_HPP
