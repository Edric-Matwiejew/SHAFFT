// diag.hpp - Centralized diagnostics helpers for FFT backends
//
// Provides unified diagnostic output controlled by SHAFFT_FFT_DIAG environment variable.
// Tag format: [SHAFFT:<module>][r<rank>][<location>]
// Module examples: 1D-hipfft, ND-hipfft, ND-gpuTT, 1D-FFTW, ND-FFTW

#ifndef SHAFFT_FFT_COMMON_DIAG_HPP
#define SHAFFT_FFT_COMMON_DIAG_HPP

#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>

namespace shafft::fft {

/// Check if diagnostics are enabled via SHAFFT_FFT_DIAG=1 environment variable.
/// Result is cached after first call.
inline bool diagEnabled() noexcept {
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("SHAFFT_FFT_DIAG");
    cached = (env && env[0] == '1') ? 1 : 0;
  }
  return cached != 0;
}

/// Get MPI rank for the given communicator. Returns -1 on error.
inline int mpiRank(MPI_Comm comm) noexcept {
  int r = -1;
  MPI_Comm_rank(comm, &r);
  return r;
}

/// Print diagnostic header: [SHAFFT:<module>][r<rank>][<location>]
/// @param module Backend identifier (e.g., "1D-hipfft", "ND-FFTW", "ND-gpuTT")
/// @param comm MPI communicator for rank info
/// @param location Function or context name
inline void dbgHdr(const char* module, MPI_Comm comm, const char* location) noexcept {
  if (!diagEnabled())
    return;
  int r = mpiRank(comm);
  std::cerr << "[SHAFFT:" << module << "]"
            << (r >= 0 ? ("[r" + std::to_string(r) + "]") : std::string("[r?]")) << "[" << location
            << "] ";
}

/// Print array contents for debugging.
/// @param name Label for the array
/// @param arr Pointer to array data
/// @param n Number of elements
template <typename T>
void printArray(const char* name, const T* arr, int n) noexcept {
  if (!diagEnabled())
    return;
  std::cerr << name << "=[";
  for (int i = 0; i < n; ++i) {
    if (i)
      std::cerr << ",";
    std::cerr << arr[i];
  }
  std::cerr << "]";
}

} // namespace shafft::fft

#endif // SHAFFT_FFT_COMMON_DIAG_HPP
