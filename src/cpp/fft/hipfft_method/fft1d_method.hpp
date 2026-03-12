// Backend interface for distributed 1D FFT using hipFFT.
// Uses Cooley-Tukey factorization with 3 MPI all-to-all operations,
// or Bluestein algorithm for arbitrary sizes.

#ifndef SHAFFT_FFT1D_METHOD_HPP
#define SHAFFT_FFT1D_METHOD_HPP

#include "fft1d_handle.hpp"

#include <hipfft/hipfft.h>

namespace shafft {

int fft1dPlan(FFT1DHandle& handle,
              size_t globalN,
              FFTType precision,
              MPI_Comm comm,
              void* in,
              void* out,
              const FFT1DConfig& config = FFT1DConfig{});

int fft1dExecute(FFT1DHandle& handle, void* in, void* out, FFTDirection direction);

int fft1dDestroy(FFT1DHandle& handle);

int fft1dSetStream(FFT1DHandle& handle, hipStream_t stream);

int fft1dSynchronize(FFT1DHandle& handle);

int fft1dNormalize(FFT1DHandle& handle, void* data, int normExponent);

int fft1dQueryLayout(size_t globalN,
                     size_t& localN,
                     size_t& localStart,
                     size_t& localAllocSize,
                     MPI_Comm comm,
                     const FFT1DConfig& config = FFT1DConfig{});

inline int fft1dFinalize() noexcept {
  return 0;
}

} // namespace shafft

#endif // SHAFFT_FFT1D_METHOD_HPP
