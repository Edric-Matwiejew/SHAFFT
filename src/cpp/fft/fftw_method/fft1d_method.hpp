#ifndef SHAFFT_FFTW_FFT1D_METHOD_HPP
#define SHAFFT_FFTW_FFT1D_METHOD_HPP

#include "fft1d_handle.hpp"
#include <cstddef>
#include <mpi.h>
#include <shafft/shafft_types.hpp>

namespace shafft {

int fft1dPlan(
    FFT1DHandle& handle, size_t globalN, FFTType precision, MPI_Comm comm, void* in, void* out);

int fft1dExecute(FFT1DHandle& handle, void* in, void* out, FFTDirection direction);

int fft1dDestroy(FFT1DHandle& handle);

int fft1dNormalize(FFT1DHandle& handle, void* data, int normExponent, size_t localCount);

int fft1dQueryLayout(
    size_t globalN, size_t& localN, size_t& localStart, size_t& localAllocSize,
    size_t& localNTrans, size_t& localStartTrans, MPI_Comm comm);

int fft1dFinalize() noexcept;

} // namespace shafft

#endif // SHAFFT_FFTW_FFT1D_METHOD_HPP
