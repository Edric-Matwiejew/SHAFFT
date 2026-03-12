#ifndef FFTND_METHOD_HPP
#define FFTND_METHOD_HPP

#include <shafft/shafft_types.hpp>

#include "fftnd_handle.hpp"

#include <cstdio>
#include <cstdlib>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

int fftndPlan(FFTNDHandle& plan,
              int nta,
              int* ta,
              int ndim,
              int* dimensions,
              shafft::FFTType precision,
              void* in,
              void* out);

int fftndSetStream(FFTNDHandle& plan, hipStream_t stream);

int fftndExecute(FFTNDHandle plan, void*& data, void*& work, shafft::FFTDirection direction);

int fftndDestroy(FFTNDHandle plan);

inline int fftndFinalize() noexcept {
  return 0;
}

#endif // FFTND_METHOD_HPP
