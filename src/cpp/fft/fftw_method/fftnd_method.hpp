#ifndef SHAFFT_FFTW_FFTND_METHOD_HPP
#define SHAFFT_FFTW_FFTND_METHOD_HPP

#include <shafft/shafft_types.hpp>

#include "fftnd_handle.hpp"

int fftndPlan(FFTNDHandle& plan,
              int nta,
              int* ta,
              int ndim,
              int* dimensions,
              shafft::FFTType precision,
              void* in,
              void* out);

int fftndExecute(FFTNDHandle plan, void*& data, void*& work, shafft::FFTDirection direction);

int fftndDestroy(FFTNDHandle plan);

int fftndFinalize() noexcept;

#endif // SHAFFT_FFTW_FFTND_METHOD_HPP
